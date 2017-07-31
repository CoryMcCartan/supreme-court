# Calculate cutting points for each case in the database, based on 2D scores.
# Generates computed/cuts.npy

import math
import json

import pandas as pd
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from tqdm import tqdm


def calculate(scores=None, votes=None, scdb=None):
    global J, K, T

    if not isinstance(scores, np.ndarray):
        scores = np.load("computed/scores_2d.npy")
    if not isinstance(votes, np.ndarray):
        votes = pd.read_csv("data/votes.csv")
    if not isinstance(votes, pd.DataFrame):
        scdb = pd.read_hdf("data/scdb_justices.hdf5")

    # load justice names
    with open("data/justice_lookup.json", "r") as f:
        justices = np.array(json.load(f))

    J = scores.shape[1]  # number of justices
    K = votes.shape[0]   # number of cases
    # terms
    t_start = votes.term.min()
    t_end = votes.term.max() + 1 # +1 for ranges, etc
    T = t_end - t_start    

    svm = SVC(C=100, kernel="linear")

    # each case has 3 vectors: cutline, reverse, affirm
    case_points = np.empty((K, 3, 2)) 

    # for each case
    for k in tqdm(range(K)):
        case = votes.iloc[k, :]
        term = int(case.term - t_start)
        case_vote = case.as_matrix()[1:-2].astype(float)

        active = ~np.isnan(case_vote)
        n_active = sum(active)

        case_vote = case_vote[active]
        case_scores = scores[term, active]

        if sum(case_vote) == n_active: # unanimous reverse
            cutline, reverse, affirm = calc_unanimous(scdb, case_scores, case.id, active, True)
        elif sum(case_vote) == 0: # unanimous affirm
            cutline, reverse, affirm = calc_unanimous(scdb, case_scores, case.id, active, False)
        else:
            cutline, reverse, affirm = calc_normal(case_vote, case_scores, svm)

        case_points[k] = np.array([cutline, reverse, affirm])

    np.save("computed/case_points.npy", case_points)

    return case_points


def calc_normal(case_vote, case_scores, svm):
    fit = svm.fit(case_scores, case_vote)
    sv = np.array([fit.coef_[0, 1], -fit.coef_[0, 0]])
    intercept = -fit.intercept_ / sv[0]    
    sv /= np.linalg.norm(sv)

    slope = sv[1] / sv[0]
    if np.isnan(slope):
        slope = 1e10  

    cutline = [slope, intercept]

    # we  missed the fit
    if abs(intercept) > 5:
        return calc_nofit(case_scores)

    intercept = np.array([[0, intercept]])
    vectors = case_scores - intercept # vector from y-int to justice pts
    proj = (vectors.dot(sv) * sv[:, None]).T # proj of vectors onto SV
    diff = vectors - proj # perp. vector from SV to justice pts
    dist = np.hypot(diff[:, 0], diff[:, 1]) # dist from SV to justice pts
    
    midpt = proj.mean(axis=0) + intercept # midpt of SV
    perp = np.array([-sv[1], sv[0]]) # perp vector to SV

    reverse = midpt[0] + perp
    affirm = midpt[0] - perp

    return cutline, reverse, affirm


def calc_nofit(case_scores):
    # assume centrism
    cutline = [0, 0]
    reverse = np.mean(case_scores, axis=0)
    affirm = reverse
    return cutline, reverse, affirm


def calc_unanimous(scdb, case_scores, case_id, active, reverse): 
    j_id = scdb[scdb.caseId == case_id].iloc[0].majOpinWriter
    if np.isnan(j_id): # per curiam, usually
        maj_score = case_scores.mean(axis = 0)
        maj_side = 1 # assume practicality
    else:
        j = scdb[scdb.justice == j_id].iloc[0].justice_n
        j = list(np.where(active)[0]).index(j)
        maj_score = case_scores[j]
        maj_side = np.sign(maj_score[1]) # which side of 2nd dimension the majority is on

    max_side = np.max(maj_side * case_scores[:, 1])
    intercept = max_side * 1.1 # add 10% past most extreme 2nd dim. justice
    
    x = case_scores[:, 0].mean()
    cutline = [0, intercept]
    
    if reverse:
        reverse = [x, maj_score[1]]
        affirm = [x, 2*intercept - maj_score[1]]
    else:
        affirm = [x, maj_score[1]]
        reverse =[x, 2*intercept - maj_score[1]]

    return cutline, reverse, affirm


if __name__ == "__main__":
    calculate()
