# Multidimensional Scaling (MDS) in N dimensions for Supreme Court Justices
# Generates data/scores_#d.npy

import sys
from itertools import combinations
import warnings
import json

import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
from tqdm import tqdm

def calculate(votes=None, n_dim=2, window=4, span=None, predictive=True, plot=False, format="npy"):
    if not 1 <= n_dim <= 2:
        raise Exception(f"Can only calculate scores in 1 or 2 dimensions, not {n_dim}.")
    if span is None:
        span = 32 if n_dim == 2 else 8

    justices = pd.read_csv("data/justices.csv")

    SCALIA = justices[justices.lastName == "Scalia"].index[0]
    GINSBURG = justices[justices.lastName == "Ginsburg"].index[0]
    BREYER = justices[justices.lastName == "Breyer"].index[0]

    # load votes matrix from file if not passed
    if not isinstance(votes, np.ndarray):
        votes = pd.read_csv("data/votes.csv")

    J = votes.shape[1] - 3  # number of justices
    K = votes.shape[0]      # number of cases
    # terms
    t_start = votes.term.min()
    t_end = votes.term.max() + 1 # +1 for ranges, etc
    terms = range(t_start, t_end)
    T = t_end - t_start    

    # terms in which each justice started and ended service
    start_terms = np.full(J, -1, dtype=int)
    end_terms = np.full(J, -1, dtype=int)

    scores = np.full((T, J, n_dim), np.nan)

    mds = MDS(n_components=n_dim, dissimilarity="precomputed", max_iter=5000, 
            eps=1e-8, n_init=100 * n_dim**2)

    # for each term
    for t in tqdm(terms):
        # get votes from this term, then slice of 'row' and 'term' columns
        t_votes = votes[votes.term == t].iloc[:, 1:-2].as_matrix()
        # remove unanimous cases
        t_votes = t_votes[np.nansum(np.isfinite(t_votes), axis=1) 
                            != np.nansum(t_votes, axis=1)]
        # pare down to justices who were active
        active = (~np.isnan(t_votes)).any(axis=0)
        n_active = sum(active)
        if predictive:
            t_votes = (votes[(votes.term - t <= window) & (votes.term - t >= 0)]
                        .iloc[:, [False, *active, False, False]].as_matrix())
        else:
            t_votes = (votes[abs(votes.term - t) <= window]
                        .iloc[:, [False, *active, False, False]].as_matrix())

        start_terms[(start_terms == -1) & active] = t - t_start
        end_terms[active] = t - t_start

        disagreement_table = np.full((n_active, n_active), np.nan)
        disagreement_table = np.zeros((n_active, n_active))

        # for each pair of justcies
        for j1, j2 in combinations(range(n_active), 2):
            j_votes =  t_votes[:, (j1, j2)]
            # cases where both justices voted
            common = (~np.isnan(j_votes)).all(axis=1)
            if not common.any(): 
                continue
            disagree = np.logical_xor(j_votes[common, 0], j_votes[common, 1])
            disagreement_rate = disagree.mean()
            # store rate in symmetric matrix
            disagreement_table[j1, j2] = disagreement_rate
            disagreement_table[j2, j1] = disagreement_rate
            disagreement_table[j1, j1] = 0
            disagreement_table[j2, j2] = 0

        if n_dim == 1:
            distances = (6*disagreement_table**5 - 15*disagreement_table**4
                            + 10*disagreement_table**3)
        else:
            distances = disagreement_table

        if t == t_start:
            proj = mds.fit_transform(distances)
        else:
            last = scores[t-t_start-1, active]
            last[np.isnan(last)] = 0
            with warnings.catch_warnings(): # ignore warning about manual init
                warnings.simplefilter("ignore")
                proj = mds.fit_transform(distances, init=last)
            if n_dim == 2:
                #proj[:,0] /= proj[:,0].std()
                #proj[:,1] /= proj[:,1].std()
                pass
            
        scores[t-t_start, active] = proj


    # align Scalia-Ginsburg
    if n_dim == 2:
        v1 = scores[-1, SCALIA] - scores[-1, GINSBURG]
        v2 = [v1[1], -v1[0]]
        transf = np.array([v1, v2]).T

        scores = scores.reshape((-1, 2))
        scores = scores.dot(transf)
        scores = scores.reshape((T, J, 2))

        # Ensure Breyer on upper side
        scores[:, :, 1] *= np.sign(scores[-1, BREYER, 1])
    # Ensure Ginsburg on left
    scores[:, :, 0] *= -np.sign(scores[-1, GINSBURG, 0])

    # Scaling: in S.D. units in each dimensino
    # Smoothing: Exponentially weighted moving average
    for n in range(n_dim):
        scores[:, :, n] /= np.nanstd(scores[:, :, n])
        scores[:, :, n] = pd.DataFrame(scores[:, :, n]).ewm(span=span).mean().as_matrix()

    # write NaNs over terms where justices aren't active
    for j in range(J):
        start = start_terms[j]
        end = end_terms[j]
        scores[:start, j] = np.nan
        scores[end+1:, j] = np.nan

    if format == "npy":
        np.save(f"computed/scores_{n_dim}d.npy", scores)
    elif format == "csv" and n_dim == 2:
        # combine into complex array
        cmplx_scores = np.empty((J, T), dtype=complex)
        cmplx_scores.real = scores[:, :, 0].T
        cmplx_scores.imag = scores[:, :, 1].T
        # create output dataframe
        data = pd.DataFrame(cmplx_scores, columns=terms)
        data.loc[:, "firstName"] = justices.firstName
        data.loc[:, "lastName"] = justices.lastName
        data.loc[:, "seat"] = justices.seat
        data.loc[:, "id"] = range(J)
        # convert to long form
        data = pd.melt(data, id_vars=["id", "lastName", "firstName", "seat"], 
                value_vars=terms, var_name="term", value_name="score1")
        # extract scores from complex numbers
        data.loc[:, "score2"] = data.score1.imag
        data.loc[:, "score1"] = data.score1.real
        # sort, trim, and output
        data.sort_values(["id", "term"], inplace=True)
        data.dropna(inplace=True)
        data.to_csv(f"computed/scores_2d.csv", index=False)
    else:
        raise Exception(f"Output file format not recognized: '{format}'")


    # plot
    if not plot: 
        return scores

    colors = ["#000000", "#f1c40f", "#e67e22", "#c0392b", "#95a5a6", "#455a6e", 
            "#8e44ad", "#3498db", "#27ae60"]

    fig = plt.figure()
    for j in range(J):
        start = max(start_terms[j], 1950 - t_start) # don't plot earlier than 1950
        end = end_terms[j] + 1
        score = scores[start:end, j]
        if len(score) == 0: continue

        name = justices.iloc[j].lastName
        if n_dim == 1:
            plt.plot(np.arange(start + t_start, end + t_start), score[:, 0],
                    lw=3, c=colors[justices.iloc[j].seat])
            plt.text(end + t_start - 0.5, score[-1, 0], name, fontsize=8,
                    va="center", ha="left")
            plt.xlim(1950, t_end)
            plt.xlabel("Term")
            plt.ylabel("Score")
        elif n_dim == 2:
            alph = 1 if end == T  else 0.2
            color = colors[justices.iloc[j].seat]
            plt.plot(score[:, 0], score[:, 1], lw=3, c=color, alpha=alph)
            plt.text(score[-1, 0], score[-1, 1], name, fontsize=8, va="center",
                    ha="left")
            plt.scatter([score[-1, 0]], [score[-1, 1]], s=32, c=color, alpha=alph)
            plt.xlabel("Score 1")
            plt.ylabel("Score 2")

    plt.title("Justice Scores over Time")
    plt.show(block=False)
    input("Press <enter> to close.")
    plt.close()

    return scores


if __name__ == "__main__":
    n_dim = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    calculate(n_dim=n_dim, plot=True, predictive=True, format="csv")
