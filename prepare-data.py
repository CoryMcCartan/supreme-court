# Preprocess SCDB data, adding new variables, etc.
# Generates (in data/ folder):
#   - votes.csv
#   - justice_lookup.json
#   - scdb_justices.hdf5
#   - scdb_cases.hdf5

import json

from tqdm import tqdm
import pandas as pd
import numpy as np
import patsy
import tables

from scores import calculate as calculate_scores
from cases import calculate as calculate_case_points

RESPONDENT = 0
PETITIONER = 1
LIBERAL = 2
CONSERVATIVE = 1

def prepare_data():
    # Load the justice-centered SCDB.
    print("Loading SCDB.")
    scdb_j = pd.read_csv("data/scdb_justices.csv")
    scdb_c = pd.read_csv("data/scdb_cases.csv")

    print("Adding indexing variables.")
    add_indexing(scdb_j)
    add_indexing(scdb_c)

    print("Extracting justice votes.")
    vote_matrix = make_vote_matrix(scdb_j)
    justice_lookup = make_lookup_table(scdb_j)

    print("Calculating justice scores.")
    scores = calculate_scores(vote_matrix)

    print("Calculating case points.")
    case_points = calculate_case_points(scores, vote_matrix, scdb_j)

    print("Adding feature variables.")
    add_variables(scdb_j, scores, case_points)
    add_variables(scdb_c)

    print("Converting features.")
    scdb_j_features = prep_variables(scdb_j)
    drop_variables(scdb_j)
    drop_variables(scdb_c)

    print("Saving data.")
    vote_matrix.to_csv("data/votes.csv")
    with open("data/justice_lookup.json", "w+") as f:
        json.dump(justice_lookup, f)

    scdb_j.to_hdf("data/scdb_justices.hdf5", "Justice_SCDB", mode="w", complib="zlib")
    scdb_c.to_hdf("data/scdb_cases.hdf5", "Case_SCDB", mode="w", complib="zlib")
    scdb_j_features.to_hdf("data/justice_features.hdf5", "Justice_SCDB_Features", mode="w", complib="zlib")


def add_indexing(scdb):
    """Create 0-indexed codes for justices, terms, natural courts, and cases."""
    global J, T, K, N

    if "justice" in scdb:
        scdb["justice_n"] = scdb.justice.astype("category").cat.codes
        J = scdb.justice.nunique()
        T = scdb.term.nunique()
        K = scdb.caseId.nunique()
        N = len(scdb)

    scdb["term_n"] = scdb.term - min(scdb.term)
    scdb["nc_n"] = scdb.naturalCourt.astype("category").cat.codes
    scdb["case_no"] = scdb.caseId.astype("category").cat.codes

    # std. decision, per curiam, equally divided, or judgement of court
    # clear affirm or reverse
    scdb["analyzeable"] = ((scdb.decisionType.isin([1, 2, 6, 7])) 
            & (scdb.caseDisposition.isin([2, 3, 4, 5, 8, 9])))

def add_variables(scdb, scores=None, case_points=None):
    """Feature engineering."""
    # Guess liberal side
    #scdb["predLiberalSide"] = RESPONDENT
    #for k in tqdm(range(K)):
    #    pick_liberal(scdb, k)

    def get_month(d):
        try: return d.month
        except: return -1

    scdb.dateArgument = pd.to_datetime(scdb.dateArgument, format="%m/%d/%Y")
    scdb.dateRearg = pd.to_datetime(scdb.dateRearg, format="%m/%d/%Y")
    scdb.dateDecision = pd.to_datetime(scdb.dateDecision, format="%m/%d/%Y")

    scdb["argumentMonth"] = scdb.dateArgument.apply(get_month)
    scdb["reargumentMonth"] = scdb.dateRearg.apply(get_month)
    scdb["decisionMonth"] = scdb.dateDecision.apply(get_month)
    scdb["delay"] = ((scdb.dateDecision - scdb.dateArgument)
        .apply(lambda x: x / np.timedelta64(1, "W"))
        .fillna(-1)
        .astype(int))

    scdb["petitionerGroup"] = scdb.petitioner.apply(get_party_group)
    scdb["respondentGroup"] = scdb.respondent.apply(get_party_group)

    scdb["sourceCourt"] = scdb.caseSource.apply(get_court_circuit)
    scdb["originCourt"] = scdb.caseOrigin.apply(get_court_circuit)

    scdb["court_vote"] = scdb.caseDisposition.isin([3,4,5,8]).astype(int) 
    scdb["liberalSide"] = (scdb.decisionDirection - 1 == scdb.court_vote)
    scdb["unanimous"] = (scdb.minVotes == 0)
    scdb.decisionDirection -= 1
    scdb.lcDispositionDirection -= 1

    if "justice" in scdb:
        scdb["dissent"] = scdb.vote.isin([2, 6, 7])
        scdb["voteSide"] = scdb.vote.isin([1, 3, 4, 5])
        scdb.voteSide = (scdb.voteSide == scdb.court_vote).astype(float)
        scdb.voteSide[np.isnan(scdb.vote)] = np.nan
        scdb.direction -= 1

        calculate_justice_rates(scdb)
        add_score_features(scdb, scores)
        add_case_point_features(scdb, case_points)

def calculate_justice_rates(scdb):
    """Calcuate absolute and relative rates of dissent, ideology, and reversal."""
    print("Calculating justice rates.")

    t_start = scdb.term.min()
    t_end = scdb.term.max() + 1

    # create columns
    scdb["justice_reverse_abs"] = 0
    scdb["justice_reverse_diff"] = 0
    scdb["justice_direction_abs"] = 0
    scdb["justice_direction_diff"] = 0
    scdb["justice_dissent_abs"] = 0
    scdb["justice_dissent_diff"] = 0
    scdb["justice_lower_diff"] = 0

    window = 10
    def extract(arr, default):
        return (lambda j: arr[j] if j in arr else default)
        
    for t in tqdm(range(t_start, t_end)):
        # use only earlier terms
        cases = scdb[(scdb.term < t) & (t - scdb.term <= window)]
        reverse_abs = cases.groupby("justice_n").voteSide.mean()
        reverse_diff = reverse_abs - cases.partyWinning.mean()
        direction_abs = cases.groupby("justice_n").direction.mean()
        direction_diff = reverse_abs - cases.decisionDirection.mean()
        dissent_abs = cases.groupby("justice_n").dissent.mean()
        dissent_diff = reverse_abs - cases.dissent.mean()
        
        # then apply to current terms
        cases = scdb.term == t
        justices = scdb.loc[cases, "justice_n"]
        scdb.loc[cases, "justice_reverse_abs"] = justices.apply(extract(reverse_abs, 0.5))
        scdb.loc[cases, "justice_reverse_diff"] = justices.apply(extract(reverse_diff, 0.0))
        scdb.loc[cases, "justice_direction_abs"] = justices.apply(extract(direction_abs, 0.5))
        scdb.loc[cases, "justice_direction_diff"] = justices.apply(extract(direction_diff, 0.0))
        scdb.loc[cases, "justice_dissent_abs"] = justices.apply(extract(dissent_abs, 0.5))
        scdb.loc[cases, "justice_dissent_diff"] = justices.apply(extract(dissent_diff, 0.0))
        scdb.loc[cases, "justice_lower_diff"] = (scdb[cases].justice_direction_abs 
                                             - scdb[cases].lcDispositionDirection)

def add_score_features(scdb, scores):
    """Add absolute and relative justice scores to database.""" 
    print("Adding score features.")
    court_scores = np.nanmedian(scores, axis=1) # median scores by term
    rel_scores = scores - court_scores[:, None, :]

    def extract(arr, i):
        return (lambda row: arr[row.term_n - 1, row.justice_n, i])

    tqdm.pandas()
    scdb.loc[:, "score1"] = scdb.progress_apply(extract(scores, 0), axis=1)
    scdb.loc[:, "score2"] = scdb.progress_apply(extract(scores, 1), axis=1)
    scdb.loc[:, "score1_rel"] = scdb.progress_apply(extract(rel_scores, 0), axis=1)
    scdb.loc[:, "score2_rel"] = scdb.progress_apply(extract(rel_scores, 1), axis=1)

def add_case_point_features(scdb, case_points):
    """Add affirm/reverse case points to database.""" 
    print("Adding case point features.")

    idx = scdb[scdb.analyzeable].caseId.astype("category").cat.codes
    def extract(i, j):
        return (lambda row: case_points[idx[row.name], i, j] if row.name in idx else np.nan)

    tqdm.pandas()
    scdb.loc[:, "reverse1"] = scdb.progress_apply(extract(1, 0), axis=1)
    scdb.loc[:, "reverse2"] = scdb.progress_apply(extract(1, 1), axis=1)
    scdb.loc[:, "affirm1"] = scdb.progress_apply(extract(2, 0), axis=1)
    scdb.loc[:, "affirm2"] = scdb.progress_apply(extract(2, 1), axis=1)


def prep_variables(scdb):
    """Select variable and binarize categorical ones."""
    # fill NA
    data = scdb.copy()
    data.loc[:, ["score1", "score2", "score1_rel", "score2_rel"]].fillna(0, inplace=True)
    data.loc[:, ["reverse1", "reverse2", "affirm1", "affirm2"]].fillna(0, inplace=True)
    data.fillna(-1, inplace=True)

    df = patsy.dmatrix("""0 + justice_n + C(justice_n)
            + C(argumentMonth) + C(reargumentMonth) + C(decisionMonth) + delay
            + C(petitioner) + C(petitionerGroup) + C(respondent) + C(respondentGroup)
            + C(jurisdiction) + C(adminAction) + C(sourceCourt) + C(originCourt)
            + lcDisagreement + C(certReason) + C(lcDisposition) 
            + C(lcDispositionDirection) + C(issue) + C(issueArea) 
            + justice_reverse_abs + justice_reverse_diff
            + justice_direction_abs + justice_direction_diff
            + justice_dissent_abs + justice_dissent_diff + justice_lower_diff
            + score1 + score2 + score1_rel + score2_rel
            + reverse1 + reverse2 + affirm1 + affirm2
            """, data, return_type="dataframe")
    return df

def drop_variables(scdb):
    scdb.drop(["docketId", "caseIssuesId", "voteId", "sctCite", "ledCite",
        "lexisCite", "lawSupp", "lawMinor"], axis=1, inplace=True)
    scdb.usCite.fillna("", inplace=True)
    scdb.docket.fillna("", inplace=True)

    return scdb

def make_vote_matrix(scdb):
    """Turn justice-centered data in to a K-by-J matrix of affirm/reverse votes.
        Limited to cases  that can be classified into a clear 'affirm' or 'reverse.'"""
    scdb = scdb.loc[scdb.analyzeable, :]

    # Code each case to "affirm" or "reverse" using the caseDisposition and vote variables.
    # 1: reverse,  0: affirm
        # This should equal the partyWinning variable, but it often doesn't.
    scdb.loc[:, "court_vote"] = scdb.caseDisposition.isin([3,4,5,8]).astype(int) 
    # vote coded as majority/dissent, so we conver to affirm/reverse
    scdb.loc[:, "vote"] = scdb.vote.isin([1,3,4,5]).astype(int) 
    scdb.vote = (scdb.vote == scdb.partyWinning).astype(int) 

    # organize by case
    case_nos = scdb.case_no.unique()
    K = len(case_nos)
    votes = np.zeros((K, J), bool) 
    mask = np.zeros((K, J), bool) 
    terms = np.empty(K, int)
    decision = np.empty(K, int)
    cid = np.empty(K, object)
    # for every case extract the justice votes
    for k, case_no in enumerate(tqdm(case_nos)):
        rows = scdb[scdb.case_no == case_no]
        justices = rows.justice_n.values
        votes[k, justices] = rows.vote.values
        mask[k, justices] = 1

        terms[k] = rows.iloc[0].term
        decision[k] = rows.iloc[0].partyWinning
        cid[k] = rows.iloc[0].caseId

    # collect into a data frame
    cols = [f"justice_{j}" for j in range(J)]
    data = pd.DataFrame(votes, columns=cols)
    data.loc[:, "term"] = terms
    data.loc[:, "decision"] = decision
    data.loc[:, "id"] = cid
    data.set_index("id", inplace=True)

    # replace masked values with NA
    for j in range(J):
        data.iloc[~mask[:, j], j] = None

    return data


def make_lookup_table(scdb):
    """Make a table for converting justice names to IDs/indexes"""
    justices = scdb[["justiceName", "justice_n"]].drop_duplicates()
    justices = justices.set_index("justiceName")
    justices = justices.sort_values("justice_n")
    justices = list(justices.index.values)

    return justices


def get_party_group(p):
    """Group respondent/petitioner codes into smaller groups."""
    dmap = {1:2, 2:5, 3:4, 4:5, 5:4, 6:13, 7:0, 8:11, 9:11, 10:11, 11:11,
    12:0, 13:11, 14:0, 15:0, 16:13, 17:0, 18:4, 19:0, 20:13, 21:4, 22:0, 23:0, 24:0,
    25:8, 26:0, 27:2, 28:5, 100:3, 101:1, 102:0, 103:1, 104:1, 105:1, 106:9, 107:0,
    108:0, 109:1, 110:3, 111:0, 112:0, 113:1, 114:0, 115:1, 116:1, 117:1, 118:1,
    119:1, 120:1, 121:1, 122:1, 123:1, 124:1, 125:1, 126:3, 127:0, 128:1, 129:0,
    130:0, 131:0, 132:1, 133:1, 134:14, 135:0, 136:0, 137:3, 138:0, 139:1, 140:0,
    141:1, 142:0, 143:1, 144:1, 145:11, 146:0, 147:1, 148:1, 149:0, 150:14, 151:0,
    152:1, 153:0, 154:11, 155:0, 156:0, 157:1, 158:1, 159:0, 160:1, 161:1, 162:0,
    163:0, 164:0, 165:1, 166:0, 167:1, 168:0, 169:0, 170:6, 171:1, 172:0, 173:0,
    174:0, 175:0, 176:0, 177:0, 178:1, 179:0, 180:0, 181:1, 182:11, 183:11, 184:1,
    185:1, 186:0, 187:1, 188:0, 189:1, 190:1, 191:1, 192:1, 193:9, 194:1, 195:0,
    196:0, 197:0, 198:1, 199:0, 200:3, 201:0, 202:0, 203:0, 204:0, 205:1, 206:0,
    207:14, 208:0, 209:1, 210:0, 211:0, 212:0, 213:3, 214:0, 215:3, 216:0, 217:3,
    218:14, 219:1, 220:1, 221:0, 222:0, 223:14, 224:0, 225:0, 226:0, 227:0, 228:1,
    229:0, 230:0, 231:1, 232:0, 233:1, 234:1, 235:1, 236:0, 237:0, 238:1, 239:0,
    240:8, 241:0, 242:0, 243:1, 244:0, 245:1, 246:1, 247:12, 248:0, 249:12, 250:0,
    251:0, 252:0, 253:0, 254:0, 255:0, 256:0, 257:0, 258:2, 259:1, 301:2, 302:2,
    303:2, 304:2, 305:2, 306:2, 307:2, 308:2, 309:2, 310:2, 311:2, 312:2, 313:2,
    314:2, 315:2, 316:2, 317:2, 318:2, 319:2, 320:2, 321:2, 322:2, 323:2, 324:2,
    325:2, 326:2, 327:2, 328:2, 329:2, 330:2, 331:2, 332:2, 333:2, 334:2, 335:2,
    336:2, 337:2, 338:2, 339:2, 340:2, 341:2, 342:2, 343:2, 344:2, 345:2, 346:2,
    347:2, 348:2, 349:2, 350:2, 351:2, 352:2, 353:2, 354:2, 355:2, 356:2, 357:2,
    358:2, 359:2, 360:2, 361:2, 362:2, 363:2, 364:2, 366:2, 367:2, 368:2, 369:2,
    370:2, 371:2, 372:2, 373:2, 374:2, 375:2, 376:2, 377:2, 378:2, 379:2, 380:2,
    381:2, 382:2, 383:2, 384:2, 385:2, 386:2, 387:2, 388:2, 389:2, 390:2, 391:2,
    392:2, 393:2, 394:2, 395:2, 396:2, 397:2, 398:2, 399:2, 400:2, 401:2, 402:2,
    403:2, 404:2, 405:2, 406:2, 407:2, 408:2, 409:2, 410:2, 411:2, 412:2, 413:2,
    414:2, 415:2, 416:2, 417:2, 501:0, 600:0}

    if p in dmap:
        return dmap[int(p)]
    else:
        return 0

def get_court_circuit(c):
    """Convert a court variable to the circuit it's on."""
    dmap = {1: 13,
            2: 13, 3: 13, 4: 14, 5: 14, 6: 13, 7: 13, 8: 13,
            9: 22, 10: 99, 12: 9, 13: 99, 14: 13, 15: 99, 16: 99,
            17: 99, 18: 99, 19: 0, 20: 22, 21: 1, 22: 2, 23: 3,
            24: 4, 25: 5, 26: 6, 27: 7, 28: 8, 29: 9, 30: 10,
            31: 11, 32: 12, 41: 11, 42: 11, 43: 11, 44: 9, 45: 9,
            46: 8, 47: 8, 48: 9, 49: 9, 50: 9, 51: 9, 52: 10, 53: 2,
            54: 3, 55: 12, 56: 11, 57: 11, 58: 11, 59: 11, 60: 11,
            61: 11, 62: 9, 63: 9, 64: 9, 65: 7, 66: 7, 67: 7, 68: 7,
            69: 7, 70: 8, 71: 8, 72: 10, 73: 6, 74: 6, 75: 5, 76: 5,
            77: 5, 78: 1, 79: 4, 80: 1, 81: 6, 82: 6, 83: 8, 84: 5,
            85: 5, 86: 8, 87: 8, 88: 9, 89: 8, 90: 9, 91: 1, 92: 3,
            93: 10, 94: 2, 95: 2, 96: 2, 97: 2, 98: 4, 99: 4, 100: 4,
            101: 8, 102: 9, 103: 6, 104: 6, 105: 10, 106: 10, 107: 10,
            108: 9, 109: 3, 110: 3, 111: 3, 112: 1, 113: 1, 114: 4,
            115: 8, 116: 6, 117: 6, 118: 6, 119: 5, 120: 5, 121: 5,
            122: 5, 123: 10, 124: 2, 125: 3, 126: 4, 127: 4, 128: 9,
            129: 9, 130: 4, 131: 4, 132: 7, 133: 7, 134: 10, 150: 5,
            151: 9, 152: 4, 153: 7, 155: 4, 160: 4, 162: 11, 163: 5,
            164: 11, 165: 7, 166: 7, 167: 8, 168: 6, 169: 5, 170: 8,
            171: 3, 172: 3, 173: 2, 174: 4, 175: 6, 176: 3, 177: 3,
            178: 5, 179: 4, 180: 4, 181: 7, 182: 6, 183: 3, 184: 9,
            185: 11, 186: 8, 187: 5, 300: 0, 301: 0, 302: 0, 400: 99,
            401: 99, 402: 99, 403: 11, 404: 8, 405: 9, 406: 2, 407: 3,
            408: 11, 409: 11, 410: 7, 411: 7, 412: 8, 413: 10, 414: 6,
            415: 5, 416: 1, 417: 4, 418: 1, 419: 6, 420: 8,
            421: 5, 422: 8, 423: 9, 424: 1, 425: 3, 426: 2,
            427: 4, 428: 6, 429: 9, 430: 3, 431: 1, 432: 4, 433: 6,
            434: 5, 435: 2, 436: 4, 437: 4, 438: 7,
            439: 10, 440: 12, 441: 8, 442: 10, 443: 9}

    if c in dmap:
        return dmap[int(c)]
    else:
        return 0



def pick_liberal(scdb, k):
    """Use SCDB rules to guess which side (petitioner/respondent) is the 'liberal' side."""
    case = scdb[scdb.case_no == k].iloc[0]
    area = case.issueArea
    issue = case.issue
    petr = case.petitioner
    resp = case.respondent
    
    # "Sympathetic" party -- minorities, native american groups, children, etc
    sympathetic = [100, 126, 129, 137, 175, 170, 212, 213, 222, 223, 
                   224, 332, 10, 11, 182, 183, 256, 319, 106, 142, 
                   154, 140, 145, 193, 162, 208, 215, 110, 138, 114,
                   136]
    # Business party -- banks, industry, etc. (excl. small business)
    business = [101, 122, 133, 151, 216, 238, 113, 139, 122, 135,
                148, 157, 158, 171, 187, 198, 205, 209, 220, 243, 
                245, 259, 231, 228, 235, 246]
    if area <= 6:
        if petr in sympathetic:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        if resp in sympathetic:
            scdb.loc[scdb.case_no == k, "liberalSide"] = RESPONDENT # override
        elif resp in [130, 257]:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr in [155, 165, 183, 9, 154, 204, 188] and issue == 50020: # abortion
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr in [19, 111] and issue != 20400: # non-liability, pro-attorney and gov't official
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif resp == 27 and 40010 <= issue <= 40060: # against gov't in due process
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif resp == 27 and issue == 30040 or issue == 80140: # against gov't in privacy and corruption
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr == 27 and resp == 28: # favor gov't over states
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr in [8, 9, 10, 11, 13, 145, 154] and area == 3: # free speech, favor employees
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
    # unions and economic activity
    elif area == 7 or area == 8:
        classA = [174, 208, 212, 213, 138, 114, 170, 150, 134]
        classB = [2, 3, 4, 5, 6, 7, 18, 19, 21, 26, 27, 28]
        if petr == 249 and issue != 70020 and issue != 70030: # pro-union except antitrust
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr in classA or petr > 300:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr in classB and resp not in classA and resp < 300: # prioritize class A
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif resp in business or (resp == 173 and petr not in business):
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr == 123 and issue == 80105: # land claims
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif resp == 172 and 80180 <= issue <= 80210: # patents
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr == 172 and 80180 <= issue <= 80210: # patents
            scdb.loc[scdb.case_no == k, "liberalSide"] = RESPONDENT
        elif ((petr == 117 and resp == 220) or (petr == 160 and resp == 161)
                or (petr == 144 and resp == 143)): # small vs big business
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
    # judicial power
    elif area == 9:
        if petr in [6, 16]:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif resp > 300 and issue == 90120: # judicial review
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr in sympathetic:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif resp in business:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
    # federalism
    elif area == 10:
        if petr == 27:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr > 300 and issue in [10080, 130015]: # congress or state vs gov't
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif resp == 28:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr in [100, 126, 136, 137]:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
    # taxes
    elif area == 12:
        if petr == 27:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
    # private law
    elif area == 14:
        scdb.loc[scdb.case_no == k, "liberalSide"] = np.nan # unspecifiable
    # misc., interstate relations
    else:
        if petr in [6,16] and 90320 <= issue <= 90390:
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER
        elif petr > 300 and issue in [10080, 130010, 130015]: # gov't agency
            scdb.loc[scdb.case_no == k, "liberalSide"] = PETITIONER


if __name__ == "__main__":
    prepare_data()
