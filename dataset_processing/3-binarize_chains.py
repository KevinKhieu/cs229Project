#####################################################################
# binarize_chains.py
# ---------------------
# Laura Cruz-Albrecht | Kevin Khieu
# CS229 Project
# 
# adds label field to chain object: 0 if time diff < 1/2 hour, else 1
#####################################################################

from __future__ import division
import numpy as np
import pickle
import sys
import csv
from datetime import datetime, timedelta
import re
from operator import itemgetter

csv.field_size_limit(sys.maxsize)

def getTimeDifference(date2, date1):
    '''
    return difference in minutes between date2 and date 1
    '''
    return (date2['standard_date'] - date1['standard_date']).total_seconds() / 60.0

def main():

    # list of (orig email, response email, time difference)
    chains = None
    with open("../pickles/pickled_chains.txt", "rb") as fp:   # Unpickling
        chains = pickle.load(fp)

    print 'num chains: %d' % len(chains)

    # shuffle dataset
    np.random.shuffle(chains)

    hour_counts = [0] * 25  # 0 through 24

    # reduced list of chains:
    # - all emails in bucket 0 (round to 1 hour)  (1820)
    # - equal amount from other buckets randomly chosen within (0-24) (1820)
    label0_cnt = 0
    label1_cnt = 0

    MAX_CNT = 1810

    reduced_chains = []
    for elem in chains:
        e1, e2, time_diff = elem 

        hour_diff = int(np.round(time_diff / 60.0)) # bc original time diff in mins

        # only consider hours between 0 and 24
        if hour_diff > 24: continue

        # get boolean label: 0 (rounds to < 1 hour); 1 (rounds to 1+ hour)
        label = 0 if hour_diff == 0 else 1

        # cap examples for each label to 1820
        if label0_cnt == MAX_CNT and label1_cnt == MAX_CNT: break

        # label 0, and still need more
        if label == 0 and label0_cnt < MAX_CNT:
            reduced_chains += [(e1, e2, time_diff, label)]
            label0_cnt += 1
            hour_counts[hour_diff] += 1

        # label 1, and still need more
        elif label == 1 and label1_cnt < MAX_CNT:
            reduced_chains += [(e1, e2, time_diff, label)]
            label1_cnt += 1
            hour_counts[hour_diff] += 1

        
    print str(hour_counts)
    print str(len(reduced_chains))
    print 'label 0 count', hour_counts[0]
    print 'label 1 count', sum(hour_counts[1:])

    # pickle reduced_chains object
    with open("../pickles/balanced_chains.pickle", "wb") as fp:
        pickle.dump(reduced_chains, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()