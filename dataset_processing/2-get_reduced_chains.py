#####################################################################
# get_reduced_chains.py
# ---------------------
# Laura Cruz-Albrecht | Kevin Khieu
# CS229 Project
# 
# Caps number of chains in each hourly bucket to 50
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

    hour_counts = [0] * 25  # 0 through 24

    # reduced list of chains: cap each hour bucket to 50 examples
    reduced_chains = []
    for elem in chains:
        e1, e2, time_diff = elem 

        hour_diff = int(np.round(time_diff / 60.0)) # original time diff in mins
        #hour_diff = int(time_diff / 60.0) 

        # only consider hours between 0 and 24
        if hour_diff > 24: continue

        # cap examples for each hour bucket to 50
        if hour_counts[hour_diff] >= 50: continue

        hour_counts[hour_diff] += 1
        reduced_chains += [elem]

    print str(hour_counts)
    print str(len(reduced_chains))

    # pickle reduced_chains object
    with open("../pickles/pickled_reduced_chains.txt", "wb") as fp:
        pickle.dump(reduced_chains, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()