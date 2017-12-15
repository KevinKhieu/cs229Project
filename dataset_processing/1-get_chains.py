#####################################################################
# get_chains.py
# -------------
# CS229 Project
# 
# Extracts email chain pairs from the chains objects
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
    

def isResponseEmail(e2, e1):
    '''
    returns whether e2 was a response to e1
    assumes time of e2 is after time of e1
    '''
    return e2['sender'] in e1['to'] and e1['sender'] in e2['to']


def main():

    subject_emails = None
    with open("../pickles/pickled_subject_emails.txt", "rb") as fp:   # Unpickling
        subject_emails = pickle.load(fp)

    print 'num subjects with 2+ emails: %d' % len(subject_emails)

    chains = [] # list of (orig email, response email, time difference)

    for subject in subject_emails:
        emails = subject_emails[subject]
        #emails = sorted(emails, key=itemgetter('standard_date'))

        emails = sorted(emails, key = lambda d: d['date']['standard_date'])

        n = len(emails)

        # case 1: 2 emails - check second is response to first
        if n == 2:
            
            e1 = emails[0]
            e2 = emails[1]

            if isResponseEmail(e2, e1):
                time_diff = getTimeDifference(e2['date'], e1['date'])
                chains += [(e1, e2, time_diff)]

        # case 2: more than 2 emails
        # for simplicity, we look at the list of time-ordered emails and see if 
        # (1, 2), (2, 3), ... (n-1, n) are in response to each other (ie, n responded to n-1)
        # and not in response to any other email
        else:
            
            num_sequence_matches = 0
            num_nonseq = 0
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if i == j: continue

                    e1 = emails[i]
                    e2 = emails[j]

                    if isResponseEmail(e2, e1):
                        if j == i + 1:
                            num_sequence_matches += 1
                        else:
                            num_nonseq += 1

            if num_sequence_matches == n - 1 and num_nonseq == 0:
                # add the consecutive pairs in the chain to the 'chains' list
                for i in range(n - 1):
                    e1 = emails[i]
                    e2 = emails[i + 1]
                    time_diff = getTimeDifference(e2['date'], e1['date'])
                    chains += [(e1, e2, time_diff)]
        
    # print stats
    print 'num chains: %d' % len(chains)

    # sanity check output
    '''
    for i in range(3):
        e1, e2, time_diff = chains[i]
        print '---------------------'
        print e1['date'], e1['sender'], e1['to'], e1['is_re']
        print str(e1['date']['standard_date'])
        #print e1['body']
        print
        print e2['date'], e2['sender'], e2['to'], e2['is_re']
        print str(e2['date']['standard_date'])
        #print e2['body']
        print
        print str(time_diff)
        print 
    '''

    # pickle chains object
    with open("../pickles/pickled_chains.txt", "wb") as fp:
        pickle.dump(chains, fp, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    main()