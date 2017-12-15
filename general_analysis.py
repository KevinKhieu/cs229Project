#####################################################################
# general_analysis.py
# -------------------
# CS229 Project
# 
# Performs general analysis. Looks at:
# - median response time (mins) vs hour of day
# - median response time (mins) vs day of week
#
# for feature extraction investigation
#####################################################################

from __future__ import division
import numpy as np
import pickle
import sys
import csv
from datetime import datetime, timedelta
import re
from operator import itemgetter
import matplotlib.pyplot as plt

csv.field_size_limit(sys.maxsize)

# list of (orig email, response email, time difference in mins)
chains = None
with open("../pickles/pickled_chains.txt", "rb") as fp:   # Unpickling
    chains = pickle.load(fp)

def hour_weekday_analysis():
    '''
    investigate:
    - median and mean response time (mins) vs hour of day
    - median and mean response time (mins) vs day of week
    '''
    hour_of_day_response = {}   # hour email sent: [list of response times]
    day_of_week_response = {}   # day of week email sent: [list of response times]

    for elem in chains:
        e1, e2, time_diff = elem
        sent_time = e1['date']['local_date']

        time_diff = int(np.round(time_diff / 60.0)) # bc original time diff in mins

        sent_hour = sent_time.hour
        sent_day = sent_time.weekday()  # 0: monday, 6: sunday

        # update hour_of_day_response
        if sent_hour not in hour_of_day_response:
            hour_of_day_response[sent_hour] = [time_diff]
        else:
            hour_of_day_response[sent_hour] += [time_diff]

        # update day_of_week_response
        if sent_day not in day_of_week_response:
            day_of_week_response[sent_day] = [time_diff]
        else:
            day_of_week_response[sent_day] += [time_diff]


    # compute median times for each bucket
    hour_of_day_response_medians = [np.median(hour_of_day_response[hour]) for hour in range(24)]
    day_of_week_response_medians = [np.median(day_of_week_response[weekday]) for weekday in range(7)]

    # print statistics
    # ---------------
    # 1. response time vs. hour of day
    print 'time (mins) vs hour of day'
    print '------------------------------------------'
    print 'hour\tmedian response time\tsample size'
    for hr in range(24):
        med = hour_of_day_response_medians[hr]
        print '%d\t%d\t%d' % (hr, med, len(hour_of_day_response[hr]))
    
    print

    # 2. response time vs. day of week
    print 'response time (mins) vs day of week'
    print '------------------------------------------'
    print 'weekday (0: monday)\tmedian response time\tsample size'
    for weekday in range(7):
        med = day_of_week_response_medians[weekday]
        print '%d\t%d\t%d' % (weekday, med, len(day_of_week_response[weekday]))

    # plot statistics
    # ---------------
    # 1. response time vs. hour of day
    plt.title('Response time vs hour of day')
    plt.xlabel('hour of day')
    plt.ylabel('median response time (minutes)')
    plt.plot(range(24), hour_of_day_response_medians)
    plt.legend()
    plt.show()

    # 2. response time vs. day of week
    plt.title('Response time vs day of week')
    plt.xlabel('day of week')
    plt.ylabel('median response time (minutes)')
    plt.plot(range(7), day_of_week_response_medians)
    plt.xticks(range(7), ['M', 'T', 'W', 'Th', 'F', 'Sa', 'Su'])    
    plt.legend()
    plt.show() 

def hour_buckets():
    hour_counts = {}
    min_counts = {}
    
    for elem in chains:
        _, _, time_diff = elem

        response_mins = int(time_diff) 
        response_hours = int(np.round(time_diff / 60.0)) 

        # update hour_counts
        if response_hours not in hour_counts:
            hour_counts[response_hours] = 1
        else:
            hour_counts[response_hours] += 1

        # update min_counts
        if response_mins not in min_counts:
            min_counts[response_mins] = 1
        else:
            min_counts[response_mins] += 1

    resp_hour_counts = [(resp_hours, hour_counts[resp_hours]) for resp_hours in hour_counts]
    resp_hour_counts.sort()

    resp_min_counts = [(resp_mins, min_counts[resp_mins]) for resp_mins in min_counts]
    resp_min_counts.sort()

    for elem in resp_hour_counts:
        resp_hour, cnt = elem
        print '', resp_hour, '\t', cnt

    first_25_sum = sum([elem[1] for elem in resp_hour_counts[:25]])
    print '0-24 sum: %d' % first_25_sum
    print 'frac of all: %f' % (first_25_sum / float(len(chains)))

    first_73_sum = sum([elem[1] for elem in resp_hour_counts[:73]])
    print '0-72 sum: %d' % first_73_sum
    print 'frac of all: %f' % (first_73_sum / float(len(chains)))

    plt.title('Response time (hours) vs count')
    plt.xlabel('response time (hours)')
    plt.ylabel('# emails with this response time')
    X = [elem[0] for elem in resp_hour_counts]
    Y = [elem[1] for elem in resp_hour_counts]
    plt.loglog(X, Y)
    plt.legend()
    plt.show() 

    plt.title('Response time (minutes) vs count')
    plt.xlabel('response time (minutes)')
    plt.ylabel('# emails with this response time')
    X = [elem[0] for elem in resp_min_counts]
    Y = [elem[1] for elem in resp_min_counts]
    plt.loglog(X, Y)
    plt.legend()
    plt.show() 


def main():
    hour_weekday_analysis()

    #hour_buckets()

    '''
    subject_length()    # number of words; number of chars

    q_in_subject()      # count occurences of q in subject

    q_in_body()         # count occurences of q in body

    keywords()          # look at present of keywords in subject and/or body
    '''
    

if __name__ == '__main__':
    main()