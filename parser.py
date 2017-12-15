#####################################################################
# parser.py
# ---------
# Laura Cruz-Albrecht | Kevin Khieu
# CS229 Project
#
# parses the enron dataset (emails.csv). Creates 2 pickled objects:
# 1. user_emails
#    map of users to lists of all emails they sent
#    format:    user: [email1, email2]
#
# 2. subject_emails
#    map of email subject lines to list of emails with that subject 
#    note: subject name filtered to remove preceeding 're:' if present
#    format:    subject: [email1, email2]
######################################################################

from __future__ import division
import numpy as np
import pickle
import sys
import csv
from datetime import datetime, timedelta
import re

csv.field_size_limit(sys.maxsize)

def getContent(field, msg):
    start_i = msg.index(field)
    end_i = msg.index('\n', start_i)
    content = msg[start_i + len(field) : end_i].strip().lower()
    return content

def getMonths():
    months = {}
    months['jan'] = 1
    months['feb'] = 2
    months['mar'] = 3
    months['apr'] = 4
    months['may'] = 5
    months['jun'] = 6
    months['jul'] = 7
    months['aug'] = 8
    months['sep'] = 9
    months['oct'] = 10
    months['nov'] = 11
    months['dec'] = 12
    return months

months = getMonths()


date_format = r'((\w+), (\d+) (\w+) (\d+) (\d+):(\d+):(\d+) -(\d+))'
def getDatetime(str):
    regex = re.search(date_format,str)
    month = months[regex.group(4)]
    offset = -1 * int(regex.group(9)) / 100
    curr = datetime(int(regex.group(5)), month, int(regex.group(3)), int(regex.group(6)), int(regex.group(7)), int(regex.group(8)))
    return curr + timedelta(hours=offset)

def main():

    # user: [emails]
    user_emails = {}

    # subject: [emails with that subject]
    subject_emails = {}

    i = 0
    tot = 0
    with open('emails.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)  # skip the headers
        for row in reader:
            print str(i)

            file = row[0]
            msg = row[1]
            
            # Date
            raw_date = getContent('Date: ', msg)
            date = getDatetime(raw_date)

            # Subject
            subject = getContent('Subject: ', msg)
            if len(subject) == 0 or subject == 're:': continue

            is_re = False
            if subject[:4] == 're: ':
                subject = subject[4:]
                if len(subject) == 0: continue
                is_re = True

            # From: email
            sender = getContent('From: ', msg)

            # To: email(s)
            raw_to = getContent('To: ', msg)
            if len(raw_to) == 0: continue
            if raw_to[-1] == ',': raw_to = raw_to[: -1]     # remove trailing comma
            to = raw_to.split(', ')

            # CC
            ccs = getContent('X-cc: ', msg)
            cc_count = 0
            if len(ccs) > 0: cc_count = len(ccs.split(', '))

            # BCC
            bccs = getContent('X-bcc: ', msg)
            bcc_count = 0
            if len(bccs) > 0: bcc_count = len(bccs.split(', '))

            # Body 
            start_i = msg.index('\n\n')
            body = msg[start_i + 2:].strip().lower()
            body = body.replace('\n', ' ').replace('\t', ' ')

            email_obj = {
                'date': date,
                'subject': subject,
                'sender': sender,
                'is_re': is_re,
                'to': to,
                'cc_count': cc_count,
                'bcc_count': bcc_count,
                'body': body
            }

            # add email to user_emails dictionary
            if sender not in user_emails:
                user_emails[sender] = [email_obj]
            else:
                user_emails[sender] += [email_obj]

            # add email to subject_emails dictionary
            if subject not in subject_emails:
                subject_emails[subject] = [email_obj]
            else:
                subject_emails[subject] += [email_obj]

            i += 1

    # filter subject_emails object to only keep subject-emails with > 1 email
    filtered_subject_emails = {}
    for subject in subject_emails:
        emails = subject_emails[subject]
        if len(emails) > 1:
            filtered_subject_emails[subject] = emails

    subject_emails = filtered_subject_emails


    # pickle user_emails object
    with open("pickled_user_emails.txt", "wb") as fp:
        pickle.dump(user_emails, fp, protocol=pickle.HIGHEST_PROTOCOL)

    # pickle subject_emails object
    with open("pickled_subject_emails.txt", "wb") as fp:
        pickle.dump(subject_emails, fp, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()