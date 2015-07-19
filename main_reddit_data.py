import sys
import praw
import pickle
import get_submissions
import setup_db
from time import time, sleep

def Limit(i):
	'''Rate limits'''

	if i==1:
		sleep(0.4)
	if i ==2:
		sleep(0.99)


def CommentsToDB(sid):
	'''Get submission by id
		 if not retrieved properly add it to not retrieved stack
		 otherwise parse the comment object and store in db '''

	not_retrieved = []
	message = ''

	db = info_db.RedditDB()
	db.connect()
	db.create_CommentInfo()
	db.create_SubmissionInfo()

	for s in sid:
		Limit(2)
		comments = submissions.ParseBySubmissionId(s)
		if not comments['success']:
			not_retrieved.append( (s, comments['content']) )
			message = "SubmissionID %s not retrieved. Error %s" % (
											s, comments['content'])
		else:
			message = "SubmissionID %s retrieved" % s
			subObj = [ c for c in comments['content'] if c.comment_id == s]
			try:
				assert len(subObj) == 1
				subObj[0].num_comments = len(comments['content'])

				db.insert_SubmissionInfo( subObj )
				db.insert_CommentInfo( comments['content'] )
				message = message + " and entered into DB. "
				message = message + "Fetched %s comments" % subObj[0].num_comments

			except AssertionError:
				not_retrieved.append( (s, "No unique sid"))
				break
		db.commit()
		print(message)
	db.close()
	return not_retrieved


def MatchingThreads(keywords, sub = 'bitcoin', pickl = True):
	'''Function to search reddit using praw for threads 
		matching the keywords list'''

	reddit = praw.Reddit('dummy')
	subreddit = reddit.get_subreddit(sub)

	submission_id = set([])
	sortby = ['relevance', 'comments', 'top', 'hot', 'new']

	print("Fetching threads matching keywords...")
	start_time = time()

	for keyword in keywords:
		for s in sortby:
			#res = subreddit.search(keyword, limit=1000, sort=s)
			res = subreddit.search(keyword, limit=200, sort=s)
			sid = set( [r.id for r in res] )
			submission_id = submission_id | sid
			Limit(1)
	if pickl:
		with open("submission_id.pkl", 'wb') as f:
			pickle.dump(submission_id, f)

	tot_time = (time() - start_time)/60.
	print("%s threads found. Total time = %s seconds" % (
							len(submission_id), tot_time))
	return submission_id


def GetKeywords(path=None):
	'''Get Keywords from file'''

	if not path:
		path = 'keywords.txt'
	with open(path, 'r') as f:
		keywords = f.readlines()
	keywords = set(keywords)
	return keywords


def GetSubmissionIDs():
	'''Get submission ids from file'''

	with open('submission_id.pkl', 'rb') as f:
		submission_id = pickle.load(f)
	return submission_id


def main(get_new_sid = False):
	if get_new_sid:
		keywords = GetKeywords()
		sid = MatchingThreads(keywords)
	else:
		print("Getting submission ids from file")
		sid = GetSubmissionIDs()
	CommentsToDB(sid)


if __name__ == '__main__':
	if len(sys.argv) > 1:
		new_sid = bool(int(sys.argv[1]))
		main(new_sid)
	else:
		main()



