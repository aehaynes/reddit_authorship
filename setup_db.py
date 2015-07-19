import sqlite3

start_date_cutoff = '2014-06-01'
end_date_cutoff = '2015-06-09'

query = {
'training cid': '''SELECT CommentId FROM (
						SELECT CommentId, Author, (datetime( time, 'unixepoch')) as DT 
						FROM commentinfo where ((DT < '%s' OR DT > '%s') AND Author != '[deleted]') ) as T
						GROUP BY CommentId''' % (start_date_cutoff, end_date_cutoff),

'training sid': '''SELECT SubmissionId FROM (
						SELECT SubmissionId, Author, (datetime( time, 'unixepoch')) as DT 
						FROM commentinfo where ((DT < '%s' OR DT > '%s') AND Author != '[deleted]') ) as T
						GROUP BY SubmissionId''' % (start_date_cutoff, end_date_cutoff),

'training cid 2': '''SELECT CommentId FROM (
						SELECT CommentId, Author, (datetime( time, 'unixepoch')) as DT 
						FROM commentinfo where (DT < '%s'  AND Author != '[deleted]') ) as T
						GROUP BY CommentId''' % (start_date_cutoff),

'training sid 2': '''SELECT SubmissionId FROM (
						SELECT SubmissionId, Author, (datetime( time, 'unixepoch')) as DT 
						FROM commentinfo where (DT < '%s' AND Author != '[deleted]') ) as T
						GROUP BY SubmissionId''' % (start_date_cutoff),

'unseen sid':	'''SELECT SubmissionId FROM(
						SELECT SubmissionId, (datetime( time, 'unixepoch')) as DT 
						FROM commentinfo where (DT >= '%s' AND DT <= '%s' AND Author != '[deleted]') ) as T
						GROUP BY SubmissionId''' % (start_date_cutoff, end_date_cutoff),

'unseen cid':	'''SELECT CommentId FROM(
						SELECT CommentId, (datetime( time, 'unixepoch')) as DT 
						FROM commentinfo where (DT >= '%s' AND DT <= '%s' AND Author != '[deleted]') ) as T
						GROUP BY CommentId''' % (start_date_cutoff, end_date_cutoff),

'full sid':		''' SELECT SubmissionId	FROM commentinfo WHERE author !='[deleted]' GROUP BY SubmissionId''',

'full cid':		''' SELECT CommentId FROM commentinfo WHERE author !='[deleted]' GROUP BY CommentId''',
}

'''Template for CommentInfo table entries'''
class CommentInfo:
	TableName = 'CommentInfo'
	ColNames  = [	'CommentId', 
					'SubmissionId', 
					'ParentId', 
					'Author', 
					'Upvotes',
					'Downvotes',
					'Comment',
					'Time']


'''Template for SubmissionInfo table entries'''
class SubmissionInfo:
	TableName = 'SubmissionInfo'
	ColNames  = [	'SubmissionId',
					'Title', 
					'Author', 
					'Upvotes',
					'Downvotes',
					'Num_comments',
					'Time']	
					

'''Template for UserInfo table entries'''
class UserInfo:
	TableName = 'SubmissionInfo'
	ColNames  = [	'SubmissionId', 
					'Author', 
					'Upvotes',
					'Downvotes',
					'Num_comments',
					'Time']	


'''RedditDB class'''
class RedditDB:
	__dbpath = ''
	
	def __init__(self):
		self.__dbpath = 'redditDB.sqlite'

	def connect(self):
		self.__conn = sqlite3.connect(	self.__dbpath, 
										detect_types=sqlite3.PARSE_DECLTYPES)
		self.db = self.__conn.cursor()

	def commit(self):
		self.__conn.commit()

	def close(self):
		self.__conn.close()

	def create_CommentInfo(self):
		querystr = "CREATE TABLE IF NOT EXISTS {tn} \
					({c0} PRIMARY KEY, \
					{c1}, {c2}, {c3}, {c4}, {c5}, {c6}, {c7} timestamp)"
		querystr =  querystr.format( 	tn=CommentInfo.TableName, 
										c0=CommentInfo.ColNames[0],
										c1=CommentInfo.ColNames[1],
										c2=CommentInfo.ColNames[2],
										c3=CommentInfo.ColNames[3],
										c4=CommentInfo.ColNames[4],
										c5=CommentInfo.ColNames[5],
										c6=CommentInfo.ColNames[6],
										c7=CommentInfo.ColNames[7])
		self.db.execute(querystr)
		
	def create_SubmissionInfo(self):
		querystr = "CREATE TABLE IF NOT EXISTS {tn} \
					({c0} PRIMARY KEY, \
					{c1}, {c2}, {c3}, {c4}, {c5}, {c6} timestamp)"
		querystr =   querystr.format( 	tn=SubmissionInfo.TableName, 
										c0=SubmissionInfo.ColNames[0],
										c1=SubmissionInfo.ColNames[1],
										c2=SubmissionInfo.ColNames[2],
										c3=SubmissionInfo.ColNames[3],
										c4=SubmissionInfo.ColNames[4],
										c5=SubmissionInfo.ColNames[5],
										c6=SubmissionInfo.ColNames[6],)
		self.db.execute(querystr)
	
	def insert_CommentInfo(self, obj_list):
		values = []
		for obj in obj_list:
			val = list([getattr(obj, 'comment_id'),
						getattr(obj, 'submission_id'),
						getattr(obj, 'parent_id'),
						getattr(obj, 'author'),
						getattr(obj, 'upvotes'),
						getattr(obj, 'downvotes'),
						getattr(obj, 'body'),
						getattr(obj, 'timestamp') ])
			values.append(val)
		table = CommentInfo.TableName
		columns = ', '.join(CommentInfo.ColNames)
		placeholders = ', '.join('?' * len(CommentInfo.ColNames))
		querystr = 'INSERT OR IGNORE INTO {} ({}) VALUES ({})'.format(
						table, columns, placeholders)
		self.db.executemany(querystr, values)

	def insert_SubmissionInfo(self, obj_list):			
		values = []
		for obj in obj_list:
			val = list([getattr(obj, 'submission_id'),
						getattr(obj, 'title'),
						getattr(obj, 'author'),
						getattr(obj, 'upvotes'),
						getattr(obj, 'downvotes'),
						getattr(obj, 'num_comments'),
						getattr(obj, 'timestamp') ])
			values.append(val)
		table = SubmissionInfo.TableName
		columns = ', '.join(SubmissionInfo.ColNames)
		placeholders = ', '.join('?' * len(SubmissionInfo.ColNames))
		querystr = 'INSERT OR IGNORE INTO {} ({}) VALUES ({})'.format(
						table, columns, placeholders)
		self.db.executemany(querystr, values)

