import praw
import requests
import json
#import lxml
#import html
#import traceback
#import sys
#import sqlite3

def ParseBySubmissionId(sid, subreddit = 'bitcoin'):
	''' Takes a reddit submission id, gets the json of the 
		corresponding thread and returns a list of 
		comment objects in the thread'''

	#Custom header so reddit doesnt respond with 429 http errorcode.
	# - See https://github.com/reddit/reddit/wiki/API
	headers = {'User-Agent': 
				'/r/bitcoin crawler for dope shit v0.02 by /u/sinnycal'}
	link = "http://reddit.com/r/%s/comments/%s/.json?" % (subreddit, sid)
	qualifiers = {'limit': '500', 'sort': 'top'}

	comments = {'success': False, 'content': None}

	for key,value in qualifiers.items():
		link = link + '%s=%s&' %(key, value)
	link = link.strip('&')

	submission = requests.get(link, headers = headers)
	if submission.status_code == 200:
		tree = json.loads( submission.text )
		comments['success'] = True
		comments['content'] = ParseComments(tree)
	else:
		comments['success'] = False
		comments['content'] = [submission.status_code]

	return comments


def ParseComments(tree):
	''' Parses the json/tree of comments and comment attributes
		to a list of comment objects'''

	comments = []
	num_nodes = len(tree)

	#Index nodes on tree that have comments:
	for j in range(num_nodes):
		subtree =  list([ tree[j] ])
		index = ParseJson.IndexTree(subtree, (0,), (0,) )

		for i in index:
			node = ParseJson.NodeByPos(subtree, i)
			
			Comment = Submission()
			Comment.SetAttr( ParseJson.InfoAtNode(node) )
			#Append comment if exists
			if Comment.author:
				comments.append(Comment)

	return comments


#----

'''Class for Comment objects'''
class Submission:
	'''
	Goal: Get a list of comments for each thread:
	[Comment_1, ..., Comment_N]

	Where Comment_i has the following attributes:
	- submission_id: 	reddit submission id
	- comment_id: 		id for comment in question (= submission_id if original post comment)
	- parent_id: 		comment_id of parent comment (= submission_id if top level comment 
																	i.e. not a nested comment)
	- author: 			username of comment author
	- timestamp: 		utc timestamp of when comment was made
	- has_replies: 		True if replies, False otherwise
	- upvotes:			number of upvotes
	- downvotes:		number of downvotes
	- num_reports:		number of "Reports" comment received

	'''

	def __init__(self):
		self.submission_id = ''
		self.comment_id = ''
		self.parent_id = ''
		self.author = ''
		self.timestamp = 0
		self.has_replies = False
		self.upvotes = 0
		self.downvotes = 0
		self.num_reports = 0
		self.num_comments = 0
		self.body = ''
		self.title = ''


	def SetAttr(self, attr_dict):
		try:
			assert type(attr_dict) == dict
			if 'body' in attr_dict.keys():
				#If comment is non empty (not deleted)
				if attr_dict['body']:
					self.submission_id = attr_dict['link_id']
					self.comment_id = attr_dict['id']
					self.parent_id = attr_dict['parent_id']
					self.author = attr_dict['author']
					self.timestamp = attr_dict['created_utc']
					self.upvotes = attr_dict['ups']
					self.downvotes = attr_dict['downs']
					self.num_reports = attr_dict['num_reports']
					self.body = attr_dict['body']
			elif 'selftext' in attr_dict.keys():
				self.submission_id = attr_dict['id']
				self.comment_id = attr_dict['id']
				self.parent_id = attr_dict['id']
				self.author = attr_dict['author']
				self.timestamp = attr_dict['created_utc']
				self.upvotes = attr_dict['ups']
				self.downvotes = attr_dict['downs']
				self.num_reports = attr_dict['num_reports']
				self.body = attr_dict['selftext']
				self.title = attr_dict['title']
			if 'replies' in attr_dict.keys():
				if attr_dict['replies']:
					self.has_replies = True
			if '_' in self.submission_id:
				loc = self.submission_id.find('_')
				self.submission_id = self.submission_id[loc+1:]
			if '_' in self.comment_id:
				loc = self.comment_id.find('_')
				self.comment_id = self.comment_id[loc+1:]	
			if '_' in self.parent_id:
				loc = self.parent_id.find('_')
				self.parent_id = self.parent_id[loc+1:]	
		except AssertionError:
			#print( "No key-value store at node")
			#_, _, tb = sys.exc_info()
			#traceback.print_tb(tb) # Fixed format
			#tb_info = traceback.extract_tb(tb)
			#filename, line, func, text = tb_info[-1]
			#print('An error occurred on line {} in statement {}'.format(line, text))
			pass

#----

'''Helper functions to parse Reddit Json'''
class ParseJson:
	''' tree = list (of nodes)
	 	node = list or dictionary
		pos = tuple'''

	'''Searches tree and returns Node at a given position'''
	def NodeByPos(tree, pos):
		node = tree
		for i in pos:
			node = ParseJson.NextTree(node)
			if node:
				node = node[i]
		return node

	'''Returns the next (sub)tree under a given node'''
	def NextTree(node):
		is_list = False
		while not(is_list):
			if type(node) == dict:
				if 'replies' in node.keys():
					node = node['replies']
				elif 'children' in node.keys():
					node = node['children']
				elif 'data' in node.keys():
					node = node['data']
				elif 'selftext' in node.keys():
					node = []
			if type(node) not in (list, dict):
				node = []
			is_list = (type(node) == list)
		return node

	'''Searches tree for location of all list nodes 
		and returns their indices'''
	def IndexTree(tree, rel_pos, abs_pos):
		index = []
		index.append( abs_pos )
		#print(abs_pos)
		node = ParseJson.NodeByPos(tree, rel_pos)
		next_tree = ParseJson.NextTree(node)
		for i in range(len(next_tree)):
			next_pos = abs_pos+(i,)
			index.extend(ParseJson.IndexTree(next_tree, (i,), next_pos ))
		return index

	'''Searches node for the next dictionary with comment info'''
	def InfoAtNode(node):
		info = ''
		try:
			assert( type(node) == dict)
			if 'data' in node.keys():
				dkeys = node['data'].keys()
				if 'replies' in dkeys or 'selftext' in dkeys:
					info = node['data']
		except AssertionError:
			pass
		return info
