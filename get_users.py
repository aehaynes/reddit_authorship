import praw

class User:
	reddit = praw.Reddit('user')

	def __init__(self, uname):
		self.name = ''
		self.num_comments = 0
		self.num_btc_comments = 0
		try:
			self.userObj = self.reddit.get_redditor(uname)
			self.name = uname
		except:
			pass

	def SetAttr(self):
		if self.name:
			comments = self.userObj.get_comments(limit = 1000)
			subreddits = [c.subreddit.display_name for c in comments]
			self.num_comments = len(subreddits)
			self.num_btc_comments = subreddits.count('Bitcoin')
			del self.userObj


def GetUserInfo(uname):
	user = User(uname)
	user.SetAttr()
	return user
