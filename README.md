# reddit_authorship
An analysis to detect authors with multiple accounts in comments from the /r/Bitcoin subreddit. See andrehaynes.me/portfolio for the full report

Project Report:
	http://andrehaynes.me/static/AA_Reddit.pdf


Project Database:
	https://www.dropbox.com/s/sn64yu46kr5z1m9/redditDB.sqlite?dl=0


Project Files:

	dependencies.txt -- Project dependencies

	keywords.txt -- List of keywords used to search for relevant threads
	get_submissions.py -- Module for getting submission id's that match keywords in keywords.txt
	submission_id.pkl -- Submission Id's for threads used in the analysis	
	main_reddit_data.py -- Run this to scrape comments from the threads in submission_id.pkl. Stores comments in sqlite  database (see dropbox link)
	
	setup_data_index.py -- Modules for preprocessing comments, setting features, setting dataframes and indices
	setup_db.py -- Module for DB schema and function calls
	stoplist.pkl -- stopwords used in the analysis
	main_tunemodels.py -- Run this to tune to optimal number of topics for the topic models
	num_topics.pkl -- Results from main_tunemodels.py used in this analysis	
	main_dataframes.py -- Run this to create dataframes for classification

	setup_experiments.py -- Functions for constructing dataframes and model building. Not well documented
  	analysis.py -- functions for training and testing the classifiers, and generating sock rankings. Not documented
  
