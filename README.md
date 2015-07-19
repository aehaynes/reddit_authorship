# reddit_authorship
An analysis to detect authors with multiple accounts in comments from the /r/Bitcoin subreddit. See andrehaynes.me/portfolio for the full report


Project Files

	dependencies.txt 	-- Project Dependencies
	keywords.txt 		-- List of keywords used to search for relevant threads
	get_submissions.py 	-- Module for getting submissions that match keywords in keywords.txt
	setup_data_index.py 	-- Modules for preprocessing comments, setting features, setting dataframes and indices
	setup_db.p		-- Module for DB schema and function calls
	stoplist.pk		-- stopwords used in the analysis
	main_reddit_data.p	-- Run this to scrape comments from the threads in submission_id.pkl. Stores comments in sqlite database
	main_tunemodels.p	-- Run this to tune to optimal number of topics for the topic models
	num_topics.pk		-- Results from main_tunemodels.py used in this analysis
	main_dataframes.p	-- Run this to create dataframes for classification
	submission_id.pkl 	-- Submission Id's for threads used in the analysis
	setup_experiments.py 	-- Functions for constructing dataframes and model building.
  	analysis.py 		-- functions for training and testing the classifiers, and generating sock rankings
  
