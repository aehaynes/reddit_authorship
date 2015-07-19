import sys
import traceback
import gc
import multiprocessing
import functools
from time import time
import h5py 

import setup_db as Sdb
from setup_db import RedditDB as Rdb
from setup_data_index import *

import numpy as np
import gensim

import scipy
from sparsesvd import sparsesvd


seed = 0
comment_cutoff = 10

profile_shelve = 'Profiles'
dataframe_shelve = 'Dataframe'
h5_file = 'Df.h5'

all_attributes = ['Id', 'Vocab',  'TotalWords', 'TotalVocab', 'Diversity', \
	'WordStats', 'CommaStats', 'EllipsisStats', 'SemicolonStats', 'ColonStats', 'DiversityStats', \
	'lda_vec', 'lsa_vec', 'tfidf_vec', 	'ngfidf_vec']

"""Top level functions to support multiprocessing"""
def distance( v1, v2, dtyp):
	return Helpers.distance(v1,v2,dtyp)

def welch_test( x1_stats, x2_stats):
	return Helpers.welch_test( x1_stats, x2_stats)

def min_max_ratio(v1, v2):
	return Helpers.min_max_ratio(v1, v2)




''' Init Indices for Comments and Authors'''

def init_index(itype = 'train'):
	'''Set Indices with comments and author list'''
	start_time = time()
	Index.reset()	
	print("\nSetting indices for feature engineering and classification")	
	if itype == 'train': 
		ikey = '%(training cid 2)s'
	elif itype == 'unseen': 
		ikey = '%(unseen cid)s'
	else: 
		print("Unknown key '%s'. Exiting..." % ikey)
		return
	DB = Rdb()
	DB.connect()
	DB.db.execute(("""SELECT Comment, Author, CommentId 
						FROM commentinfo WHERE CommentId IN ( %s ) and Comment != '' """ % ikey) % Sdb.query)
	results = DB.db.fetchall()
	DB.close()
	n = len(results)
	for i in range(n):
		r = results[i]
		Index.Comments.append( r[0] )
		if r[1] in Index.AuthorIndex.keys():
			Index.AuthorIndex[ r[1] ].append( i )
		else:
			Index.AuthorIndex[r[1]] = [i]
	Index.AuthorList = [[a,len(Index.AuthorIndex[a])] for a in Index.AuthorIndex.keys() ]
	print("Fetched indices in %s seconds. Number of Comments = %s. Number of Authors = %s" % \
			(round(time() - start_time), len(Index.Comments), len(Index.AuthorIndex) ))




'''Methods for feature engineering
 using topic models'''

def FE_set_samples(train_prop = 0.6, cutoff = comment_cutoff):
	'''Set samples for feature engineering'''
	start_time = time()
	print("\nFE: Setting samples for feature engineering")	
	index = []
	for a in Index.AuthorList:
		if a[1] >= cutoff:
			index.append( Index.AuthorIndex[a[0]] )
	idx_train = []
	idx_val = []
	for idx in index:
		n = len(idx)
		n_train = round(train_prop*n)
		rand = np.random.RandomState(seed)
		rand.shuffle(idx)
		idx_train.append( idx[0:n_train] )
		idx_val.append( idx[n_train:] )
	Index.set_indices('train features', idx_train)
	Index.set_indices('val features', idx_val)
	print("FE: Fetched samples in %s seconds. Number of Authors = %s" % (round(time() - start_time), len(index)))


def FE_set_tokens():
	'''set tokens for feature engineering'''
	start_time = time()
	print("\nFE: Setting feature engineering nlp structures")		
	tokens = [PreProcessing.remove_stoptokens( 
				PreProcessing.text2tokens(' . '.join( [Index.Comments[i] for i in idx ]), True)) \
				for idx in Index.Indices['train features']]
	Index.set_dictionary('train features', tokens)
	dictionary = Index.get_dictionary('train features')
	Index.set_corpus('train features', tokens, dictionary )
	del tokens
	gc.collect()

	tokens = [PreProcessing.remove_stoptokens( 
				PreProcessing.text2tokens(' . '.join( [Index.Comments[i] for i in idx ]), True)) \
				for idx in Index.Indices['val features']]	
	Index.set_corpus('val features', tokens, dictionary )
	del dictionary, tokens
	print("FE: Fetched tokens in %s seconds" % (round(time() - start_time)))



def tune_lsa():
	print("Tuning lsa model for optimal number of topics")
	start_time = time()
	corpus = Index.get_corpus('train features')
	dictionary = Index.get_dictionary('train features')
	lsa =  gensim.models.LsiModel( corpus=corpus, id2word=dictionary, \
			num_topics =corpus.num_docs) #should num_topics be based on number of words?
	variances = lsa.projection.s
	cumulative_explained_variances = np.cumsum(variances)/np.sum(variances)
	lsa_num_topics = np.where( cumulative_explained_variances > 0.5)[0][0]
	del lsa
	print("Tune lsa model in %s seconds. Selected %s topics." % (round(time() - start_time), lsa_num_topics) )
	return lsa_num_topics


def tune_lda():
	def tune(my_corpus, dictionary, min_topics=2,max_topics=50,step=2):
		def sym_kl(p,q):
			return np.sum([scipy.stats.entropy(p,q),scipy.stats.entropy(q,p)])

		kl = []
		Hbar = []
		perplexity = []
		n_topics = []
		l = np.array([sum(cnt for _, cnt in doc) for doc in my_corpus])
		corpus = Index.get_corpus('train features')
		for i in range(min_topics,max_topics,step):
			n_topics.append(i)
			lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary,num_topics=i, alpha = 'auto')
			m1 =  scipy.sparse.csc_matrix(lda.expElogbeta)
			U,cm1,V = sparsesvd(m1, m1.shape[0])
			#Document-topic matrix
			lda_topics = lda[my_corpus]
			m2 = gensim.matutils.corpus2dense(lda_topics, lda.num_topics).transpose()
			cm2 = l.dot(m2)
			cm2 = cm2 + 0.0001
			cm2norm = np.linalg.norm(l)
			cm2 = cm2/cm2norm
			kl.append(sym_kl(cm1,cm2))
			entropy_list = [scipy.stats.entropy([x[1] for x in lda[v]] ) for v in my_corpus]
			Hbar.append(np.mean(entropy_list))
			perplexity.append( lda.log_perplexity(my_corpus) )
			print("NumTopics: %s | Unscaled Entropy: %s | Per-word-bound: %s | Per-word-perplexity: %s | Arun measure %s" % \
					(i, Hbar[-1], perplexity[-1], np.exp2(-perplexity[-1]), kl[-1]))
		return n_topics, Hbar, perplexity, kl

	start_time = time()
	corpus = Index.get_corpus('val features')
	dictionary = Index.get_dictionary('train features')
	n_topics, Hbar, perplexity, kl = tune(my_corpus =corpus, dictionary = dictionary)
	dip = [ kl[i] < kl[i-1] for i in range(len(kl))]
	dip_index = dip[1:].index(True)+1
	lda_num_topics = n_topics[dip_index]
	print("Tune lda model in %s seconds. Selected %s topics." % (round(time() - start_time), lda_num_topics) )
	return lda_num_topics



'''Methods for Classification '''

def CL_init_index(cutoff = comment_cutoff, unseen = False, train_prop = 0.7, seed = seed):
	'''Setup indices for classification task.
		- splits comments from known authors into two groups to generate labeled samples for training classifier'''
	start_time = time()
	print("\nCLF: Initializing index for classification")
	rand = np.random.RandomState(seed)
	candidate_authors = [ author[0] for author in Index.AuthorList if author[1] >= cutoff]
	if unseen:
		Index.set_indices('unseen clf', [])
		for author in Index.AuthorList:
			if author[0] in candidate_authors:
				idx = [a for a in Index.AuthorIndex[author[0]] ]
				row_dict = {'indices': idx, 'author': author[0]}
				Index.Indices['unseen clf'].append( row_dict )
	else:
		rand.shuffle(candidate_authors)
		idx_co = round( len(candidate_authors)*train_prop)
		training_authors = candidate_authors[0:idx_co]
		test_authors = candidate_authors[idx_co:]
		Index.set_indices('train clf', [])
		Index.set_indices('test clf', [])
		for author in Index.AuthorList:
			if author[0] in training_authors + test_authors:
				idx = [a for a in Index.AuthorIndex[author[0]] ]
				idx_co = round( len(idx)*0.5 )
				rand.shuffle(idx)
				if author[0] in training_authors:
					row_dict = {'indices': idx[0:idx_co], 'author': author[0]}
					Index.Indices['train clf'].append( row_dict )
					row_dict = {'indices': idx[idx_co:], 'author': author[0]}
					Index.Indices['train clf'].append( row_dict )
				else:
					row_dict = {'indices': idx[0:idx_co], 'author': author[0]}
					Index.Indices['test clf'].append( row_dict )
					row_dict = {'indices': idx[idx_co:], 'author': author[0]}
					Index.Indices['test clf'].append( row_dict )			
		rand.shuffle(Index.Indices['train clf'])
		rand.shuffle(Index.Indices['test clf'])
	print("CLF: Fetched index in %s seconds" % (round(time() - start_time)))


def CL_set_base_models(index, lda_num_topics, lsa_num_topics):
	start_time = time()
	print("\nCLF: Setting base topic models for classification")
	print("CLF: Setting base tokens")

	tokens = [PreProcessing.char_ngram(' . '.join( [Index.Comments[i] for i in idx ]), 4) for idx in index]
	Index.set_dictionary('base 4grams', tokens)
	dictionary = Index.get_dictionary('base 4grams')
	Index.set_corpus('base 4grams', tokens, dictionary)	
	del dictionary
	gc.collect()

	tokens = [PreProcessing.remove_stoptokens( 
					PreProcessing.text2tokens(' . '.join( [Index.Comments[i] for i in idx ]), True)) for idx in index]
	Index.set_dictionary('base tokens', tokens)
	dictionary = Index.get_dictionary('base tokens')
	Index.set_corpus('base tokens', tokens, dictionary)

	print("CLF: Setting base models")
	corpus = Index.get_corpus('base tokens')
	model = gensim.models.LdaModel( corpus, id2word=dictionary, num_topics = lda_num_topics, alpha='auto')
	Index.set_model('lda', 'lda', model)
	del model
	gc.collect()

	model = Helpers.train_tfidf( corpus )
	Index.set_model('tfidf', 'tfidf', model)
	
	model_lsa = gensim.models.LsiModel( model[corpus], id2word=dictionary, num_topics = min(100,lsa_num_topics))
	Index.set_model('lsa', 'lsa', model_lsa)
	del model, model_lsa, corpus
	gc.collect()

	corpus = Index.get_corpus('base 4grams')
	model = Helpers.train_tfidf( corpus )
	Index.set_model('4gfidf', 'tfidf', model)
	del model, corpus
	gc.collect()
		
	print("CLF: Fetched models in %s seconds" % (round(time() - start_time)))


def CL_set_features( index_dict, shelf_prefix):
	start_time = time()
	print("\nCLF: Setting features on data for classification")
	n = len(index_dict)
	model_tfidf = Index.get_model('tfidf')
	model_ngfidf = Index.get_model('4gfidf')
	model_lsa = Index.get_model('lsa')
	model_lda = Index.get_model('lda')
	dict_base = Index.get_dictionary('base tokens')
	dict_4gram = Index.get_dictionary('base 4grams')
	
	num_processed = 0
	shelf_keys = []
	shelf = shelf_prefix+profile_shelve
	Helpers.del_shelf(shelf)
	for i in np.arange(n):
		shelf_key = 'profile_%s' % i
		Profile = Features()
		Profile.Id = index_dict[i]
		Profile.set_features( ' . '.join( [Index.Comments[idx] for idx in Profile.Id['indices']]), dict_base  )		
		corpus = PreProcessing.corpus(dict_base, Profile.PPTokens)
		char_corpus = PreProcessing.corpus(dict_4gram, Profile.CharTokens)		
		Profile.lda_vec = [Helpers.corp2dense(corp, model_lda ) for corp in corpus]
		Profile.tfidf_vec = [Helpers.corp2dense(corp, model_tfidf, len(dict_base) ) for corp in corpus]
		Profile.ngfidf_vec = [Helpers.corp2dense(corp, model_ngfidf, len(dict_4gram) ) for corp in char_corpus]
		Profile.lsa_vec = [Helpers.corp2dense( model_tfidf[corp], model_lsa) for corp in corpus]
		
		Helpers.set_shelf(shelf, shelf_key, Profile)
		del corpus, char_corpus, Profile
		gc.collect()

		num_processed += 1
		shelf_keys.append(shelf_key)
		if num_processed % 100 == 0:
			print("-processed %s of %s profiles in %s seconds" % (num_processed, n, round(time() - start_time)))
	print("CLF: Fetched features in %s seconds" % (round(time() - start_time)))


def CL_update_df(shelf_prefix, profile_keys, attribute, multiproc = True ):
	gc.collect()
	vector = [getattr(Helpers.get_shelf(shelf_prefix+profile_shelve, p), attribute) for p in profile_keys]
	n = len(vector)
	pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
	gc.collect()
	if attribute == 'Id':
		if multiproc:
			comparison_vec = [[ ( int(vector[x]['author'] == vector[y]['author']),\
									(vector[x]['author'], vector[y]['author'])) \
							for x in np.arange(y+1,n, 1)] for y in np.arange(n)]	
		else:
			comparison_vec = [[ ( int(vector[x]['author'] == vector[y]['author']),\
									(vector[x]['author'], vector[y]['author'])) \
							for x in np.arange(y+1,n, 1)] for y in np.arange(n)]

	elif attribute in ['lda_vec', 'lsa_vec', 'tfidf_vec', 'ngfidf_vec', 'Vocab']:
		if attribute == 'lda_vec':
			if multiproc:
				comparison_vec = [[ pool.map(functools.partial(distance, v2=vector[y][0], dtyp='hellinger'), \
									[vector[x][0] for x in np.arange(y+1,n, 1)])] for y in np.arange(n)]
			else:
				comparison_vec = [[distance( vector[x][0], vector[y][0], 'hellinger') \
								for x in np.arange(y+1,n, 1)] for y in np.arange(n)]														
		
		elif attribute == 'Vocab':
			if multiproc:
				comparison_vec = [[ pool.map(functools.partial(distance, v2=vector[y], dtyp='jaccard'), \
									[vector[x] for x in np.arange(y+1,n, 1)])] for y in np.arange(n)]
			else:
				comparison_vec = [[distance( vector[x], vector[y], 'jaccard') \
								for x in np.arange(y+1,n, 1)] for y in np.arange(n)]
		
		elif attribute in ['lsa_vec', 'tfidf_vec', 'ngfidf_vec']:
			if multiproc:
				comparison_vec = [[ pool.map(functools.partial(distance, v2=vector[y][0], dtyp='cosine'), \
									[vector[x][0] for x in np.arange(y+1,n, 1)])] for y in np.arange(n)]
			else:
				comparison_vec = [[distance( vector[x][0], vector[y][0], 'cosine') \
								for x in np.arange(y+1,n, 1)] for y in np.arange(n)]

	elif attribute in ['TotalWords', 'TotalVocab', 'Diversity']:
		if multiproc:
			comparison_vec = [[ pool.map(functools.partial(min_max_ratio, v2=vector[y]), \
								[vector[x] for x in np.arange(y+1,n, 1)])] for y in np.arange(n)]		
		else:
			comparison_vec = [[min_max_ratio( vector[x], vector[y]) \
								for x in np.arange(y+1,n, 1)] for y in np.arange(n)]
	elif attribute in ['WordStats', 'CommaStats', 'EllipsisStats', 'SemicolonStats', 'ColonStats', 'DiversityStats']:
		if multiproc:
			comparison_vec = [[ pool.map(functools.partial(welch_test, x2_stats=vector[y]), \
								[vector[x] for x in np.arange(y+1,n, 1)])] for y in np.arange(n)]			
		else:
			comparison_vec = [[ welch_test( vector[x], vector[y]) \
								for x in np.arange(y+1,n, 1)] for y in np.arange(n)]
	comparison_vec = [c for c_vec in comparison_vec for c in c_vec]	
	if comparison_vec:
		print("Computed comparisons for %s attribute" % attribute)
		Helpers.set_shelf(shelf_prefix+dataframe_shelve, attribute, comparison_vec)
	del comparison_vec, vector
	gc.collect()


def CL_set_dataframe(shelf_prefix = 'train clf', overwrite = False):
	start_time = time()
	print("\nCLF: Setting dataframe")
	shelf = shelf_prefix+dataframe_shelve
	if overwrite:
		Helpers.del_shelf(shelf)
	profile_keys = Helpers.get_keys(shelf_prefix+profile_shelve)
	recorded_attributes = Helpers.get_keys(shelf_prefix+dataframe_shelve)
	attributes = list(set(all_attributes) - set(recorded_attributes))
	num_processed = 0
	print("Processing attributes...")
	while len(attributes) > 0:
		CL_update_df(shelf_prefix=shelf_prefix, profile_keys= profile_keys, attribute=attributes.pop(0) )
		gc.collect()
		num_processed+=1
		gc.collect()
		print("-processed %s attributes in %s seconds" % (num_processed, round(time() - start_time)))
	print("CLF: Fetched dataframe in %s seconds" % (round(time() - start_time)))
	print("WRITING DATAFRAME TO SHELF COMPLETE")


def CL_df2hdf(df_name):
	start_time = time()
	print("\nCLF: Setting HDF5 dataframe")
	df_keys = Helpers.get_keys(df_name+dataframe_shelve)
	f = h5py.File(h5_file)
	if df_name in f.keys():
		del f[df_name]
	df = f.create_group(df_name)
	Ids = Helpers.get_shelf(df_name+dataframe_shelve, 'Id')
	df['Labels'] = np.array([i[0] for i in Ids])
	df['Authors'] = np.array([(i[1][0].encode('ascii','ignore'), i[1][1].encode('ascii','ignore')) for i in Ids])
	del Ids
	gc.collect()
	for attribute in all_attributes[1:]:
		gc.collect()
		values = Helpers.get_shelf(df_name+dataframe_shelve, attribute)
		if type(values[0]) == list:
			df[attribute] = np.array( [v for vl in values for v in vl] )
		else:
			df[attribute] = np.array(values)
		gc.collect()
		print("-processed %s to hdf5 in %s seconds" % (attribute, round(time() - start_time)))
	f.close()


def validityCheck():
	valid = False
	f = h5py.File(h5_file, 'r')
	df_keys = ['unseen clf', 'train clf', 'test clf']
	try:
		assert set(f.keys()) == set(df_keys)
		for key in df_keys:
			length = len(f[key]['Labels'])
			for k in f[key].keys():
				assert len(f[key][k]) == length
				print("Validity check passed for %s, %s" % (key, k))
		valid = True
	except AssertionError:
		print("Invalid HDF5 dataframe. Recompile with CL_df2hdf")
	f.close()
	return valid


def CL_design_matrix(df_name, attr_list = None, sample_df= True, sample_ratio = 1, indices = None):
	f = h5py.File(h5_file, 'r')
	try:
		valid = True#validityCheck()		
		ret = {}
		if valid:
			if attr_list is None:
					attr_list = ['Labels']
					attr_list.extend(all_attributes)
					attr_list.remove('Id')
			else:
				attr_list = ['Labels'] + attr_list
			
			#label_var = attr_list[0]
			rem_var = attr_list[1:]

			if sample_df:
				assert sample_ratio <= 10 and sample_ratio >= 0.1
				idx_true = np.where( np.array(f[df_name]['Labels']) == 1)[0]
				idx_false = np.where( np.array(f[df_name]['Labels']) == 0)[0]
				assert len(idx_false) > len(idx_true)
				n_sample = round( len(idx_true)*sample_ratio)
				n_sample = min( n_sample, len(idx_false))
				idx_false = np.random.choice( idx_false, n_sample)
				idx = idx_true.tolist() + idx_false.tolist()			
			else:
				idx = list(indices)
			assert len(idx) <= len(f[df_name]['Labels'])
			idx.sort()

			Y = np.zeros(( len(idx),))
			X = np.zeros(( len(idx), len(rem_var) ))
			tmp = np.array(f[df_name]['Labels']) #tmp, to avoid "Src and dest data spaces have different sizes" OSError a.k.a weirdness. 
			Y = tmp.take(idx)						# The error may have to do with large slices in idx
			for i in np.arange(len(rem_var)):
				tmp = np.array(f[df_name][rem_var[i]])
				X[:, i] = tmp.take(idx)
			del tmp
			ret['X'] = X
			ret['Y'] = Y
			ret['indices'] = idx
			ret['attributes'] = attr_list
	except:
		traceback.print_exc(file=sys.stdout)
		pass
	f.close()
	gc.collect()
	return ret

