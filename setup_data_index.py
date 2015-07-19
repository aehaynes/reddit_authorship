import numpy as np
import gensim
import pickle
import re
import os
import nltk
import sqlite3
import shelve
from setup_db import RedditDB as Rdb
from nltk.tag.mapping import map_tag
from scipy.special import stdtr
import scipy

temp_dir = os.getcwd()+'/tmp/'
if not os.path.exists(temp_dir):
	os.makedirs(temp_dir)
gensim_dir = temp_dir + 'gensim/'
if not os.path.exists(gensim_dir):
	os.makedirs(gensim_dir)
shelve_dir = temp_dir + 'shelve/'
if not os.path.exists(shelve_dir):
	os.makedirs(shelve_dir)


class PreProcessing:
	def __stoplist():
		'''get pickled stopwords'''
		with open('stoplist.pkl', 'rb') as f:
			stopwords = pickle.load(f)
		return stopwords

	__stopwords = __stoplist()
	regex_extra = [
		(re.compile(r'\.\.+'), ' '), #excessive dots
		(re.compile(r'[^A-Za-z0-9\-\s]+'), ' ')] #characters to exclude

	regex_light = [
		(re.compile(r'https?:\/\/[\S]*'), ''), #links
		(re.compile(r'www\.[\S]*'), ''), #links
		(re.compile(r'\/u\/|\/r\/'), ' '), #users or subreddit prefix
		(re.compile(r'\/|\\'), ' '), #forward and backward slashes between words
		(re.compile(r'\&gt\;[^\n]*\n'), ' '), #quoted text
		(re.compile(r'\"+|\'+|\“+'), ''), #contractions
		(re.compile(r'\(+|\)+|\[+|\]+'), ''), #contractions
		(re.compile(r'\&nbsp+|\&lt+|\&gt+|\&amp+'), ' '), #html escape codes
		(re.compile(r'\-\-+'), ' '), #excessive dashes
		(re.compile(r'\s\s+'), ' ')] #excessive spaces

	def substitutions(text):
		'''replace links and html tags with regex'''
		clean_text = [text]
		extra_clean = [text]
		for regx in PreProcessing.regex_light:
			last_text = clean_text[-1]
			clean_text.append( regx[0].sub(regx[1], last_text) )
			for eregx in PreProcessing.regex_extra:
				extra_clean.append( eregx[0].sub(eregx[1], clean_text[-1]))
		ret = (clean_text[-1], extra_clean[-1] )
		return ret

	def tokenize(text):
		tokens = [ word for word in text.split()]
		return tokens

	def char_ngram(text, n):
		l = len(text)
		return [ text[i:i+n] for i in range(l) if i+n-1 < l]	

	def remove_stoptokens(tokens):
		return [ token for token in tokens if token not in PreProcessing.__stopwords and len(token) > 1]

	def text2tokens(text, clean = False):
		if clean: ret = PreProcessing.tokenize( PreProcessing.substitutions(text.lower())[-1] )
		else: ret = PreProcessing.tokenize(text)
		return ret

	#def tokens2pos(tokens):
	#	return [ map_tag('wsj', 'universal',p[1]) for p in nltk.pos_tag(tokens)]

	def tokens2ngram(tokens, n):
		return list( nltk.ngrams(tokens, n))

	def dictionary(tokens):
		dictionary = gensim.corpora.Dictionary(tokens)
		return dictionary

	def corpus(dictionary, tokens):
		corpus = [dictionary.doc2bow(g) for g in tokens] 
		return corpus


class Index:
	Comments = []
	AuthorIndex = {}
	AuthorList = []

	Corpus = {}
	Dictionary = {}
	Model = {}
	Indices = {}

	def reset():
		Comments = []
		AuthorIndex = {}
		AuthorList = []		

		Corpus = {}
		Dictionary = {}
		Model = {}
		Indices = {}		

	def set_indices(name, idx):
		if name in Index.Indices.keys():
			print("Warning: Indices name '%s' already exists. Overwriting..." % name)		
		Index.Indices[name] = idx

	def set_dictionary(name, tokens):
		if name in Index.Corpus.keys():
			print("Warning: Dictionary name '%s' already exists. Overwriting..." % name)
		dictionary = PreProcessing.dictionary(tokens)
		Index.Dictionary[name] = gensim_dir+name.replace(' ', '_')+".dict"
		dictionary.save(Index.Dictionary[name])
		del dictionary		

	def set_corpus(name, tokens, dictionary):
		if name in Index.Corpus.keys():
			print("Warning: Corpus name '%s' already exists. Overwriting..." % name)
		corpus = PreProcessing.corpus(dictionary, tokens)
		Index.Corpus[name] = gensim_dir+name.replace(' ', '_')+".mm"
		gensim.corpora.MmCorpus.serialize(Index.Corpus[name], corpus)
		del corpus

	def set_model(name, model_type, model):
		if name in Index.Model.keys():
			print("Warning: Model name '%s' already exists. Overwriting..." % name)			
		Index.Model[name] = [gensim_dir+name.replace(' ', '_')+".model", model_type]
		model.save(Index.Model[name][0])
		del model

	def get_corpus(name):
		if name in Index.Corpus.keys():
			return gensim.corpora.MmCorpus(Index.Corpus[name])

	def get_dictionary(name):
		if name in Index.Dictionary.keys():
			return gensim.corpora.Dictionary().load(Index.Dictionary[name], mmap = 'r')

	def get_model(name):
		if name in Index.Model.keys():
			if Index.Model[name][1] == 'tfidf':
				return gensim.models.tfidfmodel.TfidfModel.load(Index.Model[name][0], mmap = 'r')
			elif Index.Model[name][1] == 'lsa':
				return  gensim.models.LsiModel.load(Index.Model[name][0], mmap='r')
			elif Index.Model[name][1] == 'lda':
				return  gensim.models.ldamodel.LdaModel.load(Index.Model[name][0], mmap='r')


class Helpers:

	def train_tfidf(corpus):
		tfidf = gensim.models.TfidfModel(corpus)
		return tfidf

	def corp2dense(corpus, model, length=None):
		if hasattr(model, 'num_topics'):
			dense = gensim.matutils.sparse2full( model[corpus], model.num_topics )
		else:
			if length:
				dense = gensim.matutils.sparse2full( model[corpus], length )
			else:
				print("Model has no num_topics attribute. corp2dense requires length.")
		return scipy.sparse.csr_matrix(dense) #TODO: change this function to corp2sparse

	def distance( v1, v2, dtyp = 'hellinger'):
		if dtyp == 'hellinger':
			X = (np.sqrt(v1) - np.sqrt(v2))
			X.data **=2
			distance = np.sqrt(0.5*(X).sum())
		elif dtyp == 'cosine':
			distance = v1.dot(v2.T)/np.sqrt(v1.dot(v1.T)*v2.dot(v2.T))
		elif dtyp == 'jaccard':
			distance = len(set(v1) & set(v2))/float( len(set(v1) | set(v2)) )
		return float(distance)

	def welch_test( x1_stats, x2_stats):
		x1bar, x2bar, v1, v2, n1, n2 = x1_stats[0], x2_stats[0], x1_stats[1]**2, x2_stats[1]**2, x1_stats[2], x2_stats[2]
		# Compute Welch's t-test using the descriptive statistics.
		tf = (x1bar - x2bar) / np.sqrt(v1/n1 + v2/n2)
		dof = (v1/n1 + v2/n2)**2 / (v1**2/(n1**2*(n1-1)) + v2**2/(n2**2*(n2-1)))
		pf = 2*stdtr(dof, -np.abs(tf))
		return float(pf)

	def min_max_ratio(v1, v2):
		return float(min(v1, v2)/float(max(v1,v2)))

	def set_shelf(fname, obj_key, obj):
		spath = shelve_dir+fname.replace(' ', '_')+'.shelve'
		with shelve.open(spath) as shelf:
			if obj_key in shelf.keys():
				print("Warning: shelf key '%s' already exists. Overwriting..." % obj_key)
			shelf[obj_key] = obj

	def get_shelf(fname, obj_key):
		spath = shelve_dir+fname.replace(' ', '_')+'.shelve'
		with shelve.open(spath) as shelf:
			if obj_key not in shelf.keys():
				print("Requested key '%s' does not exist in shelf." % obj_key)
				ret =  []
			else:
				ret = shelf[obj_key]
		return ret

	def get_keys(fname):
		spath = shelve_dir+fname.replace(' ', '_')+'.shelve'
		with shelve.open(spath) as shelf:
			ret = list(shelf.keys())
		return ret		

	def clear_shelf(fname):
		spath = shelve_dir+fname.replace(' ', '_')+'.shelve'
		with shelve.open(spath) as shelf:
			print("Clearing shelf '%s'..." % fname)
			shelf.clear()
			print("Shelf cleared.")

	def del_shelf(fname):
		print("Deleting shelf '%s'..." % fname)
		spath = shelve_dir+fname.replace(' ', '_')+'.shelve'
		try:
			os.remove(spath)
		except FileNotFoundError:
			pass
		print("Shelf deleted.")



class Features:

	def __init__(self):
		self.Id = 0
		
		self.TotalSentences = 0
		self.TotalWords = 0
		self.WordRate = 0
		self.StdDev_WR = 0
		
		self.CommaRate = 0
		self.EllipsisRate = 0
		self.SemicolonRate = 0
		self.PeriodRate = 0
		self.ColonRate = 0

		self.WordDiversity = 0
		self.RawTokens = []
		self.PPTokens = []
		self.CharTokens = []
		self.POS = []

		self.corpus = []
		self.lda_vec = []
		self.lsa_vec = []
		self.tfidf_vec = []
		self.ngfidf_vec = []

	def set_features(self, comments, dictionary = None):
		std_offset = 0.001
		try:
			assert type(comments) == str
		except AssertionError:
			print(" Error, comments variable passed should be a string. Creating string and continuing...")
			comments = ' . '.join(comments)
		clean_text = PreProcessing.substitutions(comments)
		self.RawTokens = [PreProcessing.text2tokens(clean_text[0])]
		self.PPTokens = [PreProcessing.remove_stoptokens( PreProcessing.text2tokens(clean_text[1].lower() ) )]
		self.CharTokens = [PreProcessing.char_ngram(clean_text[0].lower(), 4)]
		
		sentences = nltk.sent_tokenize(clean_text[0])
		words_sentences = [[token for token in PreProcessing.text2tokens(s) if re.match(r'\w+', token)] for s in sentences ] 
		words = [token for tokens in self.PPTokens for token in tokens]
		words_in_sentences = np.array([len(words)+1 for sentence in words_sentences for words in sentence])
		vocab_in_sentences = np.array([len(set(words))+1 for sentence in words_sentences for words in sentence])
		diversity_in_sentences = vocab_in_sentences / words_in_sentences
		commas_in_sentences = np.array([ s.count(',')+1 for s in sentences])
		ellipsis_in_sentences = np.array([ s.count('..')+s.count('...')+1 for s in sentences])
		semicolon_in_sentences = np.array([ s.count(';')+1 for s in sentences])
		colon_in_sentences = np.array([ s.count(':')+1 for s in sentences])

		self.TotalSentences = len(sentences)
		self.TotalWords = len(words) #Doesnt include stop word tokens
		self.Vocab = set(words)
		self.TotalVocab = len(self.Vocab)		
		
		self.WordStats = [words_in_sentences.mean(), words_in_sentences.std()+std_offset, len(words_in_sentences)]
		self.CommaStats = [commas_in_sentences.mean(), commas_in_sentences.std()+std_offset, len(commas_in_sentences)]
		self.EllipsisStats = [ellipsis_in_sentences.mean(), ellipsis_in_sentences.std()+std_offset, len(ellipsis_in_sentences)]
		self.SemicolonStats = [semicolon_in_sentences.mean(), semicolon_in_sentences.std()+std_offset, len(semicolon_in_sentences)]
		self.ColonStats = [colon_in_sentences.mean(), colon_in_sentences.std()+std_offset, len(colon_in_sentences)]
		self.DiversityStats = [diversity_in_sentences.mean(), diversity_in_sentences.std()+std_offset, len(diversity_in_sentences)]
		self.Diversity = self.TotalVocab/float(self.TotalWords)




