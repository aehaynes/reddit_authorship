from setup_experiments import *

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier as RandomF
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve as prcurve

import pandas as pd

import matplotlib.pyplot as plt 
import seaborn as sns
from itertools import cycle
from operator import itemgetter

sns.set_style("whitegrid")
sns.set_palette("bright", 10)
sns.set_context("paper")


def get_topics(unseen = True):
	if unseen == True:
		init_index('unseen')
		lsa_num_topics, lda_num_topics = 100, 6
		CL_init_index(unseen = True)
		unseen_index = [i['indices'] for i in Index.Indices['unseen clf'] ]
		CL_set_base_models(unseen_index, lda_num_topics, lsa_num_topics)
		file_prefix = 'unseen'
	else:
		init_index('train')
		lsa_num_topics, lda_num_topics = 100, 6
		CL_init_index(unseen = False)
		train_index = [i['indices'] for i in Index.Indices['train clf'] ]
		test_index = [i['indices'] for i in Index.Indices['test clf'] ]
		CL_set_base_models(train_index, lda_num_topics, lsa_num_topics)	
		file_prefix = 'train'
	model_lsa = Index.get_model('lsa')
	model_lda = Index.get_model('lda')

	column_labels = ["Topic %s" % i for i in range(1,6,1)] 
	row_labels = ["word %s" % i for i in range(1,11,1)]
	fig, ax = plt.subplots()
	ax.set_xticks(np.arange(10)+0.5, minor=False)
	ax.set_yticks(np.arange(5)+0.5, minor=False)
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	lsa_weights = np.zeros((5,len(model_lsa.show_topic(0))))
	lsa_words = np.zeros((5,len(model_lsa.show_topic(0))), dtype='object')
	for i in range(5):
		lsa_weights[i] = np.array([i[0] for i in model_lsa.show_topic(i)])
		lsa_words[i] = np.array([j[1] for j in model_lsa.show_topic(i)])
	hm = ax.pcolor(lsa_weights, cmap=plt.cm.bwr)
	#plt.set_cmap("bwr")
	for y in range(lsa_weights.shape[0]):
		for x in range(lsa_weights.shape[1]):
			ax.text(x+0.5, y+0.5, lsa_words[y,x], horizontalalignment='center', verticalalignment='center')
	ax.set_xticklabels(row_labels, minor=False)
	ax.set_yticklabels(column_labels, minor=False)
	plt.colorbar(hm)
	plt.tight_layout()
	plt.savefig("%s_lsa-top5.png" %file_prefix)

	column_labels = ["Topic %s" % i for i in range(1,6,1)] 
	row_labels = ["word %s" % i for i in range(1,len(model_lda.show_topic(0))+1,1)]
	fig, ax = plt.subplots()
	ax.set_xticks(np.arange(10)+0.5, minor=False)
	ax.set_yticks(np.arange(5)+0.5, minor=False)
	ax.invert_yaxis()
	ax.xaxis.tick_top()
	lda_weights = np.zeros((5,len(model_lda.show_topic(0))))
	lda_words = np.zeros((5,len(model_lda.show_topic(0))), dtype='object')
	for i in range(5):
		lda_weights[i] = np.array([i[0] for i in model_lda.show_topic(i)])
		lda_words[i] = np.array([j[1] for j in model_lda.show_topic(i)])
	hm = ax.pcolor(lda_weights, cmap=plt.cm.summer_r)
	for y in range(lda_weights.shape[0]):
		for x in range(lda_weights.shape[1]):
			ax.text(x+0.5, y+0.5, lda_words[y,x], horizontalalignment='center', verticalalignment='center')
	ax.set_xticklabels(row_labels, minor=False)
	ax.set_yticklabels(column_labels, minor=False)
	plt.colorbar(hm)
	plt.tight_layout()
	gc.collect()
	plt.savefig("%s_lda-top5.png" %file_prefix)


def run_RF(X_train, Y_train):
	clf = RandomF(n_estimators = 100, criterion = 'entropy')
	clf.fit(X_train, Y_train)
	return clf


with h5py.File(h5_file, 'r') as f:
	n = len(f['test clf']['Labels'])
n = min(n, 1e6)
idx = np.arange(n)
test_set = CL_design_matrix('test clf', sample_df = False, indices = idx)
nans_test = np.isnan(test_set['X']).any(axis=1)
X_test = test_set['X'][~nans_test]
Y_test = test_set['Y'][~nans_test]
idx_test = np.array(test_set['indices'])[~nans_test]


with h5py.File(h5_file, 'r') as f:
	n = len(f['unseen clf']['Labels'])
idx = np.arange(n)
unseen_set = CL_design_matrix('unseen clf', sample_df = False, indices = idx)
nans_unseen = np.isnan(unseen_set['X']).any(axis=1)
X_unseen = unseen_set['X'][~nans_unseen]
Y_unseen = unseen_set['Y'][~nans_unseen]
idx_unseen = np.array(unseen_set['indices'])[~nans_unseen]


balanced_test_set = CL_design_matrix(df_name = 'test clf', sample_ratio = 1)
nans_btest = np.isnan(balanced_test_set['X']).any(axis=1)
X_btest = balanced_test_set['X'][~nans_btest]
Y_btest = balanced_test_set['Y'][~nans_btest]
idx_btest = np.array(balanced_test_set['indices'])[~nans_btest]

''' Analysis of choice of ratio (since classes are imbalanced) '''
def ratio_analysis():
	gc.collect()
	ratios = np.array([0.1, 0.25, 0.50, 1, 2, 4, 10])
	metrics = {'roc': [], 'pr': [], 'accuracy': [], 'confusion': [], 'b_accuracy': [], 'b_confusion': []}
	for ratio in ratios:
		print("Ratio = ", ratio)
		train_set = CL_design_matrix(df_name = 'train clf', sample_ratio = ratio)
		nans_train = np.isnan(train_set['X']).any(axis=1)
		X_train = train_set['X'][~nans_train]
		Y_train = train_set['Y'][~nans_train]

		clf = run_RF(X_train, Y_train)
		Y_pred = clf.predict_proba(X_test)
		metrics['roc'].append( roc_curve(Y_test, Y_pred[:,1], pos_label = 1) )
		metrics['pr'].append( prcurve(Y_test, Y_pred[:,1], pos_label = 1) )
		Y_pred = clf.predict(X_test)
		metrics['accuracy'].append( accuracy_score(Y_test, Y_pred) )
		metrics['confusion'].append( confusion_matrix(Y_test, Y_pred) )
		Y_bpred = clf.predict(X_btest)
		metrics['b_accuracy'].append( accuracy_score(Y_btest, Y_bpred) )
		metrics['b_confusion'].append( confusion_matrix(Y_btest, Y_bpred) )
		print("Accuracy = %s for ratio = %s" % (metrics['b_accuracy'][-1], ratio) )

line_labels = [ 'Ratio: %s' % ratio for ratio in ratios]
line_styles = [ '--', ':', '-.', '-' ]
ls = cycle(line_styles)

#fig, ax = plt.subplots()
axes = [plt.subplot2grid((1, 2), (0, 0)), plt.subplot2grid((1, 2), (0, 1))]

title = 'Accuracy vs Sample Ratio'
axes[0].plot( ratios, metrics['b_accuracy'], label = "balanced test classes")
axes[0].plot( ratios, metrics['accuracy'], label = "unbalanced test classes")
axes[0].legend()
axes[0].set_title(title)
axes[0].set_xlabel("Sample Ratio")
axes[0].set_ylabel("Prediction Accuracy")

title = 'Recall vs Precision'
for pcurve, label in zip( metrics['pr'], line_labels ):
	axes[1].plot(pcurve[0], pcurve[1], next(ls), label = label)
axes[1].legend()
axes[1].set_title(title)
axes[1].set_xlabel("Precision")
axes[1].set_ylabel("Recall")

plt.tight_layout()
	plt.savefig('Accuracy and PR vs Ratio.png')
	return metrics

metrics = ratio_analysis()
with open("metrics.pkl", 'wb') as f:
	pickle.dump(metrics, f)

get_topics(False)
get_topics(True)


def performance_df():
	ratios = np.array([0.1, 0.25, 0.50, 1, 2, 4, 10])
	dfs= {}

	for i in range(len(ratios)):
		index = range(len(metrics['pr'][i][0]))
		precision = pd.Series(metrics['pr'][i][0], index = index)
		recall = pd.Series(metrics['pr'][i][1], index = index)		
		threshold = pd.Series( np.array( [0] + list(metrics['pr'][i][2])), index = index )
		dfs[i] = pd.DataFrame( {'precision': precision, 'recall': recall, 'threshold': threshold})
	return dfs

print( dfs )
print(metrics['b_accuracy'])
print(metrics['accuracy'])

""" 
Ratio, 	b_accuracy, 		accuracy, 		best precision, recall @ best precision, threshold @ best precision,
0.1, 	0.61526832955404387, 	0.22375811506665683, 	0.019024, 	0.470677, 	0.99
0.25, 	0.67573696145124718, 	0.36896117225162578,	0.043358, 	0.377444,	0.99
0.5, 	0.75283446712018143,	0.55577223905837114,	0.102529,  	0.335338,	0.99
1, 		0.80498866213151932, 	0.72286351474850319,	0.164605, 	0.260150, 	0.99
2, 		0.8344671201814059, 	0.84154605455353215,	0.369637, 	0.168421,	0.99
4, 		0.83522297808012091, 	0.90833295667885561,	0.428152, 	0.219549,	0.99
10, 	0.81632653061224492, 	0.97454109704566927, 	0.722772, 	0.109774, 	0.99

"""



ratio = 10
threshold = 0.96

#0.475645  0.249624       0.96

training_set = CL_design_matrix(df_name = 'train clf', sample_ratio = ratio)
nans_training = np.isnan(training_set['X']).any(axis=1)
X_t = training_set['X'][~nans_training]
Y_t = training_set['Y'][~nans_training]
idx_t = np.array(training_set['indices'])[~nans_training]
clf = run_RF(X_t, Y_t)
Y_pred = clf.predict_proba(X_unseen)

#chart of variable importance
importance = pd.DataFrame( {'importance': pd.Series(clf.feature_importances_), \
							'features': training_set['attributes'][1:] })
sns.barplot(y="importance", x="features", data = importance)


def get_suspects(Y_pred, threshold):
	Y_gt_threshold = np.zeros( (Y_pred.shape[0], ))
	Y_gt_threshold[ np.where( Y_pred[:,1] > threshold ) ] = 1

	Ids = Helpers.get_shelf('unseen clf'+dataframe_shelve, 'Id')
	assert len(unseen_set['indices']) == len(Ids)
	Ids = np.array( [i[1] for i in Ids] )

	suspect_idx = idx_unseen.take( np.where( Y_gt_threshold == 1 )[0] )
	suspects = Ids[suspect_idx]

	suspect_dict = {}
	for suspect in suspects:
		if suspect[0] in suspect_dict.keys():
			suspect_dict[suspect[0]].append(suspect[1])
		else:
			suspect_dict[suspect[0]] = [suspect[1]]
		if suspect[1] in suspect_dict.keys():
			suspect_dict[suspect[1]].append(suspect[0])
		else:
			suspect_dict[suspect[1]] = [suspect[0]]
	return [suspects, suspect_dict]

def suspect_threads(suspects):
	suspect_dict = {}
	DB = Rdb()
	DB.connect()

	for suspect in suspects:
		DB.db.execute(("""SELECT SubmissionId FROM CommentInfo 
							WHERE SubmissionId IN ( %s) and Author ='%s' """ )  % ('%(unseen sid)s', suspect) % Sdb.query)
		results = DB.db.fetchall()
		results = [r[0] for r in results]
		suspect_dict[suspect] = set(results)
	DB.close()
	return suspect_dict

suspect_l = get_suspects(Y_pred, 0.96)
suspect_dict = suspect_l[1]
suspect_submissions = suspect_threads( list(suspect_dict.keys()) )

index = np.array( list(suspect_submissions.keys()))
submission_jaccard_distance = np.zeros( (len(index), len(index)))
suspect_jaccard_distance = np.zeros( (len(index), len(index)))
sock_score = np.zeros( (len(index), len(index)))
sock_list = []
for i in np.arange(1,len(index), 1):
	for j in np.arange(i+1,len(index),1):
		v_i = suspect_submissions[index[i]]
		v_j = suspect_submissions[index[j]]
		submission_jaccard_distance[i,j] = Helpers.distance( v_i, v_j, 'jaccard' )
		v_i = suspect_dict[ index[i] ] + [index[i]]
		v_j = suspect_dict[ index[j] ] + [index[j]]
		suspect_jaccard_distance[i,j] = Helpers.distance( v_i, v_j, 'jaccard' )
		sock_score[i,j] = suspect_jaccard_distance[i,j] * submission_jaccard_distance[i,j]
		if sock_score[i,j] > 0.01:
			sock_list.append( [sock_score[i,j], [index[i],index[j]]] )

sock_list = sorted(sock_list, key=itemgetter(0), reverse = True)
