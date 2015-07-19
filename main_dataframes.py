import gc
import os
import sys
from setup_experiments import *

if __name__ == '__main__':
	if len(sys.argv) > 1:
		index_type = str(sys.argv[1])
		df_only = (bool(int(sys.argv[2])) if len(sys.argv) > 2 else False)
		overwrite = (bool(int(sys.argv[3])) if len(sys.argv) > 3 else False)
		if index_type in ['train', 'unseen']:
			init_index(index_type)
			if os.path.isfile('num_topics.pkl'):
				with open('num_topics.pkl', 'rb') as f:
					lsa_num_topics, lda_num_topics = pickle.load(f)
			else:
				lsa_num_topics, lda_num_topics = 100, 6
			try:
				if index_type == 'unseen':
					if not df_only:
						CL_init_index(unseen = True)
						unseen_index = [i['indices'] for i in Index.Indices['unseen clf'] ]
						CL_set_base_models(unseen_index, lda_num_topics, lsa_num_topics)
						CL_set_features(Index.Indices['unseen clf'], 'unseen clf')
					gc.collect()
					CL_set_dataframe(shelf_prefix = 'unseen clf', overwrite = overwrite)
				else:
					if not df_only:
						CL_init_index()
						train_index = [i['indices'] for i in Index.Indices['train clf'] ]
						test_index = [i['indices'] for i in Index.Indices['test clf'] ]
						CL_set_base_models(train_index, lda_num_topics, lsa_num_topics)
						CL_set_features(Index.Indices['train clf'], 'train clf')
						CL_set_features(Index.Indices['test clf'], 'test clf')
					gc.collect()
					if overwrite:
						Helpers.del_shelf('train clf'+ dataframe_shelve)
						Helpers.del_shelf('test clf'+ datagrame_shelve)
					CL_set_dataframe(shelf_prefix = 'train clf', overwrite = overwrite)
					gc.collect()
					CL_set_dataframe(shelf_prefix = 'test clf', overwrite = overwrite)
				return 1
			except MemoryError:
				print("**** Unable to complete writing dataframe. Run again with df_only=1 and overwrite=0 to continue ****")
				return -1
		else:
			print("Invalid index type indicated. Exiting...")
	else:
		print("Index type not indicated. Exiting...")
