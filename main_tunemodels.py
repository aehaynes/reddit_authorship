import sys
import pickle
from setup_experiments import *


if __name__ == '__main__':
	if len(sys.argv) > 1:
		index_type = str(sys.argv[1])
		if index_type in ['train', 'unseen']:
			init_index(index_type)
			FE_set_samples()
			FE_set_tokens()
			lda_num_topics = tune_lda()
			lsa_num_topics = tune_lsa()			
			num_topics = [lsa_num_topics, lda_num_topics]
			with open("num_topics.pkl", 'wb') as f:
				pickle.dump(num_topics, f)
		else:
			print("Invalid index type indicated. Exiting...")
	else:
		print("Index type not indicated. Exiting...")

		 

