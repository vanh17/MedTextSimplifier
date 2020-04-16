'''
	Author: Hoang Van
	Description: POS tagger for tagging parallel corpus for
	type of word prediction analysis
	This will create a file with one POS tag for each prediction
	task of next 1 word.
'''

# import flair and python necessary package
from flair.data import Sentence
from flair.models import SequenceTagger
import re
import sys

def pos_tagging(string, pos_tagger,regex):
	# find special token such as -LRB- -RRB-
	special_tokens = regex.findall(string)
	# remove - so that tagger wont think they are punctuation.
	for token in special_tokens:
		string = string.replace(token, token[1:-1])
	sentence = Sentence(string, use_tokenizer=False)
	pos_tagger.predict(sentence)
	tagged_sent = sentence.to_tagged_string().split(" ")
	return [[tagged_sent[i] for i in range(2, len(tagged_sent)-1) if i % 2 == 1], len(tagged_sent) == 2*len(string.split(" "))]

def main():
	if len(sys.argv) != 3:
		# python3 pos_tagger.py data_processing/data/simple.txt  data_processing/data/tagged_testset.txt
		print("[usage]: python3 pos_tagger.py preds_file_name  tagged_output_name")
	else:
		regex = re.compile(r'-[A-Za-z]+-')
		pos_tagger = SequenceTagger.load('pos')
		tagged_sents = []
		simple_sents = open(sys.argv[1],"r").readlines()
		print("start printing")
		for i in range(500):
			if i % 50 == 0:
				print("Done " +str(i))
			result = pos_tagging(simple_sents[i].strip("\n"), pos_tagger, regex)
			tagged_sents += result[0] 
			if not result[1]:
				print(i)
				break
		tagged_output_name = open(sys.argv[2], "w")
		print("start writing")
		for pos in tagged_sents:
			tagged_output_name.write(pos+"\n")
		tagged_output_name.close()
		print("Success!")

if __name__ == '__main__':
	main()