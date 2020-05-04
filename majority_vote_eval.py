import spacy
import sys
import difflib
import random

# to run this first do pip install spacy, then download en_core_web_lg model
# by running this code: python -m spacy en_core_web_lg
# load en_core_web_lg model
nlp = spacy.load("en_core_web_lg")

def majority_vote(rob_pred, bert_pred, xlnet_pred):
	rob_count = 0
	bert_count = 0
	xlnet_count = 0
	if rob_pred.similarity(bert_pred) >= .85:
		rob_count+=1
		bert_count+=1
	if rob_pred.similarity(xlnet_pred) >= .85:
		rob_count+=1
		xlnet_count+=1
	if bert_pred.similarity(xlnet_pred) >= .85:
		bert_count+=1
		xlnet_count+=1
	count_array = [rob_count, bert_count, xlnet_count]
	major = max(count_array)
	finalist = []
	if rob_count == major:
		finalist.append(rob_pred)
	if bert_count == major:
		finalist.append(bert_pred)
	if xlnet_count == major:
		finalist.append(xlnet_pred)
	return random.choice(finalist)

def eval(preds, labels, threshold):
	corrected = 0
	total = 0
	roberta_preds = open(preds+"/roberta/rob_context_preds.txt", "r").readlines()
	bert_preds = open(preds+"/bert/bert_context_preds.txt", "r").readlines()
	xlnet_preds = open(preds+"/xlnet/xlnet_context_preds.txt", "r").readlines()
	labels = open(labels, "r").readlines()
	for i in range(len(labels)):
		total += 1
		# spacy similarity
		rob_pred = nlp(roberta_preds[i].strip("\n").lower())
		bert_pred = nlp(bert_preds[i].strip("\n").lower())
		xlnet_pred = nlp(xlnet_preds[i].strip("\n").lower())
		label = nlp(labels[i].strip("\n").lower())
		pred = majority_vote(rob_pred, bert_pred, xlnet_pred)
		if 	pred.similarity(label) >= threshold:
		# difflib similarity
		# label = labels[i].strip("\n").lower()
		# pred = preds[i].strip("\n").lower()
		# if difflib.SequenceMatcher(None, pred, label).ratio() >= threshold:
			corrected += 1
	print("Accuracy: ", corrected / total)

def main():
	if len(sys.argv) != 4:
		print("[usage]: python3 eval.py preds_folder_name labels_file_name threshold")
	else:
		# python3 majority_vote_eval.py data_processing/preds data_processing/data/next1/dev_labels.txt .99
		eval(sys.argv[1], sys.argv[2], float(sys.argv[3]))

if __name__ == '__main__':
	main()