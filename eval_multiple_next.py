import spacy
import sys
import difflib

# to run this first do pip install spacy, then download en_core_web_lg model
# by running this code: python -m spacy en_core_web_lg
# load en_core_web_lg model
nlp = spacy.load("en_core_web_lg")

def eval(preds, labels, threshold):
	corrected = 0
	total = 0
	preds = open(preds, "r").readlines()
	labels = open(labels, "r").readlines()
	for i in range(len(labels)):
		total += 1
		# initialize satisfactory
		satisfactory = False
		# spacy similarity
		pred_list = preds[i].strip("\n").lower().split(" ")
		# this label might be have multiple words at a time
		label_list = labels[i].strip("\n").lower().split(" ")
		count = 0
		for t in range(len(label_list)):
			pred = nlp(pred_list[t])
			label = nlp(label_list[t]) 
			if 	pred.similarity(label) >= threshold:
			# difflib similarity
			# label = labels[i].strip("\n").lower()
			# pred = preds[i].strip("\n").lower()
			# if difflib.SequenceMatcher(None, pred, label).ratio() >= threshold:
				count += 1
		if count == len(label_list):
			corrected += (count / len(label_list))
	print("Accuracy: ", corrected / total)

def main():
	if len(sys.argv) != 4:
		print("[usage]: python3 eval.py preds_file_name labels_file_name threshold")
	else:
		# python3 eval_multiple_next.py data_processing/preds/roberta/rob_context_preds_2.txt data_processing/data/next2/dev_labels_2.txt .99
		eval(sys.argv[1], sys.argv[2], float(sys.argv[3]))

if __name__ == '__main__':
	main()