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
		# spacy similarity
		pred = nlp(preds[i].strip("\n").lower())
		print(pred)
		label = nlp(labels[i].strip("\n").lower())
		print(label)
		if 	pred.similarity(label) >= threshold:
		# difflib similarity
		# label = labels[i].strip("\n").lower()
		# pred = preds[i].strip("\n").lower()
		# if difflib.SequenceMatcher(None, pred, label).ratio() >= threshold:
			corrected += 1
	print("Accuracy: ", corrected / total)

def main():
	if len(sys.argv) != 4:
		print("[usage]: python3 eval.py preds_file_name labels_file_name threshold")
	else:
		# python3 eval.py data_processing/preds/roberta/rob_context_preds.txt data_processing/data/dev_labels.txt .99
		eval(sys.argv[1], sys.argv[2], float(sys.argv[3]))

if __name__ == '__main__':
	main()