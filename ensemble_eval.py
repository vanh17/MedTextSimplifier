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
		if 	rob_pred.similarity(label) >= threshold or bert_pred.similarity(label) >= threshold or xlnet_pred.similarity(label) >= threshold:
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
		# python3 ensemble_eval.py data_processing/preds data_processing/data/next1/dev_labels.txt .99
		eval(sys.argv[1], sys.argv[2], float(sys.argv[3]))

if __name__ == '__main__':
	main()