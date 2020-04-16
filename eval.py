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
	binary_preds = []
	for i in range(len(labels)):
		total += 1
		# spacy similarity
		pred = nlp(preds[i].strip("\n").lower())
		label = nlp(labels[i].strip("\n").lower())
		if 	pred.similarity(label) >= threshold:
		# difflib similarity
		# label = labels[i].strip("\n").lower()
		# pred = preds[i].strip("\n").lower()
		# if difflib.SequenceMatcher(None, pred, label).ratio() >= threshold:
			corrected += 1
			binary_preds.append(1)
		else:
			binary_preds.append(0)
	print("Accuracy: ", corrected / total)
	return binary_preds

def main():
	if len(sys.argv) != 4:
		print("[usage]: python3 eval.py preds_file_name labels_file_name threshold")
	else:
		# python3 eval.py data_processing/preds/roberta/rob_context_preds.txt data_processing/data/next1/dev_labels.txt .99
		binary_preds = eval(sys.argv[1], sys.argv[2], float(sys.argv[3]))
		output = open("xlnet_next1_binary_preds.txt", "w")
		for pred in binary_preds:
			output.write(str(pred) + "\n")
		output.close()

if __name__ == '__main__':
	main()