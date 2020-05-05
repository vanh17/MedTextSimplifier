import spacy
import sys
import difflib

# to run this first do pip install spacy, then download en_core_web_lg model
# by running this code: python -m spacy en_core_web_lg
# load en_core_web_lg model
nlp = spacy.load("en_core_web_lg")

def eval(preds, labels, threshold, output, text_file):
	# initialize the preds list and labels for populating the dataset
	roberta_preds = open(preds+"/roberta/rob_context_preds.txt", "r").readlines()
	bert_preds = open(preds+"/bert/bert_context_preds.txt", "r").readlines()
	xlnet_preds = open(preds+"/xlnet/xlnet_context_preds.txt", "r").readlines()
	labels = open(labels, "r").readlines()
	# write down headers for output file: 
	output.write("Text\tRoBERTa\tBERT\tXLnet\n")
	for i in range(len(labels)):
		# strip newline characters and add tab
		data_line = text_file[i].strip("\n")
		# spacy similarity
		rob_pred = nlp(roberta_preds[i].strip("\n").lower())
		bert_pred = nlp(bert_preds[i].strip("\n").lower())
		xlnet_pred = nlp(xlnet_preds[i].strip("\n").lower())
		label = nlp(labels[i].strip("\n").lower())
		rob_label = "0"
		bert_label = "0"
		xlnet_label = "0"
		if 	rob_pred.similarity(label) >= threshold:
			rob_label = "1"
		if bert_pred.similarity(label) >= threshold:
			bert_label = "1"
		if xlnet_pred.similarity(label) >= threshold:
			xlnet_label = "1"
		output.write(data_line+"\t"+rob_label+"\t"+bert_label+"\t"+xlnet_label+"\n")

def main():
	if len(sys.argv) != 6:
		print("Error: wrong number of parameters")
	else:
		# python3 model_classification_data_prep.py data_processing/preds data_processing/data/next1/dev_labels.txt .99 data_processing/data/next1/dev_no_context.txt data_processing/data/model_selection_data/model_train.txt
		output = open(sys.argv[5], "w")
		text_file = open(sys.argv[4], "r").readlines()
		eval(sys.argv[1], sys.argv[2], float(sys.argv[3]), output, text_file)

if __name__ == '__main__':
	main()