import spacy
import sys
import difflib
import random

# to run this first do pip install spacy, then download en_core_web_lg model
# by running this code: python -m spacy en_core_web_lg
# load en_core_web_lg model
nlp = spacy.load("en_core_web_lg")

def select_model(sequence_model):
	model_dict = {0: "RoBERTa", 1: "BERT", 2: "XLNet"}
	models = []
	for i in range(len(sequence_model)):
		if sequence_model[i] == "1":
			models.append(model_dict[i])
	if len(models) > 0:
		return random.choice(models)
	else:
		return "RoBERTa"

def eval(preds, labels, model_preds, threshold):
	corrected = 0
	total = 0
	rob_count = 0
	bert_count = 0
	xlnet_count = 0
	roberta_preds = open(preds+"/roberta/rob_context_preds.txt", "r").readlines()
	bert_preds = open(preds+"/bert/bert_context_preds.txt", "r").readlines()
	xlnet_preds = open(preds+"/xlnet/xlnet_context_preds.txt", "r").readlines()
	model_preds = open(model_preds, "r").readlines()
	labels = open(labels, "r").readlines()
	for i in range(len(labels)):
		total += 1
		sequence_model = model_preds[i+1].strip("\n").split("\t")[1:4]
		model = select_model(sequence_model)
		# spacy similarity
		rob_pred = nlp(roberta_preds[i].strip("\n").lower())
		bert_pred = nlp(bert_preds[i].strip("\n").lower())
		xlnet_pred = nlp(xlnet_preds[i].strip("\n").lower())
		label = nlp(labels[i].strip("\n").lower())
		if model == "RoBERTa":
			rob_count += 1
			if 	rob_pred.similarity(label) >= threshold:
				corrected += 1
		if model == "BERT":
			bert_count += 1
			if bert_pred.similarity(label) >= threshold:
				corrected += 1
		if model == "XLNet":
			xlnet_count += 1
			if xlnet_pred.similarity(label) >= threshold:
				corrected += 1
	print("Accuracy: ", corrected / total)
	print("RoBERTa used: ", rob_count / total)
	print("BERT used: ", bert_count / total)
	print("XLNet used: ", xlnet_count / total)

def main():
	if len(sys.argv) != 5:
		print("[usage]: python3 model_selection_ensemble_eval.py preds_folder_name labels_file_name model_preds_file threshold")
	else:
		# python3 model_selection_ensemble_eval.py data_processing/preds data_processing/data/next1/dev_labels.txt data_processing/preds/models/model_preds.txt .99
		eval(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]))

if __name__ == '__main__':
	main()