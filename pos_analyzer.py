import sys

def pos_analyse(binary_preds, pos_lst):
	dict_pos = {}
	dict_accuracy = {}
	for i in range(len(pos_lst)):
		if pos_lst[i] in dict_pos:
			dict_pos[pos_lst[i]][0] += int(binary_preds[i])
			dict_pos[pos_lst[i]][1] += 1
		else:
			dict_pos[pos_lst[i]] = [0, 1]
	for key in dict_pos.keys():
		dict_accuracy[key] = dict_pos[key][0]/dict_pos[key][1]
	print(dict_accuracy)

def main():
	# python3 pos_analyzer.py rob_next1_binary_preds.txt data_processing/data/tagged_testset.txt
	binary_preds = open(sys.argv[1], "r")
	pos_lst = open(sys.argv[2], "r")
	pos_analyse(binary_preds.readlines(), pos_lst.readlines())

if __name__ == '__main__':
	main()