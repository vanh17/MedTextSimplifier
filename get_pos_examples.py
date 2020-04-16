import sys

def get_pos_examples(word_lst, pos_lst):
	dict_pos = {}
	dict_ex = {}
	for i in range(len(pos_lst)):
		if pos_lst[i] in dict_pos:
			if len(dict_pos[pos_lst[i]]) <= 10:
				if word_lst[i].strip("\n") not in dict_pos[pos_lst[i]]:
					dict_pos[pos_lst[i]].append(word_lst[i].strip("\n"))
		else:
			dict_pos[pos_lst[i]] = [word_lst[i].strip("\n")]
	for key in dict_pos.keys():
		dict_ex[key] = dict_pos[key]
	print(dict_ex)
	

def main():
	# ppython3 get_pos_examples.py data_processing/data/next1/dev_labels.txt data_processing/data/tagged_testset.txt
	word_lst = open(sys.argv[1], "r")
	pos_lst = open(sys.argv[2], "r")
	get_pos_examples(word_lst.readlines(), pos_lst.readlines())

if __name__ == '__main__':
	main()