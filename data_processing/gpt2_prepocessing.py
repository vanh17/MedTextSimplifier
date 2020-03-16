import sys
import io

def main():
	if len(sys.argv) < 3:
		print("[usage]: python gpt2_preprocessing.py normal.aligned single.aligned processed.txt")
		return None
	with io.open(sys.argv[1], "r") as normal, io.open(sys.argv[2], "r") as simple, io.open(sys.argv[3], "w") as processed:
		normal = normal.read().splitlines()
		simple = simple.read().splitlines()
		for i in range(len(normal)):
			# add simple marker so that GPT know that we are converting one sentence to the others.
			processed.write("<|startoftext|>" + normal[i].split("\t")[2] + " [SEP] " + simple[i].split("\t")[2] + " <|endoftext|>\n")

if __name__ == '__main__':
	main()