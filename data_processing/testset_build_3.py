import io
import sys

def test_set_build(simple, normal, context_test, no_context_test, testset1, testset2, output_test, output_test2):
    simple = open(simple, "r").readlines()
    normal = open(normal, "r").readlines()
    context_test = open(context_test, "w")
    no_context_test = open(no_context_test, "w")
    output_test = open(output_test, "w")
    context_test2 = open(testset1, "w")
    no_context_test2 = open(testset2, "w")
    output_test2 = open(output_test2, "w")
    for i in range(len(simple)):
        no_context = []
        context = []
        labels = []
        s_sent = simple[i].strip("\n").split(" ")
        d_sent = normal[i].strip("\n")
        if i >= 500:
            context_test = context_test2
            no_context_test = no_context_test2
            output_test = output_test2  
        for t in range(0, len(s_sent)-4, 3):
            # fix algorithm here to include next 3 words
            temp_array = s_sent[:t+3] + ["[MASK]"] * (len(s_sent)-t-4) + [s_sent[-1]]
            label = " ".join(s_sent[t+3:t+3+min(3,len(s_sent)-t-4)])
            no_context.append(" ".join(temp_array))
            context.append(d_sent + " " + " ".join(temp_array))
            labels.append(label)
        if len(labels) != len(context) or len(labels) != len(no_context) or len(context) != len(no_context):
            print(len(labels), len(context), len(no_context))
            break
        for i in range(len(labels)):
            context_test.write(context[i] + "\n")
            no_context_test.write(no_context[i] + "\n")
            output_test.write(labels[i] + "\n")
    context_test.close()
    no_context_test.close()

def main():
    if len(sys.argv) != 9:
        print("Need 9 arguments: [usage] python3 testset_build_3.py simple.txt normal.txt dev_context_3.txt dev_no_context_3.txt test_context_3.txt test_no_context_3.txt dev_labels_3.txt test_labels_3.txt")
    else:
        test_set_build(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], sys.argv[8])

main()