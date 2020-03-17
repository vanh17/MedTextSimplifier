import gpt_2_simple as gpt2
import os
import sys

def generate(run_name, prefix_sentence, output_file):
    sess = gpt2.start_tf_sess()
    """
    sess,
             run_name='run1',
             checkpoint_dir='checkpoint',
             model_name=None,
             model_dir='models',
             sample_dir='samples',
             return_as_list=False,
             truncate=None,
             destination_path=None,
             sample_delim='=' * 20 + '\n',
             prefix=None,
             seed=None,
             nsamples=1,
             batch_size=1,
             length=1023,
             temperature=0.7,
             top_k=0,
             top_p=0.0,
             include_prefix=True
    """
    gpt2.generate(sess,
                  run_name = run_name,
                  destination_path = output_file,
                  batch_size = 2,
                  length = 100,
                  prefix = prefix_sentence,
                  include_prefix = False,
                  truncate="<|endoftext|>"
                  )

def main():
    if len(sys.argv) < 4:
    	print('[usage] python gpt2_generate.py run_name, prefix_sentence, output_directory')
    	return None
    generate(sys.argv[1], sys.argv[2], sys.argv[3])
	
if __name__ == '__main__':
    main()