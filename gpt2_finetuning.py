import gpt_2_simple as gpt2
import os
import requests

def gpt2_finetuning(model_name, data_file, step):
	if not os.path.isdir(os.path.join("models", model_name)):
		print(f"Downloading {model_name} model...")
		gpt2.download_gpt2(model_name=model_name)   # model is saved into current directory under /models/124M/

	if not os.path.isfile(file_name):
		url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
		data = requests.get(url)
	
		with open(file_name, 'w') as f:
			f.write(data.text)
    

	sess = gpt2.start_tf_sess()

	'''
   	For finetuning use this command. Following are other parameters of the finetuning function:
        sess,
        dataset,
        steps=-1,
        model_name='124M',
        model_dir='models',
        combine=50000,
        batch_size=1,
        learning_rate=0.0001,
        accumulate_gradients=5,
        restore_from='latest',
        run_name='run1',
        checkpoint_dir='checkpoint',
        sample_every=100,
        sample_length=1023,
        sample_num=1,
        multi_gpu=False,
        save_every=1000,
        print_every=1,
        max_checkpoints=1,
        use_memory_saving_gradients=False,
        only_train_transformer_layers=False,
        optimizer='adam',
        overwrite=False
	'''
	gpt2.finetune(sess,
            file_name,
            model_name=model_name,
            steps=1000)   # steps is max number of training steps

def main():

	
if __name__ == '__main__':
	main()