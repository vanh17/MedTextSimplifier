import argparse
import zipfile
import sklearn.metrics
import pandas as pd
import re
from simpletransformers.classification import MultiLabelClassificationModel


emotions = ["RoBERTa", "BERT", "XLNet"]
emotion_to_int = {"0": 0, "1": 1}

model = MultiLabelClassificationModel('roberta', 'roberta-base', num_labels=3,
    args={'train_batch_size':4, 'gradient_accumulation_steps':8, 'learning_rate': 1e-5, 'num_train_epochs': 5, 'max_seq_length': 512, 'reprocess_input_data': True, 'overwrite_output_dir': True})


def train_and_predict(train_data: pd.DataFrame,
                      dev_data: pd.DataFrame) -> pd.DataFrame:

    # doesn't train anything; just predicts 1 for all of dev set
    model.train_model(train_data, output_dir="model_selector/")
    dev_predictions = dev_data.copy()
    preds, outputs = model.predict(dev_predictions["Text"])
    for i in range(len(dev_predictions["Text"])):
        for e in range(len(emotions)):
            dev_predictions.at[i, emotions[e]] = preds[i][e]
    return dev_predictions

def tweets_cleaner(tweet):
    tweet = re.sub(r'@[A-Za-z0-9_]+','[user]',tweet)
    tweet = re.sub(r"http\S+", "[website]", tweet)
    tweet = re.sub(r"[0-9]*", "[number]", tweet)
    tweet = re.sub(r"(”|“|-|\+|`|#|,|;|\|)*", "", tweet)
    tweet = re.sub(r"&amp", "", tweet)
    tweet = tweet.lower()   
    return tweet

if __name__ == "__main__":
    # gets the training and test file names from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs='?', default="data_processing/data/model_selection_data/model_train.txt")
    parser.add_argument("test", nargs='?', default="data_processing/data/model_selection_data/model_dev.txt")
    args = parser.parse_args()

    # reads train and dev data into Pandas data frames
    read_csv_kwargs = dict(sep="\t",
                           converters={e: emotion_to_int.get for e in emotions})
    train_data = pd.read_csv(args.train, **read_csv_kwargs)
    test_data = pd.read_csv(args.test, **read_csv_kwargs)

    # clean Tweets by replacing @mention to user, https to website, [0-9]* to number, remove abundant symbols
    for i in range(len(train_data["Text"])):
        train_data.at[i, "Text"] = tweets_cleaner(train_data["Text"][i])
    for i in range(len(test_data["Text"])):
        test_data.at[i, "Text"] = tweets_cleaner(test_data["Text"][i])
    

    # processing train data
    train_data['labels'] = list(zip(train_data["RoBERTa"].tolist(), train_data["BERT"].tolist(), train_data["XLNet"].tolist()))
    train_data['text'] = train_data['Text']

    # processing test data
    test_data['labels'] = list(zip(test_data["RoBERTa"].tolist(), test_data["BERT"].tolist(), test_data["XLNet"].tolist()))
    test_data['text'] = test_data['Text']

    # makes predictions on the dev set
    test_predictions = train_and_predict(train_data, test_data)

    # prints out multi-label accuracy
    print("accuracy: {:.3f}".format(sklearn.metrics.jaccard_similarity_score(
        test_data[emotions], test_predictions[emotions])))