from flask import Flask, request, redirect, url_for, flash, jsonify, make_response
import numpy as np
import pickle as p
import json

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import io
from flask_cors import CORS
from flask_cors import cross_origin

app = Flask(__name__)
# CORS(app)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,PATCH,OPTIONS')
    response.headers.add("Access-Control-Allow-Headers",
                       "Access-Control-Allow-Headers, Origin,Accept, X-Requested-With, Content-Type, Access-Control-Request-Method, Access-Control-Request-Headers");
    return response

# app.config['CORS_HEADERS'] = 'Content-Type'
# cors = CORS(app, resources={r"/api": {"origins": "http://127.0.0.1:5000"}})


def corsapp_route(path, origin=('127.0.0.1',), **options):
    """
    Flask app alias with cors
    :return:
    """

    def inner(func):
        def wrapper(*args, **kwargs):
            print (request.method)
            if request.method == 'OPTIONS':
                response = make_response()
                response.headers.add("Access-Control-Allow-Origin", ', '.join(origin))
                response.headers.add('Access-Control-Allow-Headers', ', '.join(origin))
                response.headers.add('Access-Control-Allow-Methods', ', '.join(origin))
                return response
            else:
                result = func(*args, **kwargs)
            if 'Access-Control-Allow-Origin' not in result.headers:
                result.headers.add("Access-Control-Allow-Origin", ', '.join(origin))
            return result

        wrapper.__name__ = func.__name__

        if 'methods' in options:
            if 'OPTIONS' in options['methods']:
                return app.route(path, **options)(wrapper)
            else:
                options['methods'].append('OPTIONS')
                return app.route(path, **options)(wrapper)

        return wrapper

    return inner

@app.route('/', methods=['POST'])
def makecalc():

    data = request.get_json()


    text = data
    tlen = len(text.split(" "))

    target = text.split(" ")[tlen - 1]
    tokenized_text = tokenizer.tokenize(text)

    # Mask a token that we will try to predict back with `BertForMaskedLM`
    masked_index = tokenized_text.index(target)
    tokenized_text[tlen-1] = '[MASK]'

    # Convert token to vocabulary indices
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
    segments_ids = [1] * len(tokenized_text)
    # this is for the dummy first sentence.
    segments_ids[0] = 0
    segments_ids[1] = 0

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    # Load pre-trained model (weights)

    model.eval()

    # Predict all tokens
    predictions = model(tokens_tensor, segments_tensors)
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])

    print("Original:", text)
    print("Masked:", " ".join(tokenized_text))

    print("Predicted token:", predicted_token)
    prediction = predicted_token
    print("Other options:")
    # just curious about what the next few options look like.
    for i in range(10):
        predictions[0, masked_index, predicted_index] = -11100000
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
        print(predicted_token)



    response = jsonify(prediction)
    # response.headers.add("Access-Control-Allow-Origin", "*")
    return response

if __name__ == '__main__':
    # modelfile = 'models/final_prediction.pickle'
    # model = p.load(open(modelfile, 'rb'))
    modelpath = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(modelpath)
    model = BertForMaskedLM.from_pretrained(modelpath)
    app.run(debug=True, host='127.0.0.1')