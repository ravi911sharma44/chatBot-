import os
from flask import Flask, request
import json
from flask_cors import CORS
import base64
import pickle
import numpy as np
import onnxruntime as ort
import torch
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

lemmatizer  = WordNetLemmatizer()
porter = PorterStemmer()

responseMatrix = np.load('responseMatrix.npy',allow_pickle=True)[:,0:]



    

selfM= np.load('selfM.npy')
ort_session = ort.InferenceSession('model.onnx')
word_to_id = pickle.load(open('word_to_id.pkl', 'rb'))

with open('allResponses.pkl', 'rb') as f:
    response = pickle.load(f)

app = Flask(__name__)
CORS(app)
@app.route('/')
def check():
    return 'hello'
@app.route('/api/', methods = ['POST'])
def chat():
    data = request.data.decode('utf-8')
    




    def bestResponse(contextMatrix):
        output = ort_session.run(None, {'data': contextMatrix})[0]
        
        output = output.dot(selfM)
        
        return np.argmax(np.matmul(output, responseMatrix.T)[0])    


    def pre_process(context):
        max_context_len = 160
        context_ids = []
        context_words = context.split()
        if len(context_words) > max_context_len:
            context_words = context_words[:max_context_len]
        for word in context_words:
            if word in word_to_id:
                context_ids.append(word_to_id[word])
            else:
                context_ids.append(0)
        
        return np.array(context_ids)


    def test(context):
        contextMatrix = pre_process(context)[:,None]
        bestResp = bestResponse(contextMatrix)
        return response[bestResp]    



    C = test(data)

    return C


    