
import nltk
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
from torch.nn import init
import codecs
import pickle
import torch.autograd as autograd
import torch.nn.utils.rnn
import datetime
import operator
from tqdm import tqdm

np.random.seed(0)


def Data(path):
    question = []
    ans = []
    label = []
    with codecs.open(path, encoding='utf-8', mode='r') as file1:

        for i in file1:
            q, a, l= i.strip().split("\t")
            question.append(q)
            ans.append(a)
            label.append(l)

    #ques = []
    #ans1 = []
    #label1 = []
    #for i in range(len(label)):
        #if label[i] == '1':
           # label1.append(label[i])
           # ques.append(question[i])
           # ans1.append(ans[i])
    
    #for i in range(18):
        #label.extend(label1)
        #question.extend(ques)
        #ans.extend(ans1)
    #print(len(label))
    data = pd.DataFrame(zip(question,ans,label),columns = ['context','uttrance','label'])
    return data



def Shuffle(data):
    data.reindex(np.random.permutation(data.index))
    


def create_vocab(data):
    vocab = []
    freq = {}

    for idx, row in data.iterrows():
        context_cell = row['context']
        response_cell = row["uttrance"]
        train_words = str(context_cell).split() + str(response_cell).split()

        for word in train_words:
            if word.lower() not in vocab:
                vocab.append(word.lower())

            if word.lower() not in freq:
                freq[word.lower()] = 1

            else:
                freq[word.lower()] += 1
    
    freq_sorted = sorted(freq.items(), key=lambda item: item[1], reverse = True)
    
    vocab = ["<UNK>"] + [pair[0] for pair in freq_sorted]
    return vocab
##########################################################################



############################################################################
def create_word_to_id(vocab):
    word_to_id = {word: id for id, word in enumerate(vocab)}
    return word_to_id
###########################################################################


##############################################################################
def create_id_to_vec(word_to_id,glovefile):
       
    lines = codecs.open(glovefile, encoding="utf-8", mode='r').readlines()

    id_to_vec = {}
    vector = None

    for line in lines:
        word = line.split()[0]
        vector = np.array(line.split()[1:],dtype = 'float32')

        if word in word_to_id:
            id_to_vec[word_to_id[word]]  = torch.FloatTensor(torch.from_numpy(vector))
  
    for word, id in word_to_id.items():
        if word_to_id[word] not in id_to_vec:
            v = np.zeros(*vector.shape, dtype = 'float32')
            v[:] = np.random.randn(*v.shape)*0.01
            id_to_vec[word_to_id[word]] = torch.FloatTensor(torch.from_numpy(v))

    embedding_dim = id_to_vec[0].shape[0]

    return id_to_vec , embedding_dim
##################################################################################


##################################################################################
def load_ids_and_labels(row, word_to_id):
    context_ids = []
    response_ids = []

    context_cell = row['context']
    response_cell = row['uttrance']
    label_cell = row['label']

    max_context_len = 160

    context_words = context_cell.split()
    if len(context_words) > max_context_len:
        context_words = context_words[:max_context_len]

    for word in context_words:
        if word in word_to_id:
            context_ids.append(word_to_id[word])    
        else:
            context_ids.append(0)

    response_words = response_cell.split()
    if len(response_words)>max_context_len:
        response_words = response_words[:max_context_len]
    for word in response_words:
        if word in word_to_id:
            response_ids.append(word_to_id[word]) 
        else:
            response_ids.append(0)                         

    label = np.array(label_cell).astype(np.float32)
    return context_ids, response_ids, label

class Encoder(nn.Module):
    def __init__(self,emb_size,hidden_size,vocab_size,p_dropout):
        super(Encoder,self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.p_dropout = p_dropout
        self.embedding = nn.Embedding(self.vocab_size,self.emb_size)
        self.lstm = nn.LSTM(self.emb_size,self.hidden_size)
        self.dropout_layer = nn.Dropout(self.p_dropout)
        
        self.init_weights()
    
    def init_weights(self):
        init.uniform(self.lstm.weight_ih_l0,a=-0.01,b=0.01)
        init.orthogonal(self.lstm.weight_hh_l0)
        self.lstm.weight_ih_l0.requires_grad = True
        self.lstm.weight_hh_l0.requires_grad = True

        embedding_weights = torch.FloatTensor(self.vocab_size,self.emb_size)

        for id, vec in id_to_vec.items():
            embedding_weights[id] = vec

        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad = True)

    def forward(self,inputs):
        #inputs = inputs.transpose(1,0)
        embeddings = self.embedding(inputs)

        _, (last_hidden, _) = self.lstm(embeddings)

        last_hidden = self.dropout_layer(last_hidden[-1])
        
        return last_hidden

class DualEncoder(nn.Module):
    def __init__(self, encoder):
        super(DualEncoder, self).__init__()
        self.encoder = encoder
        self.hidden_size = self.encoder.hidden_size
        M = torch.FloatTensor(self.hidden_size, self.hidden_size)
        init.xavier_normal(M)
        self.M = nn.Parameter(M, requires_grad = True)


    def forward(self, context_tensor, response_tensor):

        
        context_last_hidden  = self.encoder(context_tensor)
        response_last_hidden  = self.encoder(response_tensor)

        context = torch.matmul(context_last_hidden, self.M)
        context = context.view(-1,1,self.hidden_size)

        response = response_last_hidden.view(-1, self.hidden_size, 1)

        score = torch.bmm(context, response).view(-1,1)

        return score
def creating_varibales():
    print(str(datetime.datetime.now()).split('.')[0],'creating variables for training and validation..')
    training_dataframe = Data("E:\chat bot intern\week 5\WikiQA-train.txt")
    vocab = create_vocab(training_dataframe)
    word_to_id = create_word_to_id(vocab)
    id_to_vec, emb_dim = create_id_to_vec(word_to_id,'E:\chat bot intern\week 5\glove.txt')

    validation_dataframe = Data("E:\chat bot intern\week 5\WikiQA-dev.txt")
    print(str(datetime.datetime.now()).split('.')[0],'variable created.\n')

    return training_dataframe,vocab,word_to_id,id_to_vec,emb_dim,validation_dataframe





def creating_model(hidden_size, p_dropout):
    print(str(datetime.datetime.now()).split('.')[0],'calling model...')

    encoder = Encoder(emb_size = emb_dim,hidden_size = hidden_size,vocab_size=len(vocab),p_dropout=p_dropout)
    dual_encoder = DualEncoder(encoder)
    print(str(datetime.datetime.now()).split('.')[0],"Model created.\n")
    print(dual_encoder)
    return encoder, dual_encoder


def increase_count(correct_count, score,label):
    if ((score.data[0][0]>=0.5) and (label.data[0][0]==1.0)) or ((score.data[0][0]<0.5) and(label.data[0][0]==0.0)):
        correct_count += 1

    return correct_count

def get_accuracy(correct_count,data):
    accuracy = correct_count/len(data)
    return accuracy


def train_model(learning_rate, l2_penalty, epochs):
    print(str(datetime.datetime.now()).split('.')[0],'starting training and validation....')
    
    
    optimizer = torch.optim.Adam(dual_encoder.parameters(),lr = learning_rate, weight_decay = l2_penalty)
    loss_func = torch.nn.BCEWithLogitsLoss()
    best_validation_accuracy = 0.0

    for epoch in range(epochs):
        Shuffle(training_dataframe)
        sum_loss_training  = 0.0
        training_correct_count = 0
        dual_encoder.train()
        iterator = tqdm(training_dataframe.iterrows(), total = len(training_dataframe))     
        
        for idx, row in iterator:
            context_ids, response_ids, label = load_ids_and_labels(row,word_to_id)
            context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1), requires_grad = False)
            response = autograd.Variable(torch.LongTensor(response_ids).view(-1,1), requires_grad = False)
            label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1))), requires_grad = False)
            optimizer.zero_grad()
            score = dual_encoder(context,response)
            loss = loss_func(score,label)
            sum_loss_training += loss.item()
            loss.backward()
            optimizer.step()
            
            training_correct_count = increase_count(training_correct_count,score,label)


        training_accuracy = get_accuracy(training_correct_count,training_dataframe)

        Shuffle(validation_dataframe)

        validation_correct_count = 0
        sum_loss_valiadtion = 0.0
        dual_encoder.eval()
        iterator = tqdm(validation_dataframe.iterrows())
        for idx, row in iterator:
            context_ids, response_ids, label = load_ids_and_labels(row,word_to_id)
            context = autograd.Variable(torch.LongTensor(context_ids).view(-1,1))
            response = autograd.Variable(torch.LongTensor(response_ids).view(-1,1))
            label = autograd.Variable(torch.FloatTensor(torch.from_numpy(np.array(label).reshape(1,1))))
            score = dual_encoder(context,response)
            loss = loss_func(score,label)
            sum_loss_valiadtion += loss.item()
            validation_correct_count = increase_count(validation_correct_count,score,label)

        validation_accuracy = get_accuracy(validation_correct_count,validation_dataframe)     
        print(str(datetime.datetime.now()).split('.')[0],
              "epoch: %d/%d" % (epoch,epochs),
              "training loss: %.3f" % (sum_loss_training/len(training_dataframe)),
              "training accuracy: %.3f" % (training_accuracy),
              "vat loss: %.3f" % (sum_loss_valiadtion/len(validation_dataframe)),
              "val accuracy: %.3f" % (validation_accuracy))

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(dual_encoder.state_dict(),"saved_model_trial2.pt")
            print("New best founed and saved")

    print(str(datetime.datetime.now()).split('.')[0],'training and validation epohces finished..')        

training_dataframe,vocab,word_to_id,id_to_vec,emb_dim,validation_dataframe = creating_varibales()
encoder, dual_encoder =  creating_model(hidden_size=40,p_dropout=0.85)
#encoder.cuda()
#dual_encoder.cuda()

for name, param in dual_encoder.named_parameters():
    if param.requires_grad:
        print(name)

train_model(learning_rate=0.0001,l2_penalty=0.0001,epochs=1)

a= dual_encoder.M.detach().numpy()
np.save('selfM.npy', np.array(a))    
dual_encoder.load_state_dict(torch.load('saved_model_trial2.pt', map_location=torch.device('cpu')))

dual_encoder.eval()
encoder.eval()

pickle.dump(word_to_id,open('word_to_id.pkl', 'wb'))

iterator = tqdm(training_dataframe.iterrows())
responseMatrix = []
responses = []
for index, row in iterator:
    responses.append(row['uttrance'])
    context_ids, response_ids,label = load_ids_and_labels(row,word_to_id)
    response_ids = np.array(response_ids)[:, None]
    with torch.no_grad():
        output = encoder(torch.from_numpy(response_ids))
        
        
    responseMatrix.append(output.numpy())    


with open('allResponses.pkl','wb') as f:
    pickle.dump(responses, f)

np.save('responseMatrix.npy', np.array(responseMatrix))    
torch.onnx.export(encoder, torch.from_numpy(response_ids),"model.onnx", verbose=True,input_names=['data'], output_names=['output'],dynamic_axes={'data': [0]})   

exit(0)
















