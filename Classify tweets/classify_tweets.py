import nltk
import pandas as pd 
import numpy as np
from collections import Counter
import re
import random

# functii 
def tokenize(text):
    text = re.sub(r'\W',' ',text)
    text = re.sub(r'\s+',' ',text)
    return text
#
def clean_text(corpus):
    '''elimina spatii albe si semne de punctuatie'''
    for i in range(len(corpus)):
        corpus[i] = tokenize(corpus[i])
    return corpus
#
def create_dictionary(corpus):
    corpus = clean_text(corpus)
    dictionar = {}
    for sentence in corpus:
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token not in dictionar.keys():
                dictionar[token] = 1
            else:
                dictionar[token] += 1
    return dictionar
# 
def get_representation(toate_cuvintele, how_many):
    '''
        Extract the first most common words from a vocabulary
        and return two dictionaries: word to index and index to word
            @  che  .   ,   di  e
        text0  0   1   0   2   0   1
        text1  1   2 ...
        ...
        textN  0   0   1   1   0   2
    '''
    toate_cuvintele = Counter(toate_cuvintele)
    most_comm = toate_cuvintele.most_common(how_many)
    wd2idx = {}
    idx2wd = {}
    for idx, itr in enumerate(most_comm):
        cuvant = itr[0]
        wd2idx[cuvant] = idx
        idx2wd[idx] = cuvant
    return wd2idx, idx2wd
#
def text_to_bow(text, wd2idx):
    '''
        Convert a text to a bag of words representation.
            @  che  .   ,   di  e
        text   0   1   0   2   0   1
    '''
    features = np.zeros(len(wd2idx))
    tokens = tokenize(text).split()
    for token in tokens:
        if token in wd2idx:
            features[wd2idx[token]] += 1
    return features
#
def corpus_to_bow(corpus, wd2idx):
    '''
        Convert a corpus to a bag of words representation.
            @  che  .   ,   di  e
        text0  0   1   0   2   0   1
        text1  1   2 ...
        ...
        textN  0   0   1   1   0   2
    '''
    all_features = []
    for text in corpus:
        all_features.append(text_to_bow(text, wd2idx))
    all_features = np.array(all_features)
    return all_features
# 
def split_data(data, labels, procentaj_valid=0.25):
    '''
        Imparte random(shuffle) datele de antrenare in 75% train, 25% valid
        :param: procentaj_valid - procentajul multimii de validare
    '''
    labels = np.array(labels)
    indici = np.arange(0, len(labels))
    random.shuffle(indici)
    N = int((1 - procentaj_valid) * len(labels))
    train_data = data[indici[:N]]
    valid_data = data[indici[N:]]
    train_labels = labels[indici[:N]]
    valid_labels = labels[indici[N:]]
    return train_data, valid_data, train_labels, valid_labels
#
def dist_euclid_array(test_vector, train_data):
    '''
        :param train_data: vector de vectori de frecventa
        :return distantele euclidiene dintre test_vector si fiecare element din train_data
    '''
    diff = test_vector.astype(np.float64) - train_data.astype(np.float64)
    diff = diff **2
    distante = np.sqrt(np.sum(diff,axis=1))
    return distante
# 
def dist_manhatan_array(test_vector, train_data):
    '''
        :param train_data: vector de vectori de frecventa
        :return distantele manhatan dintre test_vector si fiecare element din train_data
    '''
    diff = test_vector.astype(np.float64) - train_data.astype(np.float64)
    distante = np.sum(np.abs(diff), axis=1)
    return distante
#
def classify_tweet(test_tweet, train_data, train_labels, k=3, metric='l2'):
    '''functie care intoarce predictia pentru test_tweet'''
    distante = []
    if metric == 'l2':
        distante = dist_euclid_array(test_tweet, train_data)
    else:
        distante = dist_manhatan_array(test_tweet, train_data)
    k_nearest_indices = np.argsort(distante)[:k]
    k_nearest_labels = train_labels[k_nearest_indices]
    vect_frecv = np.bincount(k_nearest_labels)
    test_tweet_label = np.argmax(vect_frecv)
    return test_tweet_label
#
def get_prediction(test_data, train_data, train_labels, k=3, metric='l2'):
    ''':return: predictia pentru fiecare tweet de test'''
    predicted_labels = []
    for test_tweet in test_data:
        label = classify_tweet(test_tweet,train_data,train_labels,k=k,metric=metric)
        predicted_labels.append(label)
    return predicted_labels
#
def write_prediction(out_file, predictions):
    '''
        A function to write the predictions to a file.
        id,label
        5001,1
        5002,1
        5003,1
        ...
    '''
    with open(out_file, 'w') as fout:
        
        fout.write('id,label\n')
        start_id = 5001
        for i, pred in enumerate(predictions):
            linie = str(i + start_id) + ',' + str(pred) + '\n'
            fout.write(linie)
    
#
def confusion_matrix(y_true, y_pred):
    '''
        Returneaza matricea de confuzie (se considera y_true si y_pred de dimensiuni egale)
    '''
    m = np.zeros((2, 2))
    for i in range(0, len(y_true)):
        m[y_true[i]][y_pred[i]] += 1
    return m
#
def prediction_accuracy(y_true, y_pred):
    ''' accuracy of a prediction '''
    score = 0
    for i in range(0, len(y_true)):
        if y_pred[i] == y_true[i]:
            score +=1
    accuracy = score/len(y_true)
    return accuracy 
#---------------------------------------------------------------------------------

train_df = pd.read_csv('train.csv') 
test_df = pd.read_csv('test.csv')

corpus = train_df['text']
labels = train_df['label']

toate_cuvintele = create_dictionary(list(corpus))
wd2idx, idx2wd = get_representation(toate_cuvintele, 100)

data = corpus_to_bow(corpus, wd2idx)
labels = train_df['label']

# Modelul folosit: kNN cu k=5 si metrica 'l2'

# Antrenarea hiperparametrului k pe o multime de validare

train_data, valid_data, train_labels, valid_labels = split_data(data,labels) # crearea multimilor train si validare
valid_predicitons = get_prediction(valid_data,train_data,train_labels,k=5,metric='l2')
matrice_confuzie = confusion_matrix(y_true=valid_labels, y_pred=valid_predicitons)
accuracy = prediction_accuracy(y_true=valid_labels, y_pred=valid_predicitons)
print("Accuracy: "+ str(accuracy))
print("Matrice confuzie :\n" + str(matrice_confuzie))


# Predictia pentru datele de test

test_data = corpus_to_bow(test_df['text'], wd2idx)
test_predicitons = get_prediction(test_data,train_data,train_labels,k=5,metric='l2')

write_prediction('sample_submission.csv', test_predicitons)




