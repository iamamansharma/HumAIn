from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import random
from making_examples_of_booking import *
from making_examples_of_greeting import *
from making_examples_of_goodbye import *
from making_examples_of_thanks import *
from making_examples_of_payments import *
from making_examples_of_nfc import *
from making_examples_of_cancel import *
from making_examples_of_opentoday import *
from nltk.tokenize import word_tokenize
from interaction import *
from trie import *

example_of_booking, length_of_booking = get_booking()   # 0
example_of_greeting, length_of_greeting = get_greeting()# 1 
example_of_goodbye, length_of_goodbye = get_goodbye()   # 2
example_of_thanks, length_of_thanks = get_thanks()      # 3
example_of_payment, length_of_payment = get_payment()   # 4
example_of_nfc, length_of_nfc = get_nfc()               # 5
example_of_cancel, length_of_cancel = get_cancel()      # 6
example_of_opentoday, length_of_opentoday = get_opentoday() # 7
number_of_classes = 8

stemmer = LancasterStemmer()

# making the corpus dynamically or X
corpus = example_of_booking+example_of_greeting+example_of_goodbye+example_of_thanks+example_of_payment+example_of_nfc+example_of_cancel+example_of_opentoday

stemming_corpus = []
for sentence in corpus:
    tokenize_word = word_tokenize(sentence)
    stemmed_sentence = [stemmer.stem(w.lower()) for w in tokenize_word if w!='?']
    stemming_corpus.append(' '.join(stemmed_sentence))

corpus = stemming_corpus

y = length_of_booking+length_of_greeting+length_of_goodbye+length_of_thanks+length_of_payment+length_of_nfc+length_of_cancel+length_of_opentoday
y_fianl = []
for i in y:
    tmp = [0]*number_of_classes
    tmp[i] = 1
    y_fianl.append(tmp)
y = y_fianl

sentiments = array(y)

all_words = []
for sent in corpus:
    tokenize_word = word_tokenize(sent)
    tokenize_word = [stemmer.stem(w.lower()) for w in tokenize_word if w!='?']
    for word in tokenize_word:
        all_words.append(word)


unique_words = set(all_words)

# set vocab length
vocab_length = len(unique_words) + 10

embedded_sentences = [one_hot(sent, vocab_length) for sent in corpus]

word_count = lambda sentence: len(word_tokenize(sentence))
longest_sentence = max(corpus, key=word_count)
length_long_sentence = len(word_tokenize(longest_sentence))

padded_sentences = pad_sequences(embedded_sentences, length_long_sentence, padding='post')


def build_model(input_dim, output_classes):
    model = Sequential()
    model.add(Embedding(vocab_length+1, 20, input_length=length_long_sentence))
    model.add(Flatten())
    model.add(Dense(20, activation='relu', input_shape = (20, )))
    model.add(Dense(20, activation='relu', input_shape = (20, )))    
#   model.add(Dense(input_dim=input_dim, output_dim=12, activation='relu'))
#    model.add(Dropout(0.5))
    model.add(Dense(output_dim=output_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

model = build_model(length_long_sentence, number_of_classes)
model.fit(padded_sentences, sentiments, epochs=100, verbose=1)
print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n')
print('Railway chatbot')
print()
mobile_number = input("Please Enter your mobile number: ")
print()
while True:
    user_query = input_for_chat()
    if user_query == 'exit':
        break
    print()
    print()
    print(user_query)
    print()
    user_query_copy = user_query[:]
    tokenize_word = word_tokenize(user_query)
    user_query = [stemmer.stem(w.lower()) for w in tokenize_word if w!='?']
    user_query = ' '.join(user_query)
    test = one_hot(user_query, vocab_length)
    test = pad_sequences([test], length_long_sentence, padding = 'post')
    res = model.predict(test)
    res = numpy.argmax(res)
    if res == 0:
        book_ticket(user_query_copy)
    elif res == 1:
        greetings()
    elif res == 2:
        goodbye()
    elif res == 3:
        thanks()
    elif res == 4:
        payment()
    elif res == 5:
        nfc()
    elif res == 6:
        cancel()
    elif res == 7:
        isopen()
    else:
        print('yo!')
