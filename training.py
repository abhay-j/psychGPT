
import random 
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

"""
from tensorflow.keras.models import Sequential: This line of code imports the Sequential class from the models module of the Keras API within the TensorFlow library. Sequential is a type of model that allows you to create a neural network by adding layers to it in a sequential manner.
from tensorflow.keras.layers import Dense, Activation, Dropout: This line of code imports three layer types, Dense, Activation, and Dropout, from the layers module of the Keras API within the TensorFlow library.
Dense is a layer type that represents a fully connected layer in a neural network.
Activation is a layer type that applies an activation function to the output of the previous layer.
Dropout is a regularization technique that randomly drops out (sets to zero) a certain percentage of the neurons in the previous layer during each training iteration. This helps prevent overfitting.
from tensorflow.keras.optimizers import SGD: This line of code imports the SGD optimizer from the optimizers module of the Keras API within the TensorFlow library. SGD stands for Stochastic Gradient Descent,
which is an optimization algorithm commonly used in neural network training. It works by iteratively adjusting the weights of the network in the direction of the negative gradient of the loss function with respect to the weights.
"""
lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())


words = []
documents = []
classes=[]
ignore_words = ['?','!',',','.']
lemmaized_words = []
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list,intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])



for word in words:
    if word not in ignore_words:
        lemmaized_words.append(lemmatizer.lemmatize(word))

lemmaized_words = sorted(set(lemmaized_words))
classes = sorted(set(classes))

pickle.dump(lemmaized_words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower ( )) for word in word_patterns]
    """print(f'these are the words in the word_pattern after lemmatizing:')"""
    for word in word_patterns:
        """print(word)"""
    for word in lemmaized_words:
        bag.append(1) if word in word_patterns else bag.append(0)
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag,output_row])
        
random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(65, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]),activation='softmax'))



sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist=model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5',hist)
print('done')



