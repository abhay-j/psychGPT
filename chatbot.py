import random
import numpy as np
import pickle
import json
import subprocess
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl','rb'))
classes = pickle.load((open('classes.pkl','rb')))
model = load_model('chatbot_model.h5')


def cleanup_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = cleanup_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[1]=1
    
    return bag


def predict_class(sentence):
    bow = bag_of_words(sentence)
   
    res = model.predict(np.array([bow]))[0]
   
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key= lambda x : x[1], reverse=True)
    result_list = []

    for r in results:
        result_list.append({'intent':classes[r[0]], 'probability':str(r[1])})
    return result_list


def get_response(intent_list, intent_json):
    tag = intent_list[0]['intent']
    list_of_intents = intent_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


print("Go Ahead")
with open('example.txt', 'w') as file:
         file.write('')
while True:
        
        message = input()
        
        

        if message.lower() != "bye":
            # Open file in append mode
            with open('example.txt', 'a') as file:
             # Write data to file
             file.write(message+'\n')
            ints = predict_class(message)
            res = get_response(ints, intents)
            print("Grump.AI: ",res)
        else:
            print('Grump.AI: bye')
            subprocess.call(["python", "chatgptapi.py"])
            break

# Positive: "My day was fantastic, thanks for asking! I woke up feeling refreshed and energized, and I was able to accomplish everything on my to-do list. I also caught up with a dear friend and had a great time chatting over lunch. Overall, it was a really fulfilling day!"
# Negative: "To be honest, my day was pretty rough. I woke up late and rushed through my morning routine, which put me in a bad mood from the get-go. Work was really stressful and I had a lot of deadlines to meet. On top of that, I had a disagreement with a friend which left me feeling upset. I'm just glad the day is over, to be honest."
# Neutral: "My day was fine, thanks for asking. I had a lot of work to do, but I managed to get it all done. Nothing particularly noteworthy happened, but it was a productive day overall."



