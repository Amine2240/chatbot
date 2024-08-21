import pickle
from tensorflow.keras.models import load_model # type: ignore
import json
import random
import os
import tensorflow as tf

# Suppress TensorFlow logging (only show errors)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# tf.get_logger().setLevel('ERROR')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

intents = json.load(open('intents.json'))

words = pickle.load(open('words.pkl' , 'rb'))
classes = pickle.load(open('classes.pkl' , 'rb'))
fitted_cv = pickle.load(open('fitted_cv.pkl' , 'rb'))
model = load_model('mychatbot_model.keras')

def predict_class(sentence):
  sentence = [sentence]
  x_test = fitted_cv.transform(sentence).toarray()
  y_pred = model.predict(x_test)[0]
  error_threshold = 0.25
  results = [[i , r] for i,r in enumerate(y_pred) if r > error_threshold] # i , r => index, proba
  results.sort(key=lambda x:x[1] , reverse=True)
  # print(results)
  # print(y_pred)
  dicts_list = []
  for r in results:
    dicts_list.append({'intent' : classes[r[0]] , 'proba' : str(r[1])})
  
  # print(dicts_list)
  return dicts_list

def get_response(dicts_list, intents_json):
  response = ''
  tag = dicts_list[0]['intent'] # the item with the highest proba, it is sorted inversly
  for intent in intents_json['intents']:
    if intent['tag'] == tag:
      response = random.choice(intent['responses'])
      break
   
  return response
  

while True:
  message = input('write your message : ')
  ints = predict_class(message)
  res = get_response(ints , intents)
  print(res)