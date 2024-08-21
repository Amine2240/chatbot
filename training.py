import json
import spacy
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout 
# from tensorflow.keras.callbacks import Earlystopping
import tensorflow.keras as keras
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


intents = json.loads(open('intents.json').read())

nlp = spacy.load('en_core_web_sm')
words = []
classes = [] # list of tags
documents = []
ignored_words = ['?' , '!' , ',' ,';' , '.']


for intent in intents['intents']:
  for pattern in intent['patterns']:
    words_list = nlp(pattern)
    lemmatized_words = [token.lemma_ for token in words_list if token.text not in ignored_words]
    words.extend(lemmatized_words) # add the content of the list instead of the list
    lemmatized_words = " ".join(lemmatized_words) 
    documents.append((lemmatized_words , intent['tag']))
    if intent['tag'] not in classes:
      classes.append(intent['tag'])
    
words = sorted(set(words))
classes = sorted(set(classes))
# print(documents)
pickle.dump(words ,open('words.pkl' , 'wb'))
pickle.dump(classes , open('classes.pkl' , 'wb'))
texts = [doc[0] for doc in documents]
labels = [doc[1] for doc in documents]

cv = CountVectorizer(lowercase=True) 
X_train = cv.fit_transform(texts).toarray()
y_train = pd.get_dummies(labels)
# print(X_train[5])
#print(X_train.shape)
model = Sequential([
  Dense(128 , activation='relu' , input_shape=(X_train.shape[1],)),
  Dropout(0.5),
  Dense(64 , activation='relu'),
  Dropout(0.5),
  Dense(y_train.shape[1], activation='softmax')
]) # it is a classification model that predicts tags cotegories


callback = keras.callbacks.EarlyStopping(monitor='loss' , patience = 5)

model.compile(optimizer='adam' , loss='categorical_crossentropy' , metrics=['accuracy'])
model.summary()
history = model.fit(X_train , y_train , epochs=200 , batch_size = 5 , callbacks=[callback] )

model.save('mychatbot_model.keras')
print('done')


# Save the fitted CountVectorizer
pickle.dump(cv , open('fitted_cv.pkl' , 'wb'))
