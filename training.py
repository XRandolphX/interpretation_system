import random
import json
import pickle
import numpy as np


import nltk
nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lematizador = WordNetLemmatizer()

intenciones = json.loads(open('intenciones.json').read())

palabras = []
clases = []
documentos = []
letras_ignoradas = ['?', '!', '.', ',', ';']

for intent in intenciones['intenciones']:
    for pattern in intent['patrones']:
        lista_de_palabras = nltk.word_tokenize(pattern)
        palabras.extend(lista_de_palabras)
        documentos.append((lista_de_palabras, intent['tag']))
        if intent['tag'] not in clases:
            clases.append(intent['tag'])


palabras = [lematizador.lemmatize(
    word) for word in palabras if word not in letras_ignoradas]
palabras = sorted(set(palabras))

clases = sorted(set(clases))

pickle.dump(palabras, open('palabras.pkl', 'wb'))
pickle.dump(clases, open('clases.pkl', 'wb'))

entrenamiento = []
empty_output = [0]*len(clases)

for document in documentos:
    bag = []
    patrones_de_palabras = document[0]
    patrones_de_palabras = [lematizador.lemmatize(
        word.lower()) for word in patrones_de_palabras]
    for word in palabras:
        bag.append(1) if word in patrones_de_palabras else bag.append(0)

    output_row = list(empty_output)
    output_row[clases.index(document[1])] = 1
    entrenamiento.append([bag, output_row])

random.shuffle(entrenamiento)
entrenamiento = np.array(entrenamiento)

train_x = list(entrenamiento[:, 0])
train_y = list(entrenamiento[:, 1])

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x),np.array(train_y), epochs = 300, batch_size=5, verbose = 1)
model.save('chatbotmodel.h5',hist)

