from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
import random
import json
import pickle
import numpy as np

import nltk
nltk.download('omw-1.4')
nltk.download('punkt')

lematizador = WordNetLemmatizer()
intenciones = json.loads(open('intenciones.json').read())

palabras = pickle.load(open('palabras.pkl', 'rb'))
clases = pickle.load(open('clases.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def limpiar_sentencia(sentencia):
    palabras_sentencia = nltk.word_tokenize(sentencia)
    palabras_sentencia = [lematizador.lemmatize(
        word) for word in palabras_sentencia]
    return palabras_sentencia


def bag_of_words(sentencia):
    palabras_sentencia = limpiar_sentencia(sentencia)
    bag = [0] * len(palabras)
    for w in palabras_sentencia:
        for i, word in enumerate(palabras):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predecir_clase(sentencia):
    bow = bag_of_words(sentencia)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r]for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(
            {'itencion': clases[r[0]], 'probabilidad': str(r[1])})
        return return_list


def get_respuesta(intenciones_list, intenciones_json):
    tag = intenciones_list[0]['itencion']
    lista_de_intenciones = intenciones_json['intenciones']
    for i in lista_de_intenciones:
        if i['tag'] == tag:
            return random.choice(i['respuestas'])
    pass


def RecognizeColection(k):
    Colection = {
        'proyectos',
        'proyecto',
        'usuarios',
        'usuario',
        'reportes',
        'reporte',
        'modulos',
        'modulo'
    }
    for y in Colection:
        if (y in k):
            return True
    return False


def GetBeneficiario(message):
    EvitateWords = {
        'beneficiarios',
        'beneficiario',
        'tambogrande',
        'apellidado',
        'proyectos',
        'encuentra',
        'muestrame',
        'principal',
        'proyecto',
        'usuarios',
        'apellido',
        'proyecto',
        'nombrado',
        'reportes',
        'reporte',
        'modulos',
        'usuario',
        'sullana',
        'llamado',
        'nombres',
        'enlista',
        'muestra',
        'buscame',
        'modulo',
        'nombre',
        'metros',
        'senora',
        'señora',
        ' cuyo ',
        ' como ',
        'senor',
        'piura',
        'señor',
        ' con ',
        'busca',
        'halla',
        'lista',
        ' fin ',
        ' de ',
        ' del ',
        ' en ',
        ' un ',
        ' el ',
        ' la '
    }
    message = ' ' + message + ' '
    for elem in EvitateWords:
        message = message.replace(elem, '')
        message = ' ' + message + ' '
    message = message.replace('  ', '').strip()
    if message != '':
        return message
    else:
        pass


def BusquedaDeNumbers(Text):
    Sintext = Text.split()
    Num = ""
    for NWord in Sintext:
        if (NWord.isdigit()):
            Num = Num + NWord
    if (Num != ""):
        if (len(Num) == 8):
            return {'DNI': Num}
        elif (len(Num) == 9 or len(Num) == 6):
            return {'Telefono': Num}
    else:
        pass
    
    
def TratadoDeDatos(mensaje):
    JsonAws = {}
    JsonAws['Query'] = str(mensaje)
    try:
        #Busca si el texto contiene números
        ContainsNumbers = BusquedaDeNumbers(mensaje)
        #Si no contiene números
        if (ContainsNumbers == None):
            #Realiza la predicción de la clase
            ints = predecir_clase(mensaje.lower()) #Lo volvemos minusculas
            JsonAws['Intents'] = ints
            #Obtiene la respuesta de la predicción
            res = get_respuesta(ints, intenciones)
            #Si se encontro respuesta
            if res != None:
                if 'campos' in res:
                    #Si se espera el nombre de un beneficiario
                    if 'beneficiario' in res["campos"]:
                        #Si se encontro un beneficiario en el query
                        if GetBeneficiario(mensaje) != None : 
                            #Se asigna al query
                            res["campos"]["beneficiario"]["Value"] = GetBeneficiario(mensaje)
            #Se envia el resultado
            JsonAws['Answer'] = res
        #Si contiene números
        else:
            #Encapsulamos en una variable la lista de llaves
            ListaDeKeys = list(ContainsNumbers.keys())
            #Si es un DNI
            if ListaDeKeys[0] == 'DNI':
                #Agrega con formato de respuesta
                JsonAws['Answer'] = {
                        "collection": "proyectos",
                        "campos": {
                            "dni": {
                                "Value": ContainsNumbers['DNI'], #Envia el DNI
                                "Operator": "="
                            }
                        }
                    }
            #Si es un Telefono
            elif ListaDeKeys[0] == 'Telefono':
                #Agrega con formato de respuesta
                JsonAws['Answer'] = {
                        "collection": "proyectos",
                        "campos": {
                            "telefono": {
                                "Value": ContainsNumbers['Telefono'], #Envia el telefono
                                "Operator": "="
                            }
                        }
                    }
    #Si ocurre un error
    except Exception as err:
        #Envia el error
        JsonAws['Error'] = f"Unexpected {err=}, {type(err)=}"
    return JsonAws



app = Flask(__name__)

@app.route("/ApiQuestIA", methods=['GET'])
def home():
    mensaje = str(request.args['Query']).strip()
    return jsonify(TratadoDeDatos(mensaje))

if __name__ == '__main__':
    app.run()
