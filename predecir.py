# -*- coding: utf-8 -*-

from flask import Flask
import pickle
import numpy as np
import imp

app = Flask(__name__)

@app.route('/')
def clasificar():
     
    file = open("modelo_knn.mod", "r")
    knn = pickle.load(file)
    file.close()
     
     
    sl = 4.2
    sw = 3.8
    pl = 0.9
    pw = 4.4
     
    datos = np.array([[sl, sw, pl, pw]])
    predictions = knn.predict(datos)
    
    return predictions[0] 

if __name__ == '__main__':
   app.run()
