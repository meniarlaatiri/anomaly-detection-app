import key
from flask import request ,jsonify ,Flask
from flask_cors import CORS
import numpy as np
import pandas as pd
import scipy.stats as sp
import os 
from werkzeug.utils import secure_filename
import json
import h5py 

app= Flask(__name__)
CORS(app)

# Create a directory in a known location to save files to
uploads_dir = os.path.join(app.instance_path,'uploads')
os.makedirs(uploads_dir, exist_ok=True)
@app.route('/upload', methods=['POST'])
def upload():
    #Receive the file sended by the post method
    file= request.files['file']
    print(file)
    message= request.form["data"]
    data= json.loads(message)
    format_date=data["format_date"]
    namefile=file.filename
    file.save(os.path.join(uploads_dir, secure_filename(file.filename)))
    colonne1= key.firstElmtFirstLine('instance\\uploads\\'+namefile).split(';')[0]
    print(colonne1)
    colonne2= key.firstElmtFirstLine('instance\\uploads\\'+namefile).split(';')[1][:-1]
    #les descripteurs de la série temporelle en retour
    Time_serie={}
    Time_serie_1=key.TS_Featurizing(namefile,colonne1,colonne2,format_date,Time_serie)
    Time_serie_1.load_and_save_data()
    Time_serie_1.min()
    Time_serie_1.stationarity_test()
    x=key.to_json(Time_serie)
    new_model= load_model ("LSTM_model.h5")
    print("model loaded !")
    new_model.summary()
    return jsonify(x)

if __name__== "__main__":
    app.run(debug=True)

