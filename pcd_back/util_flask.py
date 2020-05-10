import key
from flask import request ,jsonify ,Flask
from flask_cors import CORS
import numpy as np
import pandas as pd
import scipy.stats as sp
import os 
from werkzeug.utils import secure_filename
import json
import glob
from tensorflow.keras.models import load_model
import h5py 

app= Flask(__name__)
CORS(app)

# Create a directory in a known location to save files 
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
    new_model= load_model ('LSTM_model.h5')
    dateparse = lambda x: pd.datetime.strptime(x, '%d/%m/%Y')
    df1= pd.read_csv('instance\\uploads\\'+ namefile, parse_dates=['date'], date_parser=dateparse,usecols=["date","a_withdr"], index_col=0, sep=';')
    new_test_set=key.get_window(df1["a_withdr"].values, backward=4)
    print(new_test_set)
    print("model loaded !")
    new_model.summary()
    input_timeseries1 = np.expand_dims(new_test_set, axis=2)
    output_timeseries1 = new_model.predict(x=input_timeseries1)
    dist1 = np.linalg.norm(new_test_set - output_timeseries1, axis=-1)
    ratio = 0.95
    sorted_scores = sorted(dist1)
    threshold = sorted_scores[round(len(dist1) * ratio)]
    print (threshold)
    test=df1.iloc[0:len(df1)]
    test_score_df1 = pd.DataFrame(index=test[4:].index)
    test_score_df1['loss'] = dist1
    test_score_df1['threshold'] = threshold
    test_score_df1['anomaly'] = test_score_df1.loss > test_score_df1.threshold
    test_score_df1['a_withdr'] = test[4:].a_withdr
    anomalies = test_score_df1[test_score_df1.anomaly == True]
    print(anomalies)
    x=key.to_json(Time_serie)
    return jsonify(x)

if __name__== "__main__":
    app.run(debug=True)

