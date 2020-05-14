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
    colonne2= key.firstElmtFirstLine('instance\\uploads\\'+namefile).split(';')[1][:-1]
    #create a dict
    Time_serie={}
    Time_serie_1=key.TS_Featurizing(namefile,colonne1,colonne2,format_date,Time_serie)
    Time_serie_1.load_and_save_data()

    #common descriptors
    Time_serie_1.std()
    Time_serie_1.min()
    Time_serie_1.mean()
    Time_serie_1.max()
    Time_serie_1.median()
    Time_serie_1.variance()
    Time_serie_1.TREND_decompose()
    Time_serie_1.SEASONAL_decompose()
    Time_serie_1.RESIDUAL_decompose()
    # Time_serie_1.autocorrelation()
    Time_serie_1.q_value_MA_model()
    Time_serie_1.p_value_AR_model()
    # Time_serie_1.AR_model()
    # Time_serie_1.MA_model()
    Time_serie_1.fourier_spectrum()
    # Time_serie_1.fourier_initial()
    
    # Stationnarity test
    Time_serie_1.stationarity_test()

    #STS and NSTS descriptors
    if Time_serie["stationnarity"] == True:
      Time_serie_s=key.SFeaturizing(namefile,colonne1,colonne2,format_date,Time_serie)
      Time_serie_s.get_distribution()
      Time_serie_s.distribution_fit()
    elif Time_serie["stationnarity"] == False:
      Time_serie_ns=key.NSFeaturizing(namefile,colonne1,colonne2,format_date,Time_serie)
      Time_serie_ns.Linear_reg_trend()
      ### Time_serie_ns.std_Linear_reg_trend()
      Time_serie_ns.seasonal_period()
      Time_serie_ns.residual_describe()
    
    #intégration de modéle deep
    new_model= load_model ('LSTM_model.h5')
    dateparse = lambda x: pd.datetime.strptime(x, format_date)
    df1= pd.read_csv('instance\\uploads\\'+ namefile, parse_dates=[colonne1], date_parser=dateparse,usecols=[colonne1,colonne2], index_col=0, sep=';')
    new_test_set=key.get_window(df1[colonne2].values, backward=4)
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

    #return jsonify(x)
    return json.loads(x)

if __name__== "__main__":
    app.run(debug=True)

