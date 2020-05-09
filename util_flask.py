import util
import os
import json
from flask import request ,jsonify ,Flask

app= Flask(__name__)

@app.route('/descripteurs',methods=['POST'])
def descripteurs():
    message = request.json
    namefile = message['namefile']
    colonne1= message['colonne1']
    colonne2= message['colonne2']
    format_date=message['format_date']

# Dict
    Time_serie={}     
    # Time_serie_stat={}
# Read object 
    Time_serie_1=util.TS_Featurizing(namefile,colonne1,colonne2,format_date,Time_serie)

# Common descriptors
    Time_serie_1.mean()
    # Time_serie_1.plot_serie()
    # Time_serie_1.std()
    Time_serie_1.min()
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
    # Time_serie_1.fourier_spectrum()
    # Time_serie_1.fourier_initial()

# Stationnarity test
    Time_serie_1.stationarity_test()

#STS and NSTS descriptors
    if Time_serie["stationnarity"] == True:
      Time_serie_s=util.SFeaturizing(namefile,colonne1,colonne2,format_date,Time_serie)
      Time_serie_s.get_distribution()
      Time_serie_s.distribution_fit()
    elif Time_serie["stationnarity"] == False:
      Time_serie_ns=util.NSFeaturizing(namefile,colonne1,colonne2,format_date,Time_serie)
      Time_serie_ns.Linear_reg_trend()
      ### Time_serie_ns.std_Linear_reg_trend()
      Time_serie_ns.seasonal_period()
      Time_serie_ns.residual_describe()

    # print(Time_serie)
    x=util.to_json(Time_serie)
    return json.loads(x)

if __name__== "__main__":
  app.run(debug=True)

    

