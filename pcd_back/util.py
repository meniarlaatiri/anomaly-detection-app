# Import libraries 
import numpy as np
import pandas as pd
import scipy.stats as sp
import matplotlib.dates as mdates
import math
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import to_datetime
import scipy.stats
from scipy import fftpack
import statsmodels.tsa.stattools
import statsmodels.tsa.seasonal
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.preprocessing import StandardScaler
import json
import csv

def firstElmtFirstLine(file):
    with open(file, 'r') as f:
        ligne = f.readline()
        premierElement = ligne.split(',')[0]
        return premierElement

#classe mére 
class TS_Featurizing :
  def  __init__(self,src,x_src,y_src,date_format,values = dict()):
     self.src = src
     self.x_src =x_src
     self.y_src =y_src
     self.date_format=date_format
     self.values =values

  # load data set and save to DataFrame
  def load_and_save_data(self):
     dateparse = lambda x: datetime.strptime(x,self.date_format)
     df = pd.read_csv('instance\\uploads\\'+ self.src, parse_dates=[self.x_src], date_parser=dateparse,usecols=[self.x_src,self.y_src], index_col=0, sep=';')
     df= df.dropna()
     return df

  # simple line plot
  def plot_serie(self):
     df=self.load_and_save_data()
     fig = plt.figure(figsize=(20, 10))
     plt.plot(df, color='blue')
     plt.title('Time Serie', fontsize=24)
     plt.ylabel(self.y_src)
     plt.xlabel(self.x_src)
     show = plt.show()
     return show

  def mean(self):
     df=self.load_and_save_data()
     mean= df.mean()[0]
     self.values['mean']=mean
     return mean

  def std(self):
    df=self.load_and_save_data()
    std = df.std()[0]
    self.values['std']=std
    return std

  def min(self):
    df=self.load_and_save_data()
    min = df.min()[0]
    self.values['min']=min
    return min

  def max(self):
    df=self.load_and_save_data()
    max = df.max()[0]
    self.values['max']=max
    return max

  def median(self):  
    df=self.load_and_save_data()
    median = df.median()[0]
    self.values['median']=median
    return median
        # Decomposition to TREND & SEASONALITY & RESIDUAL
  def TREND_decompose(self):
    df=self.load_and_save_data()
    decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
    trend = decomp.trend
    return trend

  def SEASONAL_decompose(self):
    df=self.load_and_save_data()
    decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
    seasonal = decomp.seasonal
    return seasonal

  def RESIDUAL_decompose(self):
    df=self.load_and_save_data()
    decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
    resid = decomp.resid
    return resid

  #Dickey-Fuller test
  def stationarity_test(self):
    df=self.load_and_save_data()
    dftest = adfuller(df[self.y_src], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['DF Statistic', 'p-value', 'Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    if dfoutput[0] < dftest [4]["5%"]:
        result=print ('Result of Dickey-Fuller test:\nTime Series is Stationary')
        self.values["stationnarity"]= True 
    else:
        result=print ('Result of Dickey-Fuller test:\nTime Series in Non-Stationary' )
        self.values["stationnarity"]= False 
    return result
class NSFeaturizing (TS_Featurizing):
   def Linear_reg_trend(self):
      df=self.load_and_save_data()
      # Linear Regression of the trend
      decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
      trend = decomp.trend
      y=np.array(np.array(trend[self.y_src].dropna().values), dtype=float)
      x=np.array(pd.to_datetime(trend[self.y_src].dropna()).index.values, dtype=float)
      slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
      xf = np.linspace(np.min(x),np.max(x),100)
      xf1 = xf.copy()
      xf1 = pd.to_datetime(xf1)
      yf = (slope*xf)+intercept  # yf = slope x + intercept (ax+b)
      # r_value , p_value,  std_err : The parameters of the regression
      self.values['linear_regression_params']= (slope,intercept)
      return slope, intercept,x,y

   def std_Linear_reg_trend(self):
      df=self.load_and_save_data()
      slope=self.Linear_reg_trend()[0]
      intercept=self.Linear_reg_trend()[1]
      x=self.Linear_reg_trend()[2]
      y=self.Linear_reg_trend()[3]
      Y=(slope*x)+intercept
      e = y - Y
      plt.plot(e)
      self.values['linear_regression_std']= e
      return e

   def seasonal_period(self):
      df=self.load_and_save_data()
      decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
      seasonal = decomp.seasonal
      # Extracting the period and its values
      A=np.array(seasonal[self.y_src].dropna().values, dtype=float)
      minima = np.min(A)  # Minimal value
      min_id = np.where(A == minima) # Index of the minimal value
      start = min_id  # The index of the star of each period
      period = []
      for i in range(start[0][1]-start[0][0]):
        period.append(A[i])
        self.values['seasonality']= period
      return period
      
   def residual_describe(self):
      #20 quantiles of the residual component
      df=self.load_and_save_data()
      decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
      resid = decomp.resid
      arr=np.linspace(0,1,20)
      quantile_r=resid.quantile(arr)
      #print('Residual quantiles : \n', quantile_r, '\n')
      QR=[]
      for i in range(len(quantile_r)):
        QR.append(quantile_r.iloc[i,0])
      self.values['residual_quantiles']= QR
      return QR

class NpEncoder(json.JSONEncoder):
   def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def to_json(mydict):
  metadata= json.dumps(mydict, cls=NpEncoder)
  return metadata 