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
     plt.figure(figsize=(20, 10))
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

  def variance(self):
    df=self.load_and_save_data()
    var=df.var(ddof=0)[0]
    self.values['variance']=var
    return var

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

  def autocorrelation(self):
      df=self.load_and_save_data()
      size=len(df[self.y_src])
      lag_list=[]
      corr_list=[]
      for i in range (1,size-2):
        acf_lag=df[self.y_src].autocorr(lag=i)
        if (acf_lag>0.2):
            lag_list.append(i)
            corr_list.append(acf_lag)
      if (len(lag_list)==0):
         msg="There is no autocorrelation"
      else:
        # results = pd.DataFrame()
        # results['Lag'] = lag_list
        # results['autocorrelation'] = corr_list
        # results.sort_values(['autocorrelation'],  ascending=False, inplace=True)
        # msg= results
        msg = ["lagList : ", lag_list, "corrolationList : ", corr_list]
        self.values['autocorrolation']= msg
      return msg
  
  def q_value_MA_model(self):
    df=self.load_and_save_data()
    size=len(df[self.y_src])
    for i in range (1,size-2):
      acf_lag=df[self.y_src].autocorr(lag=i)
      if (acf_lag<=0.2):
        q=i
        #print("q_value_for_MA:")
        self.values["q_value_for_MA"]= q
        break
    return q

  def  p_value_AR_model(self):
    df=self.load_and_save_data()
    pacf_func=pacf(df[self.y_src],nlags=100)
    pacf_func.tolist()
    j=0
    for i in pacf_func:
      j=j+1
      if (i <=0.2):
        p=j
        #print("p_value_for_AR :")
        self.values["p_value_for_AR"]= p
        break
    return p

  def AR_model(self):
    p=self.p_value_AR_model()
    df=self.load_and_save_data()
    model = ARIMA(df,order=(p,0,0))
    results_AR = model.fit(disp=-1)
    plt.plot(df)
    plt.plot(results_AR.fittedvalues,color='red')
    plt.title('RSS: %.4f '% sum((results_AR.fittedvalues-df[self.y_src])**2))
    print('Plotting AR model')

  def MA_model(self):
    q=self.q_value_MA_model()
    df=self.load_and_save_data()
    model = ARIMA(df,order=(0,0,q))
    results_AR = model.fit(disp=-1)
    plt.plot(df)
    plt.plot(results_AR.fittedvalues,color='red')
    plt.title('RSS: %.4f '% sum((results_AR.fittedvalues-df[self.y_src])**2))
    print('Plotting MA model')

  def fourier_spectrum(self):
    df=self.load_and_save_data()
    yts=np.array(df[self.y_src].dropna().values, dtype=float)
    xts=np.array(pd.to_datetime(df[self.y_src].dropna()).index.values, dtype=float)
    xts1=pd.to_datetime(xts)
    # Fast Fourier Transformation (FFT) on the original signal
    global fourier
    fourier = fftpack.fft(yts)
    freq = fftpack.fftfreq(yts.size)
    # Filtring the frequency spectrum by eliminating the values under 800000
    fourier[np.abs(fourier)<800000] = 0 
    self.values['FFT_freq']=np.abs(freq)
    self.values['FFT_coeff']=np.abs(fourier)
    #return("Fourier transformation spectrum after being filtred : ", np.abs(freq), np.abs(fourier))
    return np.abs(freq), np.abs(fourier)

  def fourier_initial(self):
    df=self.load_and_save_data()
    self.fourier_spectrum()
    xts=np.array(pd.to_datetime(df[self.y_src].dropna()).index.values, dtype=float)
    xts1=pd.to_datetime(xts)
    filtered_signal=fftpack.ifft(fourier)
    self.values['fourier_initial_x']=xts1
    self.values['fourier_initial_y']=filtered_signal
    return("Regeneration of the initial signal : ", xts1, filtered_signal)





class NSFeaturizing (TS_Featurizing):
   def Linear_reg_trend(self):
      df=self.load_and_save_data()
      # Linear Regression of the trend
      decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
      trend = decomp.trend
      y = trend.array.dropna()
      x= range(len(y))
      # y=np.array(np.array(trend[self.y_src].dropna().values), dtype=float)
      # x=np.array(pd.to_datetime(trend[self.y_src].dropna()).index.values, dtype=float)
      slope, intercept, r_value, p_value, std_err =sp.linregress(x,y)
      xf = np.linspace(np.min(x),np.max(x),100)
      # xf1 = xf.copy()
      # xf1 = pd.to_datetime(xf1)
      # yf = slope xf + intercept (ax+b)
      # r_value , p_value,  std_err : The parameters of the regression
      # self.values['linear_regression_params']= (slope,intercept)
      self.values['linear_regression_slope']= slope
      self.values['linear_regression_intercept']= intercept
      return (slope, intercept,x,y)

   def std_Linear_reg_trend(self):
      df=self.load_and_save_data()
      slope=self.Linear_reg_trend()[0]
      intercept=self.Linear_reg_trend()[1]
      x=self.Linear_reg_trend()[2]
      y=self.Linear_reg_trend()[3]
      Y=(slope*x)+intercept
      e = y - Y
      print(type(e))
      # plt.plot(e)
      self.values['linear_regression_std']= e
      return e

   def seasonal_period(self):
      df=self.load_and_save_data()
      decomp = statsmodels.tsa.seasonal.seasonal_decompose(df, model='Additive')
      seasonal = decomp.seasonal
      # seasonal = self.SEASONAL_decompose()
      # Extracting the period and its values
      # A=np.array(seasonal[self.y_src].dropna().values, dtype=float)
      A = seasonal.array.dropna()
      minima = np.min(A)  # Minimal value
      min_id = np.where(A == minima) # Indexs of the minimal value
      start = min_id  # The index of the start of each period
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
      # resid = self.RESIDUAL_decompose()
      arr=np.linspace(0,1,20)
      quantile_r=resid.quantile(arr)
      #print('Residual quantiles : \n', quantile_r, '\n')
      QR=[]
      for i in range(len(quantile_r)):
        QR.append(quantile_r.array[i])
      self.values['residual_quantiles']= QR
      return QR



class SFeaturizing (TS_Featurizing):
  #Fitting a range of distribution and test for goodness of fit
 def distributions(self):
   df=self.load_and_save_data()
   #standardise the data using sklearn’s StandardScaler
   sc=StandardScaler() 
   yy = df[self.y_src].values.reshape (-1,1)
   sc.fit(yy)
   y_std =sc.transform(yy)
   y_std = y_std.flatten()
   y_std
   del yy

   # Turn off code warnings (this is not recommended for routine use)
   import warnings
   warnings.filterwarnings("ignore")

   size = len(df[self.y_src])


   dist_names = ['alpha','anglit','arcsine','argus','beta','betaprime','bradford','burr','burr12','cauchy','chi','chi2','cosine','crystalball','dgamma','dweibull','erlang','exponnorm',
              'expon','exponweib','exponpow','f','fatiguelife','fisk','foldcauchy','foldnorm','frechet_r','frechet_l','genlogistic','gennorm','genpareto','genexpon','genextreme','gausshyper',
              'gamma','gengamma','genhalflogistic','geninvgauss','gilbrat','gompertz','gumbel_r','gumbel_l','halfcauchy','halflogistic','halfnorm','halfgennorm','hypsecant','invgamma','invgauss',
              'invweibull','johnsonsb','johnsonsu','kappa4','kappa3','ksone','kstwobign','laplace','logistic','loggamma','loglaplace','lognorm','loguniform','lomax','maxwell','mielke','moyal','nakagami','ncx2','ncf','nct','norm','norminvgauss','pareto','pearson3','powerlaw','powerlognorm','powernorm','rdist','rayleigh','rice','recipinvgauss',
              'semicircular','skewnorm','t','trapz','triang','truncexpon','truncnorm','tukeylambda','uniform','vonmises','vonmises_line','wald','wrapcauchy','weibull_min', 'weibull_max','levy','levy_l'
              ]

   # Set up empty lists to stroe results
   chi_square = []
   p_values = []

   # Set up 31 bins for chi-square test
   # Observed data will be approximately evenly distrubuted aross all bins
   percentile_bins = np.linspace(0,100,31)
   percentile_cutoffs = np.percentile(y_std, percentile_bins)
   observed_frequency, bins = (np.histogram(y_std, bins=percentile_cutoffs))
   cum_observed_frequency = np.cumsum(observed_frequency)

   # Loop through candidate distributions

   for distribution in dist_names:
     # Set up distribution and get fitted distribution parameters
     dist = getattr(scipy.stats, distribution)
     param = dist.fit(y_std)
    
     # Obtain the KS test P statistic, round it to 5 decimal places
     p = scipy.stats.kstest(y_std, distribution, args=param)[1]
     p = np.around(p, 5)
     p_values.append(p)    
    
     # Get expected counts in percentile bins
     # This is based on a 'cumulative distribution function' (cdf)
     cdf_fitted = dist.cdf(percentile_cutoffs, *param[:-2], loc=param[-2], 
                          scale=param[-1])
     expected_frequency = []
     for bin in range(len(percentile_bins)-1):
        expected_cdf_area = cdf_fitted[bin+1] - cdf_fitted[bin]
        expected_frequency.append(expected_cdf_area)
    
     # calculate chi-squared
     expected_frequency = np.array(expected_frequency) * size
     cum_expected_frequency = np.cumsum(expected_frequency)
     ss = sum (((cum_expected_frequency - cum_observed_frequency ) ** 2) /  cum_observed_frequency )
     chi_square.append(ss)
        
   # Collate results and sort by goodness of fit (best at top)

   results = pd.DataFrame()
   results['Distribution'] = dist_names
   results['chi_square'] = chi_square
   results['p_value'] = p_values
   results.sort_values(['chi_square'], inplace=True)
         
   
   # Report results

   #print('\nDistributions sorted by goodness of fit:\n*********************************************************************************************\n',results)

   return results

 def get_distribution(self):
   results=self.distributions()
   i=0
   for namedTuple in results.itertuples():
     i+=1
     if (namedTuple[3]> 0.05) :
         #print(" Now we can say that the distributions that fit the most our serie is : ",namedTuple[1] )
         self.values["distribution_fitted"]= namedTuple[1]
         break
        
     if(i>98) :
        #print("There is no distribution that fit this serie")
        self.values["distribution_fitted"]= None
        
 def distribution_fit(self):
     df=self.load_and_save_data()
     results=self.distributions()
     self.get_distribution()
     dist_name= self.values.get("distribution_fitted")
     y= df[self.y_src]
     # Create an index array (x) for data
     x = np.arange(len(y))
     # Create an empty list to stroe fitted distribution parameters
     parameters = []

     # Set up distribution and store distribution paraemters
     dist = getattr(scipy.stats,dist_name)
     param = dist.fit(y)
     parameters.append(param)

     self.values["Parameters"]=parameters
     
     return dist_name, parameters


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


def get_window(series, backward=5, forward=0, slide=1, pad=False):
    series = list(series)
    s_len = len(series)
    sliding_window = []

    if pad:
        for i in range(backward):
            window = [np.nan for i in range(backward + 1 + forward)]
            sliding_window.append(window)

    for i in range(backward, s_len - forward, slide):
        window = series[i - backward:i + forward + 1]
        sliding_window.append(window)

    if pad:
        for i in range(forward):
            window = [np.nan for i in range(backward + 1 + forward)]
            sliding_window.append(window)

    return np.array(sliding_window)