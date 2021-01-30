import scipy.stats as st
from scipy.fftpack import fft, fftfreq 
from scipy.signal import argrelextrema
import operator
import numpy as np
import pandas as pd

def stat_area_features(x, Te=1.0):

    mean_ts = np.mean(x, axis=1).values.reshape(-1,1) # mean
    max_ts = np.amax(x, axis=1).values.reshape(-1,1) # max
    min_ts = np.amin(x, axis=1).values.reshape(-1,1)# min
    std_ts = np.std(x, axis=1).values.reshape(-1,1) # std
    skew_ts = st.skew(x, axis=1).reshape(-1,1) # skew
    kurtosis_ts = st.kurtosis(x, axis=1).reshape(-1,1) # kurtosis 
    iqr_ts = st.iqr(x, axis=1).reshape(-1,1) # interquartile rante
    mad_ts = np.median(np.sort(abs(x - np.median(x, axis=1).reshape(-1,1)), axis=1), axis=1).reshape(-1,1)# median absolute deviation
    area_ts = np.trapz(x, axis=1, dx=Te).reshape(-1,1) # area under curve
    sq_area_ts = np.trapz(x ** 2, axis=1, dx=Te).reshape(-1,1) # area under curve ** 2

    return np.concatenate((mean_ts,max_ts,min_ts,std_ts,skew_ts,kurtosis_ts,
                           iqr_ts,mad_ts,area_ts,sq_area_ts), axis=1)

def frequency_domain_features(x, Te=1.0):

    # As the DFT coefficients and their corresponding frequencies are symetrical arrays
    # with respect to the middle of the array we need to know if the number of readings 
    # in x is even or odd to then split the arrays...
    if x.shape[1]%2 == 0:
        N = int(x.shape[1]/2)
    else:
        N = int(x.shape[1]/2) - 1
    xf = np.repeat(fftfreq(x.shape[1],d=Te)[:N].reshape(1,-1), x.shape[0], axis=0) # frequencies
    dft = np.abs(fft(x, axis=1))[:,:N] # DFT coefficients   

    dft = pd.DataFrame(dft)
    # statistical and area features
    dft_features = stat_area_features(dft, Te=1.0)
    # weighted mean frequency
    dft_weighted_mean_f = np.average(xf, axis=1, weights=dft).reshape(-1,1)
    # 5 first DFT coefficients 
    dft_first_coef = dft.iloc[:,:5]    
    # 5 first local maxima of DFT coefficients and their corresponding frequencies
    dft_max_coef = np.zeros((x.shape[0],5))
    dft_max_coef_f = np.zeros((x.shape[0],5))
    for row in range(x.shape[0]):
        # finds all local maximas indexes
        z = dft.iloc[row,:]
        extrema_ind = argrelextrema(z.values, np.greater, axis=0) 
        # makes a list of tuples (DFT_i, f_i) of all the local maxima
        # and keeps the 5 biggest...
        s_row = [(dft.iloc[row,:][j],xf[row,j]) for j in extrema_ind[0]]
        extrema_row = sorted(s_row,
                             key=operator.itemgetter(0), reverse=True)[:5]

        for i, ext in enumerate(extrema_row):
            dft_max_coef[row,i] = ext[0]
            dft_max_coef_f[row,i] = ext[1]    
    # print(dft_features)
    # print(dft_weighted_mean_f)
    # print(dft_first_coef)
    # print(dft_max_coef)
    # print(dft_max_coef_f)
    # exit()
    return np.concatenate((dft_features,dft_weighted_mean_f,dft_first_coef,
                           dft_max_coef,dft_max_coef_f), axis=1)

def make_feature_vector(x, Te=1.0):

    # Raw signals :  stat and area features
    features_xt = stat_area_features(x, Te=Te)

    # x = x.to_numpy/
    # Jerk signals :  stat and area features
    jerk_x = (x.iloc[:,1:]-x.iloc[:,:-1])/Te
    features_xt_jerk = stat_area_features(jerk_x, Te=Te)

    # Raw signals : frequency domain features 
    features_xf = frequency_domain_features(x, Te=1/Te)
    
    # Jerk signals : frequency domain features 
    jerk_xf = (x.iloc[:,1:]-x.iloc[:,:-1])/Te
    features_xf_jerk = frequency_domain_features(jerk_xf, Te=1/Te)
    
    
    return np.concatenate((features_xt,
                           features_xt_jerk, 
                           features_xf, 
                           features_xf_jerk), axis=1)