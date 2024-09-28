import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import StandardScaler
  
def fetch_data_ITT():

    ### Infrared Thermography Temperature ###
    # fetch dataset 
    infrared_thermography_temperature = fetch_ucirepo(id=925) 
    # data (as pandas dataframes) 
    X_itt = infrared_thermography_temperature.data.features 
    Y_itt = infrared_thermography_temperature.data.targets 

    X_itt['Gender'] = LabelEncoder().fit_transform(X_itt['Gender'])           
    X_itt['Age'] = LabelEncoder().fit_transform(X_itt['Age'])
    X_itt['Ethnicity'] = LabelEncoder().fit_transform(X_itt['Ethnicity'])

    X_itt = X_itt.fillna(X_itt.mean())

    X_itt.to_csv('./q1/csv/X_itt.csv', index=False)
    Y_itt.to_csv('./q1/csv/Y_itt.csv', index=False)
    
    return X_itt, Y_itt

def check_data(x, y):
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        print("Data contains NaN values.")
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        print("Data contains Inf values.")
    if x.ndim != 2 or y.ndim != 2:
        print("Data dimensions are not correct.")
    elif x.shape[1] != y.shape[0]:
        print("The second dimension of X does not match the size of Y.")