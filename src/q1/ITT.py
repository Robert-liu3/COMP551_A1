import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
  
def fetch_data_ITT():

    ### Infrared Thermography Temperature ###
    # fetch dataset 
    infrared_thermography_temperature = fetch_ucirepo(id=925) 
    # data (as pandas dataframes) 
    X_itt = infrared_thermography_temperature.data.features 
    Y_itt = infrared_thermography_temperature.data.targets 
    # metadata 
    print(infrared_thermography_temperature.metadata) 
    # variable information 
    print(infrared_thermography_temperature.variables) 

    X_itt['Gender'] = LabelEncoder().fit_transform(X_itt['Gender'])           
    X_itt['Age'] = LabelEncoder().fit_transform(X_itt['Age'])
    X_itt['Ethnicity'] = LabelEncoder().fit_transform(X_itt['Ethnicity'])

    # X_itt.to_csv('./q1/csv/X_itt.csv', index=False)

    # print("this is the target ----------------------------- ")
    # print(Y_itt)

    # print("this is the distribution of the target ----------------------------- ")
    # print(Y_itt.value_counts())
    # print(Y_itt.describe())

    return X_itt, Y_itt
