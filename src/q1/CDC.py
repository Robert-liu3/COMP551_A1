import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder

def fetch_data_CDC():

    ### CDC Diabetes Health Indicators ###
    # fetch dataset 
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    # data (as pandas dataframes) 
    X_CDC = cdc_diabetes_health_indicators.data.features 
    Y_CDC = cdc_diabetes_health_indicators.data.targets
    # metadata  
    print(cdc_diabetes_health_indicators.metadata) 
    # variable information 
    print(cdc_diabetes_health_indicators.variables) 

    X_CDC.to_csv('./q1/csv/X_cdc.csv', index=False)
    Y_CDC.to_csv('./q1/csv/Y_cdc.csv', index=False)

    return X_CDC, Y_CDC