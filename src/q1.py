import numpy as np
from ucimlrepo import fetch_ucirepo 
  
def fetch_data():

    ### Infrared Thermography Temperature ###
    # fetch dataset 
    infrared_thermography_temperature = fetch_ucirepo(id=925) 
    # data (as pandas dataframes) 
    X_itt = infrared_thermography_temperature.data.features 
    Y_itt = infrared_thermography_temperature.data.targets 

    ### CDC Diabetes Health Indicators ###
    # fetch dataset 
    cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
    # data (as pandas dataframes) 
    X_CDC = cdc_diabetes_health_indicators.data.features 
    Y_CDC = cdc_diabetes_health_indicators.data.targets 

    # metadata 
    print(infrared_thermography_temperature.metadata) 
    print(cdc_diabetes_health_indicators.metadata) 

    # variable information 
    print(infrared_thermography_temperature.variables) 
    print(cdc_diabetes_health_indicators.variables) 
