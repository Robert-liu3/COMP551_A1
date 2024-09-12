import numpy as np
from ucimlrepo import fetch_ucirepo 
  
def fetch_data():
    # fetch dataset 
    infrared_thermography_temperature = fetch_ucirepo(id=925) 

    # data (as pandas dataframes) 
    X = infrared_thermography_temperature.data.features 
    y = infrared_thermography_temperature.data.targets 

    # metadata 
    print(infrared_thermography_temperature.metadata) 

    # variable information 
    print(infrared_thermography_temperature.variables) 
