import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
def fetch_data_CDC():
    cdc_diabetes = fetch_ucirepo(id=891)
    X_diabetes = cdc_diabetes.data.features
    y_diabetes = cdc_diabetes.data.targets

    # Empty values are filled with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    
    X_diabetes = pd.DataFrame(imputer.fit_transform(X_diabetes), columns=X_diabetes.columns)

    # Feature scaling so that there is no bias
    scaler = StandardScaler()
    X_diabetes_scaled = pd.DataFrame(scaler.fit_transform(X_diabetes), columns=X_diabetes.columns)

    return X_diabetes.to_numpy(), y_diabetes['Diabetes_binary'].to_numpy()

    


