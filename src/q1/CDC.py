import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo 
from sklearn.preprocessing import LabelEncoder
def fetch_data_CDC():
    # Example data loading function; replace with actual data fetching
    cdc_diabetes = fetch_ucirepo(id=891)  
    X_diabetes = cdc_diabetes.data.features
    y_diabetes = cdc_diabetes.data.targets

    # Empty values are filled with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    X_diabetes_imputed = pd.DataFrame(imputer.fit_transform(X_diabetes), columns=X_diabetes.columns)

    # Feature scaling so that there is no bias
    scaler = StandardScaler()
    X_diabetes_scaled = pd.DataFrame(scaler.fit_transform(X_diabetes_imputed), columns=X_diabetes.columns)

    # Save the scaled data
    X_diabetes_scaled.to_csv('./q1/csv/X_cdc.csv', index=False)
    y_diabetes.to_csv('./q1/csv/Y_cdc.csv', index=False)

    return X_diabetes_scaled.to_numpy(), y_diabetes['Diabetes_binary'].to_numpy()

    


