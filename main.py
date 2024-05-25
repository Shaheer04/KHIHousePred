import pickle
import pandas as pd
import numpy as np


try:
    with open('finalized_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading pickle file: {e}")


features = pd.read_csv('features.csv')

def prediction(location,sqft,bedrooms,baths):
    loc_index = np.where(features.columns==location)[0][0]

    x = np.zeros(len(features.columns))
    x[0] = baths
    x[1] = sqft
    x[2] = bedrooms
    if loc_index >= 0:
        x[loc_index] = 1
    return model.predict([x])[0] / 100000

print(str(int(prediction('Nazimabad', 1800, 4, 3))) + " Lakhs")



