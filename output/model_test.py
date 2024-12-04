import numpy as np
import pandas as pd
from keras import models

def generate_mass_table(model, complete_data, name):
    """
    Generate a mass table file 
    """
    # Testing the model
    y = np.array(model.predict(complete_data[:,:-1])).flatten()
    
    #Generating data
    Z = np.array(complete_data[:,1])
    N = np.array(complete_data[:,0])
    AME20 = np.array(complete_data[:,-1])
    diff = AME20 - y

    # Saving the data
    data = {
            'Z': Z,
            'N': N,
            'AME20': AME20,
            'Prediction': y,
            'Difference': diff
            }
    
    df = pd.DataFrame(data)
    df = df.sort_values(by=['Z', 'N'])
    df.to_csv(f'output/mass_tables/{name}.dat', sep=';', index=False)

    return df
