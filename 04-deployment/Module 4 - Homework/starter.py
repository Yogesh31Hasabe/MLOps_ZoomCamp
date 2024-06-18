#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')




## After conversion from ipynb notebook, this script/ the Code in the script is re-structured for making efficient calls from CLI
import sys
import pickle
import pandas as pd
import numpy as np

def read_data(filename):
    categorical = ['PULocationID', 'DOLocationID']
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def run():
    if len(sys.argv) != 3:
        print("Usage: python script.py <year> <month>")
        sys.exit(1)
    
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    
    # Load the model and DictVectorizer
    with open('model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)
    
    # Read the data for the given year and month
    file_path = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    df = read_data(file_path)
    
    # Prepare the data for prediction
    categorical = ['PULocationID', 'DOLocationID']
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    
    # Output the standard deviation and mean of the predictions
    print(f"The standard deviation of the trips duration in {year}-{month:02d} is: {np.std(y_pred)}")
    print(f"The mean of the trips duration in {year}-{month:02d} is: {np.mean(y_pred)}")
    
    # Create the results dataframe
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    pred_series = pd.Series(y_pred, name='duration')
    df_results = pd.concat([df[['ride_id']], pred_series], axis=1)
    
    # Save the results to a parquet file
    df_results.to_parquet(
        'output.parquet',
        engine='pyarrow',
        compression=None,
        index=False
    )

if __name__ == '__main__':
    run()