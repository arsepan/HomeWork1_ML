from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
import pandas as pd
from sklearn.linear_model import Ridge
import joblib
import re
import numpy as np

app = FastAPI()

class Item(BaseModel):
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

model = joblib.load('elasticnet_model.pkl')
scaler = joblib.load("scaler.pkl")

nan_columns = ['mileage', 'engine', 'max_power', 'torque', 'seats']
nan_medians = [19.3, 1248.0, 82.0, 171.0, 5.0]

def extract_number(value):
    if pd.isnull(value):
        return np.nan
    match = re.search(r'[\d.]+', str(value))
    if match:
        return float(match.group(0))
    return np.nan

def preprocess_torque(value):
    if pd.isnull(value):
        return np.nan
    match = re.search(r'([\d.]+)\s*(Nm|kgm)', value, re.IGNORECASE)
    if match:
        torque_value = float(match.group(1))
        if match.group(2).lower() == 'kgm':
            torque_value *= 9.80665
        return torque_value
    return np.nan

def preprocess_item(item: Item) -> pd.DataFrame:
    df = pd.DataFrame([item.dict()])

    df['mileage'] = df['mileage'].apply(extract_number)
    df['engine'] = df['engine'].apply(extract_number)
    df['max_power'] = df['max_power'].apply(extract_number)
    df['torque'] = df['torque'].apply(preprocess_torque)

    for i in range(len(nan_columns)):
        df[nan_columns[i]].fillna(nan_medians[i], inplace=True)

    df[['mileage', 'engine', 'max_power']] = df[['mileage', 'engine', 'max_power']].astype(float)
    df[['engine', 'seats']] = df[['engine', 'seats']].astype(int)

    df_scaled = scaler.transform(df.select_dtypes(include=['int', 'float']))
    return df_scaled

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    processed_data = preprocess_item(item)
    prediction = model.predict(processed_data)
    print('Предсказал студент Панов А. С.')
    return float(prediction[0])

@app.post("/predict_items")
def predict_items(file: UploadFile = File(...)) -> str:
    df = pd.read_csv(file.file)
    processed_data = pd.concat([preprocess_item(Item(**row)) for _, row in df.iterrows()], axis=0)
    predictions = model.predict(processed_data)
    print('Предсказал студент Панов А. С.')
    df['predicted_price'] = predictions
    df.to_csv('predicted_results.csv', index=False)
    return 'predicted_results.csv'
