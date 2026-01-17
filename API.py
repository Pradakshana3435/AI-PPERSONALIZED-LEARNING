from fastapi import FastAPI
import pickle
import pandas as pd
app=FastAPI()
with open("learning_model.pkl","rb") as f:
    model = pickle.load(f)
@app.post("/predict")
def predict_learning_level(score: int,subject_encoded:int):
    data = pd.DataFrame([[score,subject_encoded]],columns=["Score","Subject_Encoded"])
    prediction = model.predict(data)[0]
    return {"Predicted_Learning_Level":prediction}
