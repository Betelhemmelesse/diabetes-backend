from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
import traceback # Used to print the error details

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the models
# NOTE: If these fail to load here, the app won't start.
try:
    lr_pipeline = joblib.load('lr_pipeline.joblib')
    dt_pipeline = joblib.load('dt_pipeline.joblib')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Critical Error loading models: {e}")

class PredictionInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
async def predict(data: PredictionInput):
    try:
        # 1. Convert input to DataFrame
        # note: using model_dump() is preferred in Pydantic v2, but dict() works too
        input_data = data.dict() 
        input_df = pd.DataFrame([input_data])
        
        print("Received Data shape:", input_df.shape)

        # 2. Make Predictions
        lr_pred = lr_pipeline.predict(input_df)[0]
        dt_pred = dt_pipeline.predict(input_df)[0]
        
        # 3. Return Success
        return {
            "logistic_regression": int(lr_pred),
            "decision_tree": int(dt_pred),
            "status": "success"
        }

    except Exception as e:
        # This prints the REAL error to your terminal window
        print("------------------------------------------------")
        print("❌ PREDICTION FAILED")
        print(traceback.format_exc()) 
        print("------------------------------------------------")
        # Return a 500 error to the browser, but now we know why
        raise HTTPException(status_code=500, detail=str(e))