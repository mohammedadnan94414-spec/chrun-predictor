
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

# Locate and load the trained model
MODEL_PATH = Path(__file__).resolve().parent / "models" / "telecom_churn_model.pkl"

model = joblib.load(MODEL_PATH)

# Temporary prediction threshold.
# We can replace this with your verified Step 6 threshold later.
THRESHOLD = 0.50

# Create FastAPI application
from fastapi.responses import HTMLResponse
from fastapi.openapi.docs import get_swagger_ui_html

app = FastAPI(
    title="Telecom Customer Churn Prediction",
    description="Predict whether a telecom customer may leave.",
    version="1.0",
    docs_url=None
)

@app.get("/docs", include_in_schema=False)
async def custom_docs():

    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="Telecom Customer Churn Prediction"
    )

    html = response.body.decode("utf-8")

    custom_css = """
    <style>
        .swagger-ui .version-stamp,
        .swagger-ui .version-pragma {
            display: none !important;
        }
    </style>
    """

    html = html.replace("</head>", custom_css + "</head>")

    return HTMLResponse(content=html)

# Customer input details
class CustomerData(BaseModel):
    AccountWeeks: int = Field(ge=0)
    ContractRenewal: int = Field(ge=0, le=1)
    DataPlan: int = Field(ge=0, le=1)
    DataUsage: float = Field(ge=0)
    CustServCalls: int = Field(ge=0)
    DayMins: float = Field(ge=0)
    DayCalls: int = Field(ge=0)
    MonthlyCharge: float = Field(ge=0)
    OverageFee: float = Field(ge=0)
    RoamMins: float = Field(ge=0)


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Telecom Churn Prediction API is running!"
    }


# Prediction endpoint
@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert customer details into a dictionary
    data = customer.model_dump()

    # Calculate engineered features
    data["AvgDayMinsPerCall"] = (
        data["DayMins"] / data["DayCalls"]
        if data["DayCalls"] > 0 else 0
    )

    data["HighServiceCalls"] = (
        1 if data["CustServCalls"] >= 3 else 0
    )

    # Keep the exact feature order used during training
    feature_order = list(model.feature_names_in_)

    input_df = pd.DataFrame(
        [[data[feature] for feature in feature_order]],
        columns=feature_order
    )

    # Predict churn probability
    classes = list(model.classes_)
    churn_index = classes.index(1)

    probability = model.predict_proba(input_df)[0][churn_index]

    # Apply prediction threshold
    prediction = 1 if probability >= THRESHOLD else 0

    return {
        "prediction": prediction,
        "result": "Customer Likely to Churn" if prediction == 1
                  else "Customer Likely to Stay",
        "churn_probability": round(float(probability) * 100, 2),
        "threshold": THRESHOLD
    }
