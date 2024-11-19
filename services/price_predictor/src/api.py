from fastapi import FastAPI, HTTPException, Query
import json
from pydantic import BaseModel
import joblib
from typing import Dict

from loguru import logger

from src.model_registry import get_model_name
from src.config import config, comet_config, CometConfig
from src.hopsworks_api import push_value_to_feature_group
from src.price_predictor import PricePredictor


app = FastAPI()

predictors: Dict[str, PricePredictor] = {}

# we create a predictor object that loads the model from the registry
# predictor = PricePredictor.from_model_registry(
#     product_id=request.product_id,
#     ohlc_window_sec=request.ohlc_window_sec,
#     forecast_steps=request.forecast_steps,
#     status=request.model_status,
# )

# class PredictionRequest(BaseModel):
#     product_id: str
    

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict")
def predict(
    product_id: str=Query(..., description="The product ID to predict" )
):

    logger.info(f"Received request for product id: {product_id}")

    try:
        if product_id not in config.api_supported_product_ids:
            raise HTTPException(status_code=400, detail="Product ID not supported")
        
        if product_id not in predictors:
            predictors[product_id] = PricePredictor.from_model_registry(
                product_id=product_id,
                # these parameters are read from config
                ohlc_window_sec=config.ohlc_window_sec,
                forecast_steps=config.forecast_steps,
                status=config.ml_model_status,
            )

        # Extract the predictor object this produt_id
        predictor = predictors[product_id]

        prediction = predictor.predict()
        return {"prediction": prediction.to_json()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        


# if __name__ == "__main__":

#     import uvicorn
#     uvicorn.run(app,host="0.0.0.0", port =8000, reload=True)