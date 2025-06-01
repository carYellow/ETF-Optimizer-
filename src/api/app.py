from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from ..models.predict import StockPredictor

app = FastAPI(
    title="Stock Performance Prediction API",
    description="API for predicting whether a stock will outperform S&P 500",
    version="1.0.0"
)

# Initialize predictor
predictor = StockPredictor()

class PredictionRequest(BaseModel):
    symbol: str
    date: Optional[str] = None

class PredictionResponse(BaseModel):
    symbol: str
    date: str
    prediction: bool
    probability: float
    features: dict

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make a prediction for a given stock and date.
    
    Args:
        request (PredictionRequest): Request containing stock symbol and optional date
        
    Returns:
        PredictionResponse: Prediction results
    """
    try:
        # Validate date format if provided
        if request.date:
            try:
                datetime.strptime(request.date, "%Y-%m-%d")
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid date format. Use YYYY-MM-DD"
                )
        
        # Make prediction
        result = predictor.predict(request.symbol, request.date)
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    return {"status": "healthy"} 