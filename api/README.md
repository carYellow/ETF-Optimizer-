# Stock Prediction API

This API serves predictions from trained stock performance models. It predicts whether a given stock will outperform the S&P 500 index.

## Features

- RESTful API built with FastAPI
- Accepts stock symbol and optional date for prediction
- Returns prediction with probability and certainty
- Handles feature alignment with trained models automatically
- Includes health check endpoint

## Endpoints

### Predict Stock Performance

```
POST /predict
```

Request body:
```json
{
  "symbol": "AAPL",
  "date": "2025-05-30"  // Optional, defaults to latest available date
}
```

Response:
```json
{
  "symbol": "AAPL",
  "date": "2025-05-30",
  "prediction": true,  // true = outperform, false = underperform
  "probability": 0.75,
  "certainty": 0.5,
  "features": {
    "feature1": 0.5,
    "feature2": -0.2,
    ...
  }
}
```

### Health Check

```
GET /health
```

Response:
```json
{
  "status": "healthy"
}
```

## Running the API

```bash
python api/run.py
```

Optional arguments:
- `--host`: Host to bind the server to (default: 0.0.0.0)
- `--port`: Port to bind the server to (default: 8000)
- `--reload`: Enable automatic reloading on code changes

## Testing the API

```bash
python api/test.py --symbol AAPL
```

Optional arguments:
- `--host`: API host (default: localhost)
- `--port`: API port (default: 8000)
- `--symbol`: Stock symbol to predict (default: AAPL)
- `--date`: Date for prediction in YYYY-MM-DD format (default: None)
