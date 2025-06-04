#!/usr/bin/env python3
"""
API Test Script

This script tests the stock prediction API with a sample request.
"""

import argparse
import requests
import json

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the stock prediction API')
    
    parser.add_argument('--host', type=str, default='localhost',
                       help='API host')
    parser.add_argument('--port', type=int, default=8000,
                       help='API port')
    parser.add_argument('--symbol', type=str, default='AAPL',
                       help='Stock symbol to predict')
    parser.add_argument('--date', type=str, default=None,
                       help='Date for prediction (YYYY-MM-DD)')
    
    return parser.parse_args()

def test_api(host: str, port: int, symbol: str, date: str = None):
    """Test the API with a prediction request."""
    url = f"http://{host}:{port}/predict"
    
    # Prepare request
    request_data = {
        "symbol": symbol
    }
    
    if date:
        request_data["date"] = date
    
    print(f"\nSending request to {url}")
    print(f"Request data: {json.dumps(request_data, indent=2)}")
    
    try:
        # Make prediction request
        response = requests.post(url, json=request_data)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            print("\nPrediction result:")
            print(f"Symbol: {result.get('symbol', 'N/A')}")
            print(f"Date: {result.get('date', 'N/A')}")
            print(f"Prediction: {'Outperform' if result.get('prediction') else 'Underperform'}")
            print(f"Probability: {result.get('probability', 'N/A')}")
            print(f"Certainty: {result.get('certainty', 'N/A')}")
            
            # Print a few sample features
            features = result.get('features', {})
            if features:
                print("\nSample features:")
                sample_features = list(features.items())[:5]  # First 5 features
                for feature, value in sample_features:
                    print(f"  {feature}: {value}")
                print(f"\nTotal features: {len(features)}")
            else:
                print("No features returned.")
            
            return True
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return False
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

def main():
    """Run the API test."""
    args = parse_args()
    
    print("=== TESTING STOCK PREDICTION API ===")
    
    # Test health endpoint
    health_url = f"http://{args.host}:{args.port}/health"
    print(f"\nChecking API health at {health_url}")
    
    try:
        health_response = requests.get(health_url)
        if health_response.status_code == 200:
            print("API is healthy")
        else:
            print(f"API health check failed: {health_response.status_code}")
            print(health_response.text)
            return
    except Exception as e:
        print(f"Error connecting to API: {str(e)}")
        print("Make sure the API server is running")
        return
    
    # Test prediction endpoint
    test_api(args.host, args.port, args.symbol, args.date)

if __name__ == "__main__":
    main()
