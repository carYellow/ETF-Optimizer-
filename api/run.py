#!/usr/bin/env python3
"""
API Server Runner

This script starts the FastAPI server for stock predictions.
"""

import uvicorn
import os
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the stock prediction API server')
    
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind the server to')
    parser.add_argument('--port', type=int, default=8000,
                       help='Port to bind the server to')
    parser.add_argument('--reload', action='store_true',
                       help='Enable automatic reloading on code changes')
    
    return parser.parse_args()

def main():
    """Run the API server."""
    args = parse_args()
    
    print("=== STARTING STOCK PREDICTION API SERVER ===")
    print(f"Server will be available at http://{args.host}:{args.port}")
    print("API documentation will be available at http://localhost:8000/docs")
    
    # Use the correct import path for the app
    uvicorn.run(
        "api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )

if __name__ == "__main__":
    main()
