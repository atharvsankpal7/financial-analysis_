"""
Example usage scripts for the Financial Analysis Prediction API

This file demonstrates various ways to interact with the API.
Run the API server first: python main.py
"""

import requests
import json
from typing import List, Dict
import pandas as pd


# API Base URL
BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60 + "\n")


def example_1_health_check():
    """Example 1: Check API health"""
    print_section("Example 1: Health Check")
    
    response = requests.get(f"{BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print("✅ API is healthy!")
        print(f"   Status: {data['status']}")
        print(f"   Model Loaded: {data['model_loaded']}")
        print(f"   Version: {data['version']}")
    else:
        print(f"❌ Health check failed: {response.status_code}")


def example_2_model_info():
    """Example 2: Get model information"""
    print_section("Example 2: Model Information")
    
    response = requests.get(f"{BASE_URL}/model/info")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Model Type: {data['model_type']}")
        print(f"Architecture: {data['architecture']}")
        print(f"Sequence Length: {data['sequence_length']} days")
        print(f"Total Parameters: {data['total_parameters']:,}")
        print(f"Supported Assets: {len(data['supported_assets'])} assets")
    else:
        print(f"❌ Failed to get model info: {response.status_code}")


def example_3_single_prediction():
    """Example 3: Single asset prediction"""
    print_section("Example 3: Single Prediction - RELIANCE")
    
    payload = {
        "asset": "RELIANCE.NS",
        "days_ahead": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Asset: {data['asset']}")
        print(f"Current ROI: {data['current_roi']:.2f}%")
        print(f"Predicted ROI: {data['predicted_roi']:.2f}%")
        print(f"Expected Change: {data['predicted_change_percent']:.2f}%")
        print(f"Confidence: {data['confidence']}")
        
        # Interpret prediction
        if data['predicted_change_percent'] > 0:
            print(f"\n💹 Prediction: BULLISH (Expected gain: {data['predicted_change_percent']:.2f}%)")
        else:
            print(f"\n📉 Prediction: BEARISH (Expected loss: {data['predicted_change_percent']:.2f}%)")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(response.json())


def example_4_batch_prediction():
    """Example 4: Batch prediction for multiple assets"""
    print_section("Example 4: Batch Prediction")
    
    assets = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "GOLD"]
    
    payload = {
        "assets": assets,
        "days_ahead": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        print(f"Total Predictions: {data['total_assets']}\n")
        
        # Display predictions in table format
        predictions = data['predictions']
        
        print(f"{'Asset':<15} {'Current ROI':<12} {'Predicted ROI':<14} {'Change %':<10} {'Confidence':<10}")
        print("-" * 70)
        
        for pred in predictions:
            print(f"{pred['asset']:<15} "
                  f"{pred['current_roi']:>10.2f}% "
                  f"{pred['predicted_roi']:>12.2f}% "
                  f"{pred['predicted_change_percent']:>8.2f}% "
                  f"{pred['confidence']:<10}")
    else:
        print(f"❌ Batch prediction failed: {response.status_code}")
        print(response.json())


def example_5_top_performers():
    """Example 5: Find top predicted performers"""
    print_section("Example 5: Top Predicted Performers")
    
    # Get all supported stocks
    stocks = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", 
        "ICICIBANK.NS", "LT.NS", "SBIN.NS", "BHARTIARTL.NS"
    ]
    
    payload = {
        "assets": stocks,
        "days_ahead": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        predictions = data['predictions']
        
        # Sort by predicted change
        sorted_preds = sorted(
            predictions, 
            key=lambda x: x['predicted_change_percent'], 
            reverse=True
        )
        
        print("🚀 TOP 3 PREDICTED GAINERS:")
        print("-" * 50)
        for i, pred in enumerate(sorted_preds[:3], 1):
            print(f"{i}. {pred['asset']:<15} +{pred['predicted_change_percent']:.2f}%")
        
        print("\n📉 TOP 3 PREDICTED LOSERS:")
        print("-" * 50)
        for i, pred in enumerate(sorted_preds[-3:], 1):
            print(f"{i}. {pred['asset']:<15} {pred['predicted_change_percent']:.2f}%")
    else:
        print(f"❌ Failed: {response.status_code}")


def example_6_commodity_prediction():
    """Example 6: Commodity predictions"""
    print_section("Example 6: Commodity Predictions")
    
    commodities = ["GOLD", "SILVER", "PLATINUM"]
    
    for commodity in commodities:
        payload = {
            "asset": commodity,
            "days_ahead": 1
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            trend = "📈 UP" if data['predicted_change_percent'] > 0 else "📉 DOWN"
            print(f"{commodity:<10} {trend}  Change: {data['predicted_change_percent']:>6.2f}%")
        else:
            print(f"{commodity:<10} ❌ Failed")


def example_7_error_handling():
    """Example 7: Error handling"""
    print_section("Example 7: Error Handling")
    
    # Test 1: Invalid asset
    print("Test 1: Invalid Asset Symbol")
    payload = {"asset": "INVALID.NS", "days_ahead": 1}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")
    
    # Test 2: Invalid days_ahead
    print("Test 2: Invalid Days Ahead (>30)")
    payload = {"asset": "RELIANCE.NS", "days_ahead": 100}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def example_8_export_to_csv():
    """Example 8: Export predictions to CSV"""
    print_section("Example 8: Export Predictions to CSV")
    
    assets = [
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "SBIN.NS", "GOLD", "SILVER"
    ]
    
    payload = {
        "assets": assets,
        "days_ahead": 1
    }
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    
    if response.status_code == 200:
        data = response.json()
        predictions = data['predictions']
        
        # Convert to DataFrame
        df = pd.DataFrame(predictions)
        
        # Select relevant columns
        df = df[['asset', 'current_roi', 'predicted_roi', 
                'predicted_change_percent', 'confidence']]
        
        # Save to CSV
        filename = "predictions_export.csv"
        df.to_csv(filename, index=False)
        
        print(f"✅ Predictions exported to {filename}")
        print(f"\nPreview:")
        print(df.to_string(index=False))
    else:
        print(f"❌ Export failed: {response.status_code}")


def example_9_supported_assets():
    """Example 9: Get list of supported assets"""
    print_section("Example 9: Supported Assets")
    
    response = requests.get(f"{BASE_URL}/assets/supported")
    
    if response.status_code == 200:
        data = response.json()
        
        print(f"Total Supported Assets: {data['total']}\n")
        
        print(f"Stocks ({len(data['stocks'])}):")
        for i, stock in enumerate(data['stocks'], 1):
            print(f"  {i:2d}. {stock}")
        
        print(f"\nCommodities ({len(data['commodities'])}):")
        for i, commodity in enumerate(data['commodities'], 1):
            print(f"  {i}. {commodity}")
    else:
        print(f"❌ Failed: {response.status_code}")


def example_10_compare_sectors():
    """Example 10: Compare different sectors"""
    print_section("Example 10: Sector Comparison")
    
    sectors = {
        "Banking": ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS", "KOTAKBANK.NS"],
        "IT": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
        "Commodities": ["GOLD", "SILVER"]
    }
    
    for sector_name, assets in sectors.items():
        payload = {
            "assets": assets,
            "days_ahead": 1
        }
        
        response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            predictions = data['predictions']
            
            avg_change = sum(p['predicted_change_percent'] for p in predictions) / len(predictions)
            
            trend = "🟢" if avg_change > 0 else "🔴"
            print(f"{trend} {sector_name:<15} Avg Change: {avg_change:>6.2f}%")
        else:
            print(f"❌ {sector_name}: Failed")


def run_all_examples():
    """Run all examples"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║  Financial Analysis API - Example Usage Scripts         ║")
    print("╚" + "="*58 + "╝")
    
    try:
        example_1_health_check()
        example_2_model_info()
        example_3_single_prediction()
        example_4_batch_prediction()
        example_5_top_performers()
        example_6_commodity_prediction()
        example_7_error_handling()
        example_8_export_to_csv()
        example_9_supported_assets()
        example_10_compare_sectors()
        
        print("\n" + "="*60)
        print("  ✅ All examples completed successfully!")
        print("="*60 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server.")
        print("Make sure the API is running: python main.py")
        print("Then try running this script again.\n")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}\n")


if __name__ == "__main__":
    # Check if pandas is installed for CSV export
    try:
        import pandas as pd
    except ImportError:
        print("⚠️ Warning: pandas not installed. CSV export will be skipped.")
        print("Install with: pip install pandas\n")
    
    run_all_examples()
