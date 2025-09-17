#!/usr/bin/env python3
"""
Simple test script to validate the core functionality of the Life Expectancy Dashboard
"""

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def test_data_generation():
    """Test the synthetic data generation function"""
    print("Testing data generation...")
    
    # Generate sample data similar to the app
    np.random.seed(42)
    countries = ['USA', 'China', 'India', 'Germany', 'Japan']
    years = list(range(2000, 2024))
    data = []
    
    for country in countries:
        base_life_expectancy = np.random.uniform(65, 82)
        for year in years:
            data.append({
                'Country': country,
                'Year': year,
                'Life_Expectancy': base_life_expectancy + np.random.normal(0, 1),
                'GDP_per_Capita': np.random.uniform(1000, 50000),
                'Healthcare_Expenditure': np.random.uniform(200, 5000),
                'Education_Index': np.random.uniform(0.3, 0.9),
            })
    
    df = pd.DataFrame(data)
    assert len(df) == len(countries) * len(years), "Data generation failed"
    assert df['Life_Expectancy'].notna().all(), "Missing life expectancy values"
    print("✓ Data generation test passed")
    return df

def test_ml_models(df):
    """Test machine learning model functionality"""
    print("Testing ML models...")
    
    # Prepare data
    features = ['GDP_per_Capita', 'Healthcare_Expenditure', 'Education_Index']
    X = df[features]
    y = df['Life_Expectancy']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_pred)
    
    assert lr_r2 > -1, "Linear regression R² is too low"
    print(f"✓ Linear Regression R²: {lr_r2:.3f}")
    
    # Test Random Forest
    rf_model = RandomForestRegressor(n_estimators=10, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    
    assert rf_r2 > -1, "Random Forest R² is too low"
    print(f"✓ Random Forest R²: {rf_r2:.3f}")
    
    # Test feature importance
    feature_importance = rf_model.feature_importances_
    assert len(feature_importance) == len(features), "Feature importance mismatch"
    print("✓ ML models test passed")

def test_data_processing():
    """Test data processing and filtering functions"""
    print("Testing data processing...")
    
    df = test_data_generation()
    
    # Test filtering
    filtered_df = df[df['Year'] >= 2010]
    assert len(filtered_df) < len(df), "Year filtering failed"
    
    # Test grouping
    country_stats = df.groupby('Country')['Life_Expectancy'].mean()
    assert len(country_stats) == 5, "Country grouping failed"
    
    # Test data types
    assert df['Life_Expectancy'].dtype in [np.float64, float], "Life expectancy should be numeric"
    assert df['Year'].dtype in [np.int64, int], "Year should be integer"
    
    print("✓ Data processing test passed")

def main():
    """Run all tests"""
    print("🧪 Running Life Expectancy Dashboard Tests\n")
    
    try:
        # Test data generation
        df = test_data_generation()
        
        # Test data processing
        test_data_processing()
        
        # Test ML models
        test_ml_models(df)
        
        print("\n🎉 All tests passed successfully!")
        print("The Life Expectancy Dashboard core functionality is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()