# 🌍 Life Expectancy Dashboard

An interactive Streamlit dashboard for forecasting demographic trends and resource needs using large-scale socio-economic data, powered by machine learning algorithms with clear, data-driven visualizations.

## 🚀 Features

- **Interactive Visualizations**: Comprehensive charts and graphs showing life expectancy trends over time
- **Machine Learning Forecasting**: Advanced ML algorithms (Linear Regression, Random Forest) for predicting future demographic trends
- **Resource Planning**: Healthcare workforce and infrastructure planning based on demographic projections
- **Data Exploration**: Interactive data explorer with correlation analysis and downloadable datasets
- **Multi-country Analysis**: Compare trends across different countries and regions
- **Scenario Planning**: Adjust parameters to see predicted outcomes for different scenarios

## 📊 Dashboard Components

### 1. Trends Analysis
- Life expectancy trends over time by country
- GDP per capita vs life expectancy correlations
- Healthcare expenditure impact analysis
- Comparative analysis across nations

### 2. ML Forecasting
- **Linear Regression**: Simple, interpretable trend forecasting
- **Random Forest**: Advanced ensemble method for complex pattern recognition
- **Feature Importance Analysis**: Understanding which factors most impact life expectancy
- **Interactive Prediction**: Adjust socio-economic parameters to predict life expectancy

### 3. Resource Planning
- Healthcare workforce needs estimation (doctors per 1000 people)
- Hospital bed requirements calculation
- Healthcare budget projections
- Resource efficiency analysis

### 4. Data Explorer
- Complete dataset visualization
- Statistical summaries and correlations
- Data export functionality
- Interactive filtering and exploration

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nupur-git-bit/Life-Expectancy-Dashboard.git
   cd Life-Expectancy-Dashboard
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   streamlit run app.py
   ```

4. **Open your browser** and navigate to `http://localhost:8501`

## 📋 Requirements

- Python 3.8+
- Streamlit 1.28.0+
- Pandas 2.0.0+
- NumPy 1.24.0+
- Scikit-learn 1.3.0+
- Plotly 5.15.0+
- Seaborn 0.12.0+
- Matplotlib 3.7.0+

See `requirements.txt` for complete dependency list.

## 🧮 Data Features

The dashboard uses comprehensive socio-economic indicators including:

- **Life Expectancy**: Primary target variable for analysis and forecasting
- **GDP per Capita**: Economic development indicator
- **Healthcare Expenditure**: Investment in health infrastructure
- **Education Index**: Educational development metrics
- **Infant Mortality**: Early childhood health indicator
- **Adult Mortality**: Adult health and mortality rates
- **BMI**: Population health and nutrition status
- **Immunization Coverage**: Preventive healthcare coverage
- **Alcohol Consumption**: Lifestyle factor affecting health

## 🔬 Machine Learning Models

### Linear Regression
- **Purpose**: Baseline forecasting with interpretable coefficients
- **Strengths**: Simple, fast, easy to interpret
- **Use Case**: Understanding linear relationships between variables

### Random Forest
- **Purpose**: Advanced ensemble learning for complex patterns
- **Strengths**: Handles non-linear relationships, provides feature importance
- **Use Case**: More accurate predictions with complex socio-economic interactions

## 🎯 Use Cases

1. **Government Planning**: Healthcare policy and resource allocation
2. **Research**: Academic studies on demographic trends
3. **Healthcare Organizations**: Strategic planning for healthcare delivery
4. **International Development**: Understanding development impacts on health outcomes
5. **Educational**: Teaching data science and public health concepts

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📈 Future Enhancements

- [ ] Real-time data integration from WHO, World Bank APIs
- [ ] Additional ML models (XGBoost, Neural Networks)
- [ ] Geographic mapping with interactive world maps
- [ ] Time series forecasting with ARIMA/Prophet
- [ ] Multi-language support
- [ ] Advanced statistical testing
- [ ] Mobile-responsive design improvements

## 📞 Support

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact the development team
- Check the documentation

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- World Health Organization for health indicators methodology
- Streamlit team for the excellent dashboard framework
- Scikit-learn contributors for machine learning tools
- Plotly team for interactive visualization capabilities

---

**Built with ❤️ using Streamlit, Python, and Machine Learning**
