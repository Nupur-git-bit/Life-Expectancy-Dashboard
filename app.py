import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Life Expectancy Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🌍 Life Expectancy Dashboard")
st.markdown("""
**Interactive Dashboard for Forecasting Demographic Trends and Resource Needs**

This dashboard provides comprehensive analysis and forecasting of life expectancy trends using socio-economic data and machine learning algorithms.
""")

# Generate synthetic life expectancy data for demonstration
@st.cache_data
def load_data():
    """Load and prepare synthetic life expectancy data"""
    np.random.seed(42)
    
    countries = ['United States', 'China', 'India', 'Germany', 'Japan', 'Brazil', 'Nigeria', 'Bangladesh', 'Russia', 'Mexico',
                'Iran', 'Turkey', 'Vietnam', 'Philippines', 'Ethiopia', 'Egypt', 'United Kingdom', 'France', 'Italy', 'South Africa']
    
    years = list(range(2000, 2024))
    data = []
    
    for country in countries:
        base_life_expectancy = np.random.uniform(65, 82)
        gdp_per_capita_base = np.random.uniform(1000, 50000)
        
        for year in years:
            # Simulate trends over time
            year_factor = (year - 2000) * 0.1
            life_expectancy = base_life_expectancy + year_factor + np.random.normal(0, 1)
            gdp_per_capita = gdp_per_capita_base * (1 + (year - 2000) * 0.02) + np.random.normal(0, 1000)
            
            data.append({
                'Country': country,
                'Year': year,
                'Life_Expectancy': max(60, life_expectancy),
                'GDP_per_Capita': max(500, gdp_per_capita),
                'Healthcare_Expenditure': np.random.uniform(200, 5000),
                'Education_Index': np.random.uniform(0.3, 0.9),
                'Population': np.random.uniform(1000000, 1400000000),
                'Infant_Mortality': max(1, np.random.uniform(2, 50)),
                'Adult_Mortality': max(50, np.random.uniform(100, 400)),
                'Alcohol_Consumption': np.random.uniform(0, 15),
                'BMI': np.random.uniform(18, 35),
                'Immunization_Coverage': np.random.uniform(60, 100)
            })
    
    return pd.DataFrame(data)

# Load data
df = load_data()

# Sidebar for filters and controls
st.sidebar.header("🔧 Dashboard Controls")

# Country selection
countries = st.sidebar.multiselect(
    "Select Countries",
    options=df['Country'].unique(),
    default=df['Country'].unique()[:5]
)

# Year range selection
year_range = st.sidebar.slider(
    "Select Year Range",
    min_value=int(df['Year'].min()),
    max_value=int(df['Year'].max()),
    value=(int(df['Year'].min()), int(df['Year'].max()))
)

# Filter data
filtered_df = df[
    (df['Country'].isin(countries)) & 
    (df['Year'] >= year_range[0]) & 
    (df['Year'] <= year_range[1])
]

# Main dashboard content
tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends Analysis", "🤖 ML Forecasting", "📊 Resource Planning", "📋 Data Explorer"])

with tab1:
    st.header("Life Expectancy Trends Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Life expectancy over time
        fig1 = px.line(
            filtered_df, 
            x='Year', 
            y='Life_Expectancy', 
            color='Country',
            title="Life Expectancy Trends Over Time"
        )
        fig1.update_layout(height=400)
        st.plotly_chart(fig1, use_container_width=True)
        
        # GDP vs Life Expectancy scatter
        latest_year = filtered_df['Year'].max()
        latest_data = filtered_df[filtered_df['Year'] == latest_year]
        
        fig3 = px.scatter(
            latest_data,
            x='GDP_per_Capita',
            y='Life_Expectancy',
            size='Population',
            hover_name='Country',
            title=f"GDP per Capita vs Life Expectancy ({latest_year})"
        )
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        # Average life expectancy by country
        avg_life_exp = filtered_df.groupby('Country')['Life_Expectancy'].mean().sort_values(ascending=True)
        
        fig2 = px.bar(
            x=avg_life_exp.values,
            y=avg_life_exp.index,
            orientation='h',
            title="Average Life Expectancy by Country"
        )
        fig2.update_layout(height=400)
        st.plotly_chart(fig2, use_container_width=True)
        
        # Healthcare expenditure vs Life expectancy
        fig4 = px.scatter(
            latest_data,
            x='Healthcare_Expenditure',
            y='Life_Expectancy',
            color='Country',
            title=f"Healthcare Expenditure vs Life Expectancy ({latest_year})"
        )
        fig4.update_layout(height=400)
        st.plotly_chart(fig4, use_container_width=True)

with tab2:
    st.header("Machine Learning Forecasting")
    
    # Model selection
    model_type = st.selectbox(
        "Select ML Model",
        ["Linear Regression", "Random Forest"]
    )
    
    # Feature selection for prediction
    features = ['GDP_per_Capita', 'Healthcare_Expenditure', 'Education_Index', 
                'Infant_Mortality', 'Adult_Mortality', 'Alcohol_Consumption', 
                'BMI', 'Immunization_Coverage']
    
    selected_features = st.multiselect(
        "Select Features for Prediction",
        options=features,
        default=['GDP_per_Capita', 'Healthcare_Expenditure', 'Education_Index', 'Infant_Mortality']
    )
    
    if len(selected_features) > 0:
        # Prepare data for ML
        ml_data = filtered_df.dropna()
        X = ml_data[selected_features]
        y = ml_data['Life_Expectancy']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        if model_type == "Linear Regression":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("R² Score", f"{r2_score(y_test, y_pred):.3f}")
        with col2:
            st.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        with col3:
            st.metric("Training Samples", len(X_train))
        
        # Prediction vs Actual plot
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=y_test, 
            y=y_pred,
            mode='markers',
            name='Predictions',
            text=[f"Actual: {a:.1f}<br>Predicted: {p:.1f}" for a, p in zip(y_test, y_pred)]
        ))
        fig_pred.add_trace(go.Scatter(
            x=[y_test.min(), y_test.max()],
            y=[y_test.min(), y_test.max()],
            mode='lines',
            name='Perfect Prediction',
            line=dict(dash='dash')
        ))
        fig_pred.update_layout(
            title="Predicted vs Actual Life Expectancy",
            xaxis_title="Actual Life Expectancy",
            yaxis_title="Predicted Life Expectancy"
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Feature importance (for Random Forest)
        if model_type == "Random Forest":
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=True)
            
            fig_importance = px.bar(
                feature_importance,
                x='importance',
                y='feature',
                orientation='h',
                title="Feature Importance in Random Forest Model"
            )
            st.plotly_chart(fig_importance, use_container_width=True)
        
        # Future predictions
        st.subheader("📮 Future Scenario Prediction")
        
        st.write("Adjust the sliders below to see predicted life expectancy:")
        
        prediction_inputs = {}
        cols = st.columns(len(selected_features))
        
        for i, feature in enumerate(selected_features):
            with cols[i % len(cols)]:
                min_val = float(ml_data[feature].min())
                max_val = float(ml_data[feature].max())
                mean_val = float(ml_data[feature].mean())
                
                prediction_inputs[feature] = st.slider(
                    feature.replace('_', ' '),
                    min_value=min_val,
                    max_value=max_val,
                    value=mean_val,
                    key=f"pred_{feature}"
                )
        
        # Make prediction for scenario
        scenario_df = pd.DataFrame([prediction_inputs])
        scenario_prediction = model.predict(scenario_df)[0]
        
        st.success(f"Predicted Life Expectancy: **{scenario_prediction:.1f} years**")

with tab3:
    st.header("Resource Planning Dashboard")
    
    # Resource needs estimation based on demographic trends
    st.subheader("📊 Healthcare Resource Planning")
    
    # Calculate resource needs
    resource_data = filtered_df.groupby(['Country', 'Year']).agg({
        'Life_Expectancy': 'mean',
        'Population': 'mean',
        'Healthcare_Expenditure': 'mean',
        'Infant_Mortality': 'mean'
    }).reset_index()
    
    # Estimate healthcare workforce needs (doctors per 1000 people)
    resource_data['Doctors_Needed'] = (resource_data['Population'] / 1000) * 2.5  # WHO recommendation
    resource_data['Hospital_Beds_Needed'] = (resource_data['Population'] / 1000) * 3.0  # WHO recommendation
    
    # Healthcare budget estimation
    resource_data['Healthcare_Budget'] = resource_data['Healthcare_Expenditure'] * resource_data['Population'] / 1000
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Healthcare workforce projection
        fig_workforce = px.line(
            resource_data,
            x='Year',
            y='Doctors_Needed',
            color='Country',
            title="Projected Healthcare Workforce Needs (Doctors)"
        )
        st.plotly_chart(fig_workforce, use_container_width=True)
        
        # Healthcare budget trends
        fig_budget = px.area(
            resource_data,
            x='Year',
            y='Healthcare_Budget',
            color='Country',
            title="Healthcare Budget Requirements Over Time"
        )
        st.plotly_chart(fig_budget, use_container_width=True)
    
    with col2:
        # Hospital bed requirements
        fig_beds = px.line(
            resource_data,
            x='Year',
            y='Hospital_Beds_Needed',
            color='Country',
            title="Hospital Bed Requirements"
        )
        st.plotly_chart(fig_beds, use_container_width=True)
        
        # Resource efficiency analysis
        resource_data['Resource_Efficiency'] = resource_data['Life_Expectancy'] / (resource_data['Healthcare_Expenditure'] / 1000)
        
        fig_efficiency = px.scatter(
            resource_data[resource_data['Year'] == resource_data['Year'].max()],
            x='Healthcare_Expenditure',
            y='Life_Expectancy',
            size='Population',
            color='Resource_Efficiency',
            hover_name='Country',
            title="Healthcare Resource Efficiency Analysis",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    # Resource summary table
    st.subheader("📋 Resource Summary for Latest Year")
    latest_resources = resource_data[resource_data['Year'] == resource_data['Year'].max()].copy()
    latest_resources = latest_resources.round(2)
    st.dataframe(
        latest_resources[['Country', 'Life_Expectancy', 'Doctors_Needed', 'Hospital_Beds_Needed', 'Healthcare_Budget']],
        use_container_width=True
    )

with tab4:
    st.header("Data Explorer")
    
    # Display filtered data
    st.subheader("📊 Filtered Dataset")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Data statistics
    st.subheader("📈 Dataset Statistics")
    st.write(filtered_df.describe())
    
    # Correlation matrix
    st.subheader("🔗 Correlation Matrix")
    numeric_cols = ['Life_Expectancy', 'GDP_per_Capita', 'Healthcare_Expenditure', 
                   'Education_Index', 'Infant_Mortality', 'Adult_Mortality', 
                   'Alcohol_Consumption', 'BMI', 'Immunization_Coverage']
    
    correlation_matrix = filtered_df[numeric_cols].corr()
    
    fig_corr = px.imshow(
        correlation_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Download data option
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"life_expectancy_data_{year_range[0]}_{year_range[1]}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
**About this Dashboard:**
This interactive dashboard demonstrates forecasting demographic trends and resource needs using machine learning algorithms. 
The data used is synthetic and generated for demonstration purposes. 

**Features:**
- Interactive visualizations for life expectancy trends
- Machine learning models for forecasting
- Resource planning and healthcare workforce estimation
- Comprehensive data exploration tools
""")