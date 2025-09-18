# Life Expectancy Dashboard (https://life-expectancy-dashboard-cwc5grlviqbixquyqpeajf.streamlit.app/)

This is an interactive **Streamlit dashboard** for exploring global life expectancy data, performing exploratory data analysis (EDA), and building regression, classification, and clustering models.
Users can upload their own CSV datasets or use a default dataset to explore health, demographic, and development indicators.

---

## Features

### 1. **Dataset used**
- The dataset used in this project contains life expectancy and related health indicators for 206 countries over the period 2000–2019. It includes both male and female populations with a total of 9,928 records:

Female records: 4,964

Male records: 4,964

The dataset has been filtered using a country mapping list to ensure consistency in country names. It includes the following types of information:

Demographic: Country, Year, Gender

Life Expectancy: Life expectancy at birth

Health Indicators: Infant Mortality, DPT Immunization, HepB3 Immunization, Measles Immunization, Tuberculosis Incidence, Tuberculosis treatment, Mortality caused by road traffic injury

Socioeconomic Indicators: Unemployment, Per Capita income, Access to clean fuels and cooking technologies, Basic sanitation services, Hospital beds.

### 2. **Exploratory Data Analysis (EDA)**
- Correlation heatmaps of numeric features.
- Trend analysis of health indicators over time.
- Top/bottom countries by selected indicators.
- Pairplots for numeric features.
- Countries with declining life expectancy over the last 10 years.

### 3. **Interactive Visualizations**
- Select countries, gender, and indicators to visualize trends with **Plotly** line charts.
- Filter by year range with a slider.

### 4. **Regression Model**
- Predicts **Life Expectancy** using a trained Random Forest regressor.
- Users can adjust numeric features using sliders to see the predicted value.
- Compare prediction against country and global averages.
- Gauge visualization of predicted life expectancy.

### 5. **Classification Model**
- Predicts **Life Expectancy Category** (`Low`, `Medium`, `High`) using Random Forest classifier.
- Interactive sliders for numeric features to generate predictions.
- Displays probability of each category.
- Shows **classification report** and **confusion matrix**.

### 6. **Clustering**
- K-Means clustering of countries based on health and development indicators.
- Interactive **choropleth map** of clusters over time.
- Shows cluster centers as mean values for numeric features.

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
