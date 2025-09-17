# Life Expectancy Dashboard

This is an interactive **Streamlit dashboard** for exploring global life expectancy data, performing exploratory data analysis (EDA), and building regression, classification, and clustering models.
Users can upload their own CSV datasets or use a default dataset to explore health, demographic, and development indicators.

---

## Features

### 1. **Data Upload**
- Upload your own CSV file containing life expectancy and health indicators or use can default dataset available in this repo.
- Handles missing values, outliers, and basic data cleaning automatically.

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
