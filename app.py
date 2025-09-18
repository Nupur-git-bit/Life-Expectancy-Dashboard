# -*- coding: utf-8 -*-
# redeploy trigger



# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# # app.py
# import os
# import streamlit as st
# st.set_page_config(layout="wide", page_title="Life Expectancy Dashboard")
# 
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.express as px
# 
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.utils import check_random_state
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.cluster import KMeans  # <-- Add this line
# from sklearn.metrics import silhouette_score
# 
# import tensorflow as tf
# from tensorflow.keras import layers, models, callbacks
# 
# import warnings
# warnings.filterwarnings("ignore")
# 
# # -----------------------
# # Config: default column lists (based on your data)
# # -----------------------
# DEFAULT_NUMERIC = [
#     'Year', 'Unemployment', 'Infant Mortality',
#     'Clean fuels and cooking technologies', 'Per Capita',
#     'Mortality caused by road traffic injury', 'Tuberculosis Incidence',
#     'DPT Immunization', 'HepB3 Immunization', 'Measles Immunization',
#     'Hospital beds', 'Basic sanitation services', 'Tuberculosis treatment'
# ]
# DEFAULT_CATEGORICAL = ['Country', 'Gender']
# DEFAULT_TARGET = 'Life expectancy'
# 
# # -----------------------
# # Helpers
# # -----------------------
# @st.cache_data
# def load_dataframe(uploaded_file):
#     if uploaded_file is None:
#         return None
#     df = pd.read_csv(uploaded_file)
#     return df
# 
# def build_preprocessor(numeric_cols, categorical_cols):
#     # numeric pipeline: impute median -> scale
#     numeric_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="median")),
#         ("scaler", StandardScaler())
#     ])
# 
#     # categorical pipeline: impute most frequent -> onehot
#     # handle sklearn older/newer OneHotEncoder params
#     try:
#         ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
#     except TypeError:
#         ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
# 
#     categorical_pipe = Pipeline([
#         ("imputer", SimpleImputer(strategy="most_frequent")),
#         ("onehot", ohe)
#     ])
# 
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ("num", numeric_pipe, numeric_cols),
#             ("cat", categorical_pipe, categorical_cols)
#         ],
#         remainder="drop",
#         sparse_threshold=0
#     )
#     return preprocessor
# 
# def get_feature_names(preprocessor, numeric_cols, categorical_cols):
#     names = []
#     if numeric_cols:
#         names += list(numeric_cols)
#     if categorical_cols:
#         # extract OHE names robustly
#         try:
#             ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
#             cat_names = list(ohe.get_feature_names_out(categorical_cols))
#         except Exception:
#             # fallback: no cat transformer or different structure
#             cat_names = []
#         names += cat_names
#     return names
# 
# def train_nn_regressor(X_train, y_train, X_val=None, y_val=None, epochs=100, verbose=0):
#     tf.random.set_seed(42)
#     model = models.Sequential([
#         layers.Input(shape=(X_train.shape[1],)),
#         layers.Dense(64, activation='relu'),
#         layers.Dropout(0.25),
#         layers.Dense(32, activation='relu'),
#         layers.Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     es = callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#     if X_val is None or y_val is None:
#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
#                             validation_split=0.2, callbacks=[es], verbose=verbose)
#     else:
#         history = model.fit(X_train, y_train, epochs=epochs, batch_size=32,
#                             validation_data=(X_val, y_val), callbacks=[es], verbose=verbose)
#     return model, history
# 
# # -----------------------
# # Streamlit UI
# # -----------------------
# st.title("Life Expectancy: EDA + Regression + Interactive Dashboard")
# st.markdown(
#     "Upload your CSV (or use the default dataset) and explore EDA, preprocessing, regression models, and interactive plots."
# )
# 
# # File uploader
# uploaded_file = st.file_uploader("Upload life expectancy CSV", type=["csv"])
# 
# if uploaded_file is not None:
#     df = load_dataframe(uploaded_file)
#     st.success("Loaded uploaded CSV file successfully!")
# else:
#     st.info("No file uploaded. Using default dataset.")
# 
#     default_csv_path = "life_expectancy.csv"
#     df = pd.read_csv(default_csv_path)
# 
# if df is None or df.empty:
#     st.error("No data available. Please upload a CSV file or check the default dataset.")
#     st.stop()
# 
# # Round HepB3 Immunization to whole number
# if 'HepB3 Immunization' in df.columns:
#     df['HepB3 Immunization'] = df['HepB3 Immunization'].round(0)
#     st.write("HepB3 Immunization (rounded):", df['HepB3 Immunization'].head())
# 
# # Cap Basic Sanitation Services values at 100
# if 'Basic sanitation services' in df.columns:
#     df['Basic sanitation services'] = df['Basic sanitation services'].clip(upper=100)
#     st.write("Basic sanitation services (capped at 100):", df['Basic sanitation services'].head())
# 
# # Drop unnecessary columns
# cols_to_drop = [
#     'GDP',
#     'GNI',
#     'Urban population',
#     'Rural population',
#     'Non-communicable Mortality',
#     'Sucide Rate'
# ]
# cols_before = df.columns.tolist()
# df = df.drop(columns=cols_to_drop, errors='ignore')
# cols_after = df.columns.tolist()
# st.write(f"Columns before dropping: {cols_before}")
# st.write(f"Dropping columns: {cols_to_drop}")
# st.write(f"Columns after dropping: {cols_after}")
# 
# # Filter valid countries using mapping file
# try:
#     country_mapping = pd.read_csv('countries.csv')
#     valid_countries = country_mapping['Country'].unique()
#     df = df[df['Country'].isin(valid_countries)]
#     st.write(f"Filtered to valid countries. Remaining rows: {len(df)}")
# except Exception as e:
#     st.warning(f"Could not load country mapping file or filter countries: {e}")
# 
# # Make a copy for clustering
# df_kmeans = df.copy()
# 
# st.write("Dataset preview:")
# st.dataframe(df.head())
# 
# # Let user confirm target, numeric and categorical features (auto-detect)
# all_cols = df.columns.tolist()
# target_col = st.selectbox("Target column (continuous)", [c for c in all_cols if df[c].dtype in [np.float64, np.int64]] + [DEFAULT_TARGET], index=0)
# st.write("Selected target:", target_col)
# 
# # auto-suggest numeric/categorical sets
# detected_numeric = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target_col]
# detected_categorical = [c for c in df.columns if c not in detected_numeric and c != target_col]
# 
# st.write("Auto-detected numeric columns (you can edit):")
# numeric_cols = st.multiselect("Numeric features", options=df.columns.tolist(), default=[c for c in detected_numeric if c in DEFAULT_NUMERIC] or detected_numeric)
# st.write("Auto-detected categorical columns (you can edit):")
# categorical_cols = st.multiselect("Categorical features", options=df.columns.tolist(), default=[c for c in detected_categorical if c in DEFAULT_CATEGORICAL] or detected_categorical)
# 
# # Simple cleaning options
# st.sidebar.header("Preprocessing options")
# fill_na_method = st.sidebar.selectbox("Numeric missing value fill", ["median", "mean", "zero"], index=0)
# cap_outliers = st.sidebar.checkbox("Cap outliers (IQR method) on numeric features", value=True)
# 
# def cap_outliers_df(df_in, numeric_feats):
#     df2 = df_in.copy()
#     for col in numeric_feats:
#         if col in df2 and pd.api.types.is_numeric_dtype(df2[col]):
#             q1 = df2[col].quantile(0.25)
#             q3 = df2[col].quantile(0.75)
#             iqr = q3 - q1
#             if pd.isna(iqr) or iqr == 0:
#                 continue
#             lower = q1 - 1.5 * iqr
#             upper = q3 + 1.5 * iqr
#             df2[col] = np.where(df2[col] < lower, lower, np.where(df2[col] > upper, upper, df2[col]))
#     return df2
# 
# # Apply simple imputation and outlier capping to a working copy
# df_clean = df.copy()
# 
# # imputations for numeric
# for col in numeric_cols:
#     if fill_na_method == "median":
#         val = df_clean[col].median()
#     elif fill_na_method == "mean":
#         val = df_clean[col].mean()
#     else:
#         val = 0.0
#     df_clean[col] = df_clean[col].fillna(val)
# 
# # categorical impute with mode
# for col in categorical_cols:
#     df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0] if not df_clean[col].mode().empty else "Unknown")
# 
# if cap_outliers and numeric_cols:
#     df_clean = cap_outliers_df(df_clean, numeric_cols)
# 
# st.write("After simple imputation and outlier capping (preview):")
# st.dataframe(df_clean.head())
# 
# # -----------------------
# # EDA (8 plots)
# # -----------------------
# st.header("Exploratory Data Analysis (8 charts)")
# col1, col2 = st.columns(2)
# 
# with col1:
#     # 1) Correlation heatmap of core indicators (subset to numeric_cols intersection)
#     st.subheader("1. Correlation heatmap")
#     corr_cols = [c for c in numeric_cols if c in df_clean.columns]
#     if len(corr_cols) >= 2:
#         fig, ax = plt.subplots(figsize=(8, 6))
#         sns.heatmap(df_clean[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax)
#         st.pyplot(fig)
#     else:
#         st.write("Need at least 2 numeric columns for correlation heatmap.")
# 
#     # 2) Life Expectancy vs Infant Mortality regression plot
#     if 'Infant Mortality' in df_clean.columns and target_col in df_clean.columns:
#         st.subheader("2. Life expectancy vs Infant Mortality")
#         fig, ax = plt.subplots(figsize=(6, 4))
#         sns.regplot(data=df_clean, x='Infant Mortality', y=target_col, scatter_kws={'alpha':0.4}, line_kws={'color':'red'}, ax=ax)
#         ax.set_ylabel(target_col)
#         st.pyplot(fig)
#     else:
#         st.write("Infant Mortality or target missing for regression plot.")
# 
#     # 3) Trend of Infant Mortality over time
#     if 'Year' in df_clean.columns and 'Infant Mortality' in df_clean.columns:
#         st.subheader("3. Infant mortality trend over years")
#         trend = df_clean.groupby('Year')['Infant Mortality'].mean().dropna()
#         fig, ax = plt.subplots(figsize=(6, 4))
#         ax.plot(trend.index, trend.values, marker='o')
#         ax.set_xlabel("Year"); ax.set_ylabel("Avg Infant Mortality")
#         st.pyplot(fig)
#     else:
#         st.write("No Year/Infant Mortality for trend.")
# 
# with col2:
#     # 4) Health metrics trend
#     st.subheader("4. Health metrics trends")
#     sample_metrics = [c for c in ['Tuberculosis Incidence', 'Measles Immunization', 'DPT Immunization', 'HepB3 Immunization', 'Tuberculosis treatment'] if c in df_clean.columns]
#     if 'Year' in df_clean.columns and sample_metrics:
#         metrics_trend = df_clean.groupby('Year')[sample_metrics].mean()
#         fig, ax = plt.subplots(figsize=(6,4))
#         for col in metrics_trend.columns:
#             ax.plot(metrics_trend.index, metrics_trend[col], marker='o', label=col)
#         ax.legend(); ax.set_xlabel("Year"); ax.set_ylabel("Value")
#         st.pyplot(fig)
#     else:
#         st.write("Metrics or Year missing.")
# 
#     # 5) Top/Bot 15 countries by Infant Mortality (top showing lowest mortality)
#     if 'Country' in df_clean.columns and 'Infant Mortality' in df_clean.columns:
#         st.subheader("5. Top 15 countries: lowest Infant Mortality")
#         top15 = df_clean.groupby('Country')['Infant Mortality'].mean().sort_values().head(15)
#         fig, ax = plt.subplots(figsize=(6,4))
#         sns.barplot(x=top15.values, y=top15.index, palette='mako', ax=ax)
#         st.pyplot(fig)
#     else:
#         st.write("Country or Infant Mortality missing.")
# 
# # 6) Pairplot (sample to avoid extreme cost)
# st.subheader("6. Pairplot (sample of numeric features)")
# pairplot_features = [c for c in [target_col, 'Clean fuels and cooking technologies', 'Mortality caused by road traffic injury', 'DPT Immunization', 'Measles Immunization'] if c in df_clean.columns]
# if len(pairplot_features) >= 2:
#     sample_df = df_clean[pairplot_features].dropna().sample(n=min(500, len(df_clean)), random_state=42)
#     pp = sns.pairplot(sample_df, diag_kind='kde', corner=True)
#     st.pyplot(pp)
# else:
#     st.write("Not enough features for pairplot.")
# 
# # 7) Correlation of features with life expectancy (vertical heatmap)
# st.subheader("7. Correlation with Life Expectancy")
# corr_cols2 = [c for c in [target_col, 'Clean fuels and cooking technologies', 'Per Capita', 'Tuberculosis Incidence', 'DPT Immunization', 'HepB3 Immunization', 'Measles Immunization'] if c in df_clean.columns]
# if len(corr_cols2) > 1:
#     corr_life = df_clean[corr_cols2].corr()[[target_col]].sort_values(by=target_col, ascending=False)
#     fig, ax = plt.subplots(figsize=(4, 4))
#     sns.heatmap(corr_life, annot=True, cmap='coolwarm', cbar=False, ax=ax)
#     st.pyplot(fig)
# else:
#     st.write("Not enough columns for correlation-to-life.")
# 
# # 8) Countries with most declining life expectancy in last 10 years (if Year present)
# st.subheader("8. Countries with declining life expectancy (last 10 years)")
# if 'Year' in df_clean.columns and target_col in df_clean.columns:
#     max_year = int(df_clean['Year'].max())
#     recent_year = max_year - 10
#     trends = df_clean[df_clean['Year'] >= recent_year].groupby('Country')[target_col].agg(['first', 'last']).dropna()
#     if not trends.empty:
#         trends['change'] = trends['last'] - trends['first']
#         dec = trends.sort_values('change').head(15)
#         fig, ax = plt.subplots(figsize=(6,4))
#         sns.barplot(x=dec['change'], y=dec.index, palette='Reds_r', ax=ax)
#         ax.set_xlabel("Change in Life Expectancy (years)")
#         st.pyplot(fig)
#     else:
#         st.write("Not enough data for decline analysis.")
# else:
#     st.write("Year or target column missing.")
# 
# # -----------------------
# # Interactive Plot (Plotly) - stakeholders exploration
# # -----------------------
# st.header("Interactive Plot: Select countries, gender & indicator (Plotly)")
# 
# country_options = sorted(df_clean['Country'].dropna().unique().tolist()) if 'Country' in df_clean.columns else []
# sel_countries = st.multiselect("Choose countries to show (pick up to 10)", country_options, default=country_options[:6], max_selections=10)
# 
# # Add gender selection (multi-select to allow Male, Female, or both)
# gender_options = sorted(df_clean['Gender'].dropna().unique().tolist()) if 'Gender' in df_clean.columns else []
# sel_genders = st.multiselect("Choose gender(s) to show", gender_options, default=gender_options)
# 
# sel_indicator = st.selectbox("Choose indicator for y-axis", options=[c for c in df_clean.columns if c not in ['Country', 'Year', 'Gender', target_col]])
# 
# year_range = None
# if 'Year' in df_clean.columns:
#     min_year, max_year = int(df_clean['Year'].min()), int(df_clean['Year'].max())
#     year_range = st.slider("Year range", min_year, max_year, (min_year, max_year))
# 
# if sel_countries and sel_indicator and sel_genders:
#     plot_df = df_clean[
#         (df_clean['Country'].isin(sel_countries)) &
#         (df_clean['Gender'].isin(sel_genders))
#     ].copy()
#     if 'Year' in df_clean.columns and year_range:
#         plot_df = plot_df[(plot_df['Year'] >= year_range[0]) & (plot_df['Year'] <= year_range[1])]
#     if plot_df.empty:
#         st.write("No data for selected filters.")
#     else:
#         fig = px.line(
#             plot_df.sort_values(['Country', 'Year']),
#             x='Year' if 'Year' in plot_df.columns else None,
#             y=sel_indicator,
#             color='Country',
#             line_dash='Gender',  # differentiate gender by line style
#             markers=True,
#             title=f"{sel_indicator} over time for selected countries and genders"
#         )
#         st.plotly_chart(fig, use_container_width=True)
# 
# 
# # Regression: Life Expectancy Predictor (Streamlit interactive sliders)
# 
# st.header("Regression ML Model for Predicting Life Expectancy")
# 
# if 'predictor_model' not in st.session_state or 'predictor_preprocessor' not in st.session_state:
#     with st.spinner("Training Random Forest predictor model..."):
#         from sklearn.ensemble import RandomForestRegressor
#         from sklearn.preprocessing import StandardScaler
#         from sklearn.compose import ColumnTransformer
#         from sklearn.impute import SimpleImputer
#         from sklearn.pipeline import Pipeline
#         import numpy as np
# 
#         numeric_features = [col for col in df.select_dtypes(include=['int64', 'float64']).columns
#                            if col not in ['Year', 'Life expectancy']]
# 
#         X = df[numeric_features].copy()
#         y = df['Life expectancy'].copy()
# 
#         X = X.fillna(X.median())
#         y = y.fillna(y.median())
# 
#         numeric_transformer = Pipeline(steps=[
#             ('imputer', SimpleImputer(strategy='median')),
#             ('scaler', StandardScaler())
#         ])
# 
#         predictor_preprocessor = ColumnTransformer(
#             transformers=[
#                 ('num', numeric_transformer, numeric_features)
#             ],
#             remainder='drop'
#         )
# 
#         X_processed = predictor_preprocessor.fit_transform(X)
# 
#         predictor_model = RandomForestRegressor(
#             n_estimators=100,
#             random_state=42,
#             n_jobs=-1,
#             max_depth=15
#         )
# 
#         predictor_model.fit(X_processed, y)
# 
#         st.session_state['predictor_model'] = predictor_model
#         st.session_state['predictor_preprocessor'] = predictor_preprocessor
#         st.session_state['numeric_features'] = numeric_features
# 
#         # Calculate country averages for baseline comparison
#         country_averages = {}
#         for country in df['Country'].unique():
#             country_data = df[df['Country'] == country]
#             if len(country_data) > 0:
#                 country_avg = country_data[numeric_features].mean().to_dict()
#                 country_averages[country] = country_avg
#         st.session_state['country_averages'] = country_averages
# 
# else:
#     predictor_model = st.session_state['predictor_model']
#     predictor_preprocessor = st.session_state['predictor_preprocessor']
#     numeric_features = st.session_state['numeric_features']
#     country_averages = st.session_state['country_averages']
# 
# # Sidebar or main panel widgets for country, gender, and numeric features
# 
# country = st.selectbox("Select Country", sorted(df['Country'].dropna().unique()))
# gender = st.selectbox("Select Gender", sorted(df['Gender'].dropna().unique()))
# 
# st.markdown("### Adjust health indicator values:")
# 
# input_data = {}
# for feature in numeric_features:
#     min_val = float(df[feature].min())
#     max_val = float(df[feature].max())
#     mean_val = float(df[feature].mean())
#     input_data[feature] = st.slider(f"{feature}", min_val, max_val, mean_val, step=(max_val - min_val) / 100)
# 
# # Prepare input DataFrame for prediction
# input_df = pd.DataFrame([input_data])
# input_df['Country'] = country
# input_df['Gender'] = gender
# 
# # The model uses only numeric features, so ignore country/gender in preprocess (or you can extend preprocessing)
# 
# # Preprocess input features (only numeric)
# X_input_processed = predictor_preprocessor.transform(input_df[numeric_features])
# 
# # Predict life expectancy
# prediction = predictor_model.predict(X_input_processed)[0]
# 
# # Compute baseline averages for comparison
# country_data = df[(df['Country'] == country) & (df['Gender'] == gender)]
# if not country_data.empty:
#     latest_year = int(country_data['Year'].max())
#     country_avg_life_exp = country_data[country_data['Year'] == latest_year]['Life expectancy'].mean()
# else:
#     latest_year = int(df['Year'].max())
#     country_avg_life_exp = df[df['Year'] == latest_year]['Life expectancy'].mean()
# 
# global_gender_avg = df[(df['Year'] == latest_year) & (df['Gender'] == gender)]['Life expectancy'].mean()
# 
# # Display results
# st.markdown("### Prediction Results")
# st.write(f"Predicted Life Expectancy: **{prediction:.2f} years**")
# st.write(f"{country} ({gender}) average life expectancy in {latest_year}: **{country_avg_life_exp:.2f} years**")
# st.write(f"Global {gender} average life expectancy in {latest_year}: **{global_gender_avg:.2f} years**")
# st.write(f"Difference from {country} average: **{prediction - country_avg_life_exp:+.2f} years**")
# st.write(f"Difference from global average: **{prediction - global_gender_avg:+.2f} years**")
# 
# # Optional: Plot a gauge chart using Plotly
# import plotly.graph_objects as go
# 
# fig = go.Figure(go.Indicator(
#     mode="gauge+number+delta",
#     value=prediction,
#     delta={'reference': country_avg_life_exp, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
#     gauge={
#         'axis': {'range': [0, 90]},
#         'bar': {'color': "darkblue"},
#         'steps': [
#             {'range': [0, 60], 'color': "lightcoral"},
#             {'range': [60, 75], 'color': "lightyellow"},
#             {'range': [75, 90], 'color': "lightgreen"}
#         ],
#         'threshold': {
#             'line': {'color': "red", 'width': 4},
#             'thickness': 0.75,
#             'value': country_avg_life_exp
#         }
#     },
#     title={'text': f"Life Expectancy Predictor\n{country} - {gender}"}
# ))
# 
# fig.update_layout(height=300)
# st.plotly_chart(fig, use_container_width=True)
# 
# st.header("Classification ML Model: Predict Life Expectancy Category with Overfitting Control")
# 
# # Prepare classification dataset
# df_clf = df.copy()
# df_clf['LifeExp_Category'] = pd.cut(
#     df_clf['Life expectancy'],
#     bins=[0, 60, 75, 100],
#     labels=['Low', 'Medium', 'High'],
#     right=True
# )
# 
# df_clf = df_clf.dropna(subset=['LifeExp_Category'])
# 
# # Features and target
# X = df_clf.select_dtypes(include=[np.number])
# y = df_clf['LifeExp_Category']
# 
# # Numeric features only
# numeric_cols = X.columns.tolist()
# 
# # Train/test split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, stratify=y, random_state=42
# )
# 
# # Create pipeline with overfitting control parameters
# preprocessor = ColumnTransformer([
#     ('num', StandardScaler(), numeric_cols)
# ])
# 
# # Add max_depth and min_samples_leaf to reduce overfitting
# rf_model = RandomForestClassifier(
#     random_state=42,
#     class_weight='balanced',
#     max_depth=10,          # limit tree depth
#     min_samples_leaf=5,    # minimum samples per leaf
#     n_estimators=100       # number of trees, you can tune this as well
# )
# 
# pipeline = Pipeline([
#     ('preprocessor', preprocessor),
#     ('classifier', rf_model)
# ])
# 
# @st.cache_resource(show_spinner=False)
# def train_pipeline():
#     pipeline.fit(X_train, y_train)
#     return pipeline
# 
# pipeline = train_pipeline()
# 
# # Show classification report on test set
# y_pred = pipeline.predict(X_test)
# st.subheader("Classification Report (Test Set)")
# st.text(classification_report(y_test, y_pred))
# 
# # Confusion matrix plot
# cm = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])
# fig, ax = plt.subplots(figsize=(6,4))
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High'])
# disp.plot(ax=ax)
# plt.title("Confusion Matrix (Test Set)")
# st.pyplot(fig)
# # -----------------------
# # Interactive Prediction Inputs
# st.subheader("Predict Life Expectancy Category (Interactive)")
# 
# 
# input_data = {}
# for feature in numeric_cols:
#     min_val = float(df[feature].min())
#     max_val = float(df[feature].max())
#     mean_val = float(df[feature].mean())
#     input_data[feature] = st.slider(
#         f"{feature}",
#         min_val,
#         max_val,
#         mean_val,
#         step=(max_val - min_val) / 100,
#         key=f"classify_{feature}"
#     )
# 
# input_df = pd.DataFrame([input_data])
# 
# prediction = pipeline.predict(input_df)[0]
# prediction_proba = pipeline.predict_proba(input_df)[0]
# 
# st.write(f"Predicted Life Expectancy Category: **{prediction}**")
# 
# # Show prediction probabilities as a bar chart
# proba_df = pd.DataFrame({
#     'Category': pipeline.classes_,
#     'Probability': prediction_proba
# }).sort_values(by='Probability', ascending=False)
# 
# st.bar_chart(proba_df.set_index('Category'))
# 
# st.header("Clustering: Country Groups based on Development & Health Indicators")
# 
# # ****** K-means *********
# 
# # Log transform Per Capita (skewed)
# if 'Per Capita' in df_kmeans.columns:
#     df_kmeans['log_gdp_per_capita'] = np.log1p(df_kmeans['Per Capita'])
# else:
#     st.warning("Column 'Per Capita' missing for clustering log transform.")
# 
# # Select numeric columns (exclude Year, Per Capita original)
# exclude_cols = ['Year', 'Per Capita', 'Country', 'Gender']
# numeric_cols = [col for col in df_kmeans.select_dtypes(include='number').columns if col not in exclude_cols]
# 
# if not numeric_cols:
#     st.warning("No numeric columns available for clustering.")
# else:
#     data = df_kmeans[numeric_cols].dropna()
# 
#     # Standardize data
#     scaler = StandardScaler()
#     scaled_data = scaler.fit_transform(data)
# 
#     # Find best k with silhouette score
#     silhouette_scores = []
#     K_range = range(2, 10)
#     for k in K_range:
#         kmeans = KMeans(n_clusters=k, random_state=42)
#         labels = kmeans.fit_predict(scaled_data)
#         score = silhouette_score(scaled_data, labels)
#         silhouette_scores.append(score)
# 
#     # Plot silhouette scores
#     fig, ax = plt.subplots(figsize=(8, 4))
#     ax.plot(list(K_range), silhouette_scores, marker='o')
#     ax.set_title("Silhouette Score for KMeans Clustering")
#     ax.set_xlabel("Number of clusters (k)")
#     ax.set_ylabel("Silhouette Score")
#     ax.grid(True)
#     st.pyplot(fig)
# 
#     # Choose best k
#     best_k = K_range[np.argmax(silhouette_scores)]
#     st.write(f"Best k by silhouette score: **{best_k}**")
# 
#     # Final KMeans with best k
#     kmeans_final = KMeans(n_clusters=best_k, random_state=42)
#     cluster_labels = kmeans_final.fit_predict(scaled_data)
# 
#     # Assign cluster labels back (only to rows without missing)
#     df_kmeans.loc[data.index, 'Cluster'] = cluster_labels
# 
#     # Prepare country names for plotly choropleth
#     # Replace country names to ISO standard using pycountry and alt_names dict
#     alt_names = {
#       'USA': 'United States',
#       'UK': 'United Kingdom',
#       'South Korea': 'Korea, Republic of',
#       'North Korea': "Korea, Democratic People's Republic of",
#       'Russia': 'Russian Federation',
#       'Iran': 'Iran, Islamic Republic of',
#       'Syria': 'Syrian Arab Republic',
#       'Vietnam': 'Viet Nam',
#       'Tanzania': 'Tanzania, United Republic of',
#       'Bahamas, The': 'Bahamas',
#       'Bolivia': 'Bolivia, Plurinational State of',
#       'Channel Islands': 'Guernsey',  # Or Jersey, no ISO for "Channel Islands" as a group
#       'Congo, Dem. Rep.': 'Democratic Republic of the Congo',
#       'Congo, Dem. Rep.': 'Congo, the Democratic Republic of the',
#       'Congo, Rep.': 'Congo',
#       "Cote d'Ivoire": "Côte d'Ivoire",
#       'Curacao': 'Curaçao',
#       'Czech Republic': 'Czechia',
#       'Egypt, Arab Rep.': 'Egypt',
#       'Gambia, The': 'Gambia',
#       'Heavily indebted poor countries (HIPC)': None,
#       'Hong Kong SAR, China': 'Hong Kong',
#       'Iran, Islamic Rep.': 'Iran, Islamic Republic of',
#       "Korea, Dem. People's Rep.": "Korea, Democratic People's Republic of",
#       'Korea, Rep.': 'Korea, Republic of',
#       'Kosovo': None,  # Not in ISO
#       'Kyrgyz Republic': 'Kyrgyzstan',
#       'Lao PDR': "Lao People's Democratic Republic",
#       'Macao SAR, China': 'Macao',
#       'Micronesia, Fed. Sts.': 'Micronesia, Federated States of',
#       'Moldova': 'Moldova, Republic of',
#       'Slovak Republic': 'Slovakia',
#       'St. Kitts and Nevis': 'Saint Kitts and Nevis',
#       'St. Lucia': 'Saint Lucia',
#       'St. Martin (French part)': 'Saint Martin (French part)',
#       'St. Vincent and the Grenadines': 'Saint Vincent and the Grenadines',
#       'Turkey': 'Türkiye',
#       'Venezuela, RB': 'Venezuela, Bolivarian Republic of',
#       'Virgin Islands (U.S.)': 'Virgin Islands, U.S.',
#       'West Bank and Gaza': 'Palestine, State of',
#       'Yemen, Rep.': 'Yemen'
#     }
#     df_kmeans['Country'] = df_kmeans['Country'].replace(alt_names)
# 
#     # Filter data for plotting: remove rows with missing Cluster or Country
#     plot_df = df_kmeans.dropna(subset=['Cluster', 'Country', 'Year'])
# 
#     # Convert cluster to int for coloring
#     plot_df['Cluster'] = plot_df['Cluster'].astype(int)
# 
#     # Plot interactive choropleth with year slider
#     fig = px.choropleth(
#         plot_df,
#         locations='Country',
#         locationmode='country names',
#         color='Cluster',
#         animation_frame='Year',
#         title='Country Clusters Based on Development and Health Indicators Over Time',
#         color_continuous_scale=px.colors.qualitative.Set1
#     )
#     st.plotly_chart(fig, use_container_width=True)
# 
#     # Show cluster means table
#     st.subheader("Cluster Centers (mean values)")
#     cluster_means = plot_df.groupby('Cluster')[numeric_cols].mean().round(2)
#     st.dataframe(cluster_means)
#


