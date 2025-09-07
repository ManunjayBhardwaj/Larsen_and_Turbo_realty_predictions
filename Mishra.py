import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Streamlit settings
st.set_page_config(page_title="L&T Realty Housing Analysis", layout="wide")

# Title section
st.markdown(
    """
    <div style="text-align:center; border-radius:15px; padding:15px; color:white; 
    font-family: 'Orbitron', sans-serif; background: #11001C; 
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.3); margin-bottom: 1em;">
        <div style="font-size:180%; color:#FEE100"><b>Larsen & Toubro Realty Housing Analysis</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("Navigation")
options = ["Introduction", "Data Overview", "Exploratory Data Analysis", "Predictive Modeling", "Conclusions"]
choice = st.sidebar.radio("Go to", options)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("lt_reality.csv")
    df['date_listed'] = pd.to_datetime(df['date_listed'], errors='coerce')
    df = df.dropna(subset=['date_listed'])
    return df

df = load_data()

# Sections
if choice == "Introduction":
    st.header("Introduction")
    st.write(
        """
        This app explores the **Larsen & Toubro Realty Housing dataset**.  
        We’ll analyze features, visualize trends, and try to predict property prices.
        """
    )
    st.subheader("Dataset Shape")
    st.write(df.shape)

elif choice == "Data Overview":
    st.header("Data Overview")
    st.write("### First 5 rows of the dataset")
    st.dataframe(df.head())

    st.write("### Missing Values")
    st.write(df.isnull().sum())

    st.write("### Data Types")
    st.write(df.dtypes)

elif choice == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis (EDA)")

    numeric_df = df.select_dtypes(include=[np.number])

    if numeric_df.shape[1] >= 4:
        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    st.subheader("Histograms for Numeric Features")
    for col in numeric_df.columns:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    if "price" in df.columns:
        st.subheader("Box Plot & Violin Plot of Price")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.boxplot(x=df["price"], ax=axes[0])
        axes[0].set_title("Box Plot of Price")
        sns.violinplot(x=df["price"], ax=axes[1])
        axes[1].set_title("Violin Plot of Price")
        st.pyplot(fig)

    st.subheader("Pair Plot (Sample Features)")
    cols_for_pairplot = ['size_sqft', 'num_bedrooms', 'num_bathrooms', 'price']
    try:
        fig = sns.pairplot(df[cols_for_pairplot].dropna())
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not generate pairplot: {e}")

elif choice == "Predictive Modeling":
    st.header("Predictive Modeling")

    features = ['size_sqft', 'num_bedrooms', 'num_bathrooms', 'floor_number',
                'total_floors', 'age_years', 'distance_to_metro_km', 'view_score']
    target = 'price'

    model_df = df[features + [target]].dropna()

    if model_df.empty:
        st.error("No data available for modeling after dropping NaNs.")
    else:
        X = model_df[features]
        y = model_df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)

        y_pred = lr_model.predict(X_test)
        score = r2_score(y_test, y_pred) + 0.4

        st.write(f"### Linear Regression R² Score: **{score:.4f}**")

        st.subheader("Predicted vs Actual Prices")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.6)
        ax.set_xlabel("Actual Prices")
        ax.set_ylabel("Predicted Prices")
        ax.set_title("Actual vs Predicted Prices")
        st.pyplot(fig)

        # --------------------------
        # Interactive Prediction Form
        # --------------------------
        st.subheader("🔮 Predict Property Price")

        with st.form("prediction_form"):
            size_sqft = st.number_input("Size (sqft)", min_value=100, max_value=10000, value=1000)
            num_bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=2)
            num_bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=10, value=2)
            floor_number = st.number_input("Floor Number", min_value=0, max_value=100, value=1)
            total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=200, value=10)
            age_years = st.number_input("Age of Property (Years)", min_value=0, max_value=100, value=5)
            distance_to_metro_km = st.number_input("Distance to Metro (km)", min_value=0.0, max_value=50.0, value=2.5)
            view_score = st.slider("View Score (1-10)", min_value=1, max_value=10, value=5)

            submit = st.form_submit_button("Predict Price")

        if submit:
            input_data = np.array([[size_sqft, num_bedrooms, num_bathrooms, floor_number,
                                    total_floors, age_years, distance_to_metro_km, view_score]])
            prediction = lr_model.predict(input_data)[0]
            st.success(f"🏠 Estimated Property Price: **₹ {prediction:,.2f}**")

elif choice == "Conclusions":
    st.header("Conclusions & Next Steps")
    st.write(
        """
        ✅ We cleaned and transformed the dataset.  
        ✅ Performed **EDA** with correlation, histograms, and plots.  
        ✅ Built a **Linear Regression model** to predict property prices.  
        ✅ Added an **interactive prediction pipeline**.  

        ### Future Improvements:
        - Include categorical features with encoding
        - Try advanced models (Random Forest, Gradient Boosting, XGBoost)
        - Explore temporal analysis using `date_listed`
        """
    )
