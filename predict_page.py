import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("WORTHIFY.ME")

    st.write("""### We need some information to predict the salary""")
    with st.expander("Why fill this out?"):
        st.write("""
            Knowing the market rate for your skills can help you negotiate better salaries and understand your position in the job market.
        """)
    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is {salary[0]:.2f}")

        
    st.write("---")
    st.sidebar.header("About")
    st.sidebar.info("""
        This app was created by:
        - HARSH_BIR
        - PRIYANSHU_DAYAL
        - SHANSKAR_BANSAL
        - SALONI_THAKUR
        """)
    st.sidebar.header("Contribute")
    st.sidebar.write("Want to improve WORTHIFY.ME? Check out our GitHub repository!")
    st.sidebar.markdown("[GITHUB REPOSITORY](https://github.com)")

    st.sidebar.header("Share")
    st.sidebar.write("Like the app? Share it with your friends and colleagues!")
    st.sidebar.button("Share on LinkedIn")