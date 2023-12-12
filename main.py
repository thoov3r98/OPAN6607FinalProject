# import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st


st.markdown("<h1 style='text-align: center;'>Thomas Hoover's OPAN6607 Final Project</h1>", unsafe_allow_html=True)

st.markdown(
    "<h3 style='text-align: center;'>This app will provide a probability of how likely a person is to use LinkedIn based on the given inputs</h3>",
    unsafe_allow_html=True
)


# Sidebar header
st.sidebar.header("User Inputs")

# User input fields
income_slider = st.sidebar.slider("Select your income:", min_value=0, max_value=151000, value=50000, step=100)

# Map income to the specified ranges
income_mapping = {
    (0, 10000): 1,
    (10000, 20000): 2,
    (20000, 30000): 3,
    (30000, 40000): 4,
    (40000, 50000): 5,
    (50000, 75000): 6,
    (75000, 100000): 7,
    (100000, 150000): 8,
    (150000, 151000): 9,
}

# Determine the income range based on the slider value
for income_range, mapped_value in income_mapping.items():
    if income_slider >= income_range[0] and income_slider < income_range[1]:
        income = mapped_value
        break
educ2 = st.sidebar.text_input("Enter your education level:", value="Enter a number here")
par = st.sidebar.selectbox("Are you a parent?", ('Yes', 'No'))
marital = st.sidebar.selectbox("Are you married?", ('Yes', 'No'))
gender = st.sidebar.selectbox("Select your gender", ('Male', 'Female', 'Other'))
age = st.sidebar.slider("How old are you?", min_value=18, max_value=100, value=25)

# Display user inputs
st.subheader("User Inputs:")
st.write(f"**Income:** {income}")
st.write(f"**Education Level:** {educ2}")
st.write(f"**Parent:** {par}")
st.write(f"**Marital Status:** {marital}")
st.write(f"**Gender:** {gender}")
st.write(f"**Age:** {age}")


def train_model():
    s = pd.read_csv("social_media_usage.csv")

    ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age']]
    ss.loc[:, 'sm_li'] = s['web1h'].apply(clean_sm)
    ss.loc[:, 'income'] = np.where(s['income'] > 9, np.nan, ss['income'])
    ss.loc[:, 'educ2'] = np.where(s['educ2'] > 8, np.nan, ss['educ2'])
    ss.loc[:, 'par'] = s['par'].apply(clean_sm)
    ss.loc[:, 'marital'] = s['marital'].apply(clean_sm)
    ss.loc[:, 'gender'] = np.where(s['gender'] == 2, 1, 0)
    ss.loc[:, 'age'] = np.where(s['age'] > 98, np.nan, ss['age'])

    ss = ss.dropna()

    y = ss["sm_li"]
    X = ss[["income", "educ2", "par", "marital", "gender", "age"]]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        stratify=y,
                                                        test_size=0.2,
                                                        random_state=987)

    lr = LogisticRegression(class_weight='balanced')
    lr.fit(X_train, y_train)
    return lr


def pred_from_new_data(income, educ2, par, marital, gender, age):
    lr = train_model()

    newdata = pd.DataFrame({
        "income": [income],
        "educ2": [educ2],
        "par": [par],
        "marital": [marital],
        "gender": [gender],
        "age": [age]
    })

    prediction = lr.predict(newdata)
    probs = lr.predict_proba(newdata)
    if prediction == 1:
        sm_pred = "Linkedin User"
    else:
        sm_pred = "Not a Linkedin User"

    print(f"Predicted class (42 year old): %s" % sm_pred)
    print(f"Probability that this person is a linked in user (42 year old): {probs[0][1]}\n")

    newdata2 = pd.DataFrame({
        "income": [8],
        "educ2": [7],
        "par": [0],
        "marital": [1],
        "gender": [1],
        "age": [82]
    })

    prediction2 = lr.predict(newdata2)
    probs2 = lr.predict_proba(newdata2)

    if prediction2 == 1:
        sm_pred2 = "Linkedin User"
    else:
        sm_pred2 = "Not a Linkedin User"

    st.write(f"Predicted class (82 year old): %s" % sm_pred2)
    st.write("Probability that this person is a linked in user (82 year old): {probs2[0][1]}")


def clean_sm(x):
    x = np.where(x == 1, 1, 0)
    return x


if st.button("Generate Prediction"):
    pred_from_new_data(income, educ2, par, marital, gender, age)
