# import packages
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import streamlit as st

st.markdown("# Thomas Hoover's OPAN6607 Final Project")

st.markdown("### This app will provide a probability of how likely a person is to use linked in based on the given inputs")

def log_app():
    # 1: Read the data
    s = pd.read_csv("social_media_usage.csv")

    ss = s[['income', 'educ2', 'par', 'marital', 'gender', 'age']]
    ss.loc[:,'sm_li'] = s['web1h'].apply(clean_sm)
    ss.loc[:,'income'] = np.where(s['income'] > 9, np.nan, ss['income'])
    ss.loc[:,'educ2'] = np.where(s['educ2'] > 8, np.nan, ss['educ2'])
    ss.loc[:,'par'] = s['par'].apply(clean_sm)
    ss.loc[:,'marital'] = s['marital'].apply(clean_sm)
    ss.loc[:,'gender'] = np.where(s['gender'] == 2, 1, 0)
    ss.loc[:,'age'] = np.where(s['age'] > 98, np.nan, ss['age'])

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


def pred_from_new_data():
    newdata = pd.DataFrame({
        "income": [8],
        "educ2": [7],
        "par": [0],
        "marital": [1],
        "gender": [1],
        "age": [42]
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

    print(f"Predicted class (82 year old): %s" % sm_pred2)
    print(f"Probability that this person is a linked in user (82 year old): {probs2[0][1]}")

def clean_sm(x):
    x = np.where(x==1,1,0)
    return x