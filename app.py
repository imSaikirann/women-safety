import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# small sample dataset
tweets = [
    ("Street is very dark and unsafe at night", 1),
    ("Harassment reported near metro station", 1),
    ("Police patrolling makes area safe", 0),
    ("Well lit roads and security cameras installed", 0),
    ("Eve teasing incidents increasing in this area", 1),
    ("Safe environment for women travelers", 0),
    ("Lack of street lights causing safety issues", 1),
    ("Women helpline working efficiently", 0),
    ("Crowded bus stop with no police security", 1),
    ("Good street lighting improves safety", 0),
    ("Frequent harassment complaints near college road", 1),
    ("Women police patrol at night ensures safety", 0),
    ("Unsafe alley with no lighting", 1),
    ("Security guards present in shopping area", 0),
    ("Reports of theft and harassment increasing", 1),
    ("Safe residential area with CCTV cameras", 0),
    ("Women feel unsafe walking alone at night", 1),
    ("Well maintained public transport system", 0),
    ("Dark roads without security patrol", 1),
    ("Emergency helpline working properly", 0),
    ("High crime rate reported in this neighborhood", 1),
    ("Police station located nearby provides safety", 0),
    ("Unsafe bus route late at night", 1),
    ("Security checkpoints installed across city", 0),
    ("No lighting near park causing fear among women", 1),
    ("Public places monitored by CCTV cameras", 0),
    ("Increase in harassment cases reported downtown", 1),
    ("Women safety awareness programs conducted", 0),
    ("Lack of police presence makes area unsafe", 1),
    ("Safe shopping complex with guards", 0),
]


df = pd.DataFrame(tweets, columns=["tweet","label"])

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["tweet"])
y = df["label"]

model = LogisticRegression()
model.fit(X,y)

# Streamlit UI
st.title("Women Safety Tweet Analyzer")

user_input = st.text_area("Enter Tweet")

if st.button("Analyze"):
    vec = vectorizer.transform([user_input])
    pred = model.predict(vec)

    if pred[0] == 1:
        st.error("Unsafe Area Mentioned")
    else:
        st.success("Safe Area Mentioned")
