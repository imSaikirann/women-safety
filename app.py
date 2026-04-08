import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

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
y = df["label"]

evaluation_pipeline = Pipeline(
    [
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_predictions = cross_val_predict(evaluation_pipeline, df["tweet"], y, cv=cv)
cv_accuracy_scores = cross_val_score(
    evaluation_pipeline,
    df["tweet"],
    y,
    cv=cv,
    scoring="accuracy",
)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["tweet"])

model = LogisticRegression(max_iter=1000)
model.fit(X,y)

feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_unsafe_terms = (
    pd.DataFrame({"term": feature_names, "weight": coefficients})
    .sort_values("weight", ascending=False)
    .head(10)
    .rename(columns={"weight": "importance"})
)

top_safe_terms = (
    pd.DataFrame({"term": feature_names, "weight": coefficients})
    .sort_values("weight", ascending=True)
    .head(10)
    .assign(importance=lambda data: data["weight"].abs())
    .drop(columns=["weight"])
)

label_counts = (
    df["label"]
    .map({0: "Safe", 1: "Unsafe"})
    .value_counts()
    .rename_axis("category")
    .reset_index(name="count")
)

accuracy_value = cv_accuracy_scores.mean()

metric_summary = pd.DataFrame(
    {
        "metric": ["Accuracy", "Precision", "Recall", "F1-score"],
        "score": [
            accuracy_value,
            *precision_recall_fscore_support(
                y, cv_predictions, average="binary", zero_division=0
            )[:3],
        ],
    }
)

class_metrics_values = precision_recall_fscore_support(
    y, cv_predictions, labels=[0, 1], zero_division=0
)
class_metrics = pd.DataFrame(
    {
        "class": ["Safe", "Unsafe"],
        "Precision": class_metrics_values[0],
        "Recall": class_metrics_values[1],
        "F1-score": class_metrics_values[2],
    }
).set_index("class")

confusion = confusion_matrix(y, cv_predictions, labels=[0, 1])
confusion_df = pd.DataFrame(
    confusion,
    index=["Actual Safe", "Actual Unsafe"],
    columns=["Predicted Safe", "Predicted Unsafe"],
)

# Streamlit UI
st.title("Women Safety Tweet Analyzer")
st.caption("Analyze tweet text and explore model performance graphs for faculty presentation.")

user_input = st.text_area("Enter Tweet")

if st.button("Analyze"):
    vec = vectorizer.transform([user_input])
    pred = model.predict(vec)
    probabilities = model.predict_proba(vec)[0]

    if pred[0] == 1:
        st.error("Unsafe Area Mentioned")
    else:
        st.success("Safe Area Mentioned")

    probability_df = pd.DataFrame(
        {
            "category": ["Safe", "Unsafe"],
            "probability": [probabilities[0], probabilities[1]],
        }
    ).set_index("category")

    st.subheader("Prediction Confidence")
    st.bar_chart(probability_df)

st.subheader("Model Performance")

metric_cols = st.columns(4)
metric_cols[0].metric("Accuracy", f"{accuracy_value * 100:.1f}%")
metric_cols[1].metric("Precision", f"{metric_summary.iloc[1]['score'] * 100:.1f}%")
metric_cols[2].metric("Recall", f"{metric_summary.iloc[2]['score'] * 100:.1f}%")
metric_cols[3].metric("F1-score", f"{metric_summary.iloc[3]['score'] * 100:.1f}%")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Overall Metric Scores")
    st.bar_chart(metric_summary.set_index("metric"))

with col2:
    st.subheader("Class-wise Performance")
    st.bar_chart(class_metrics)

st.subheader("Confusion Matrix")
st.dataframe(confusion_df, use_container_width=True)

st.subheader("Safety Distribution")
st.bar_chart(label_counts.set_index("category"))

col1, col2 = st.columns(2)

with col1:
    st.subheader("Top Unsafe Terms")
    st.bar_chart(top_unsafe_terms.set_index("term"))

with col2:
    st.subheader("Top Safe Terms")
    st.bar_chart(top_safe_terms.set_index("term"))
