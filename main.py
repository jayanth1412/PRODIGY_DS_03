import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Bank Marketing Dashboard",
    layout="wide"
)

st.title("📊 Bank Marketing Decision Tree Dashboard")
st.markdown(
    "Predict whether a customer will purchase a product/service"
)

# ======================================================
# LOAD DATA
# ======================================================

@st.cache_data
def load_data(file_path):

    try:
        df = pd.read_csv("data/bank-additional.csv", sep=";")

        if df.shape[1] == 1:
            df = pd.read_csv("data/bank-additional.csv")

    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

    return df


df = load_data("data/bank-additional.csv")

if df is None:
    st.stop()

# ======================================================
# DATA PREVIEW
# ======================================================

st.subheader("Dataset Preview")

st.write("Shape:", df.shape)
st.dataframe(df.head())

# ======================================================
# PREPROCESSING
# ======================================================

df_encoded = df.copy()

label_encoders = {}

categorical_columns = df_encoded.select_dtypes(
    include=["object"]
).columns

for col in categorical_columns:

    le = LabelEncoder()

    df_encoded[col] = le.fit_transform(
        df_encoded[col].astype(str)
    )

    label_encoders[col] = le

# ======================================================
# SIDEBAR SETTINGS
# ======================================================

st.sidebar.header("Model Settings")

max_depth = st.sidebar.slider(
    "Max Depth",
    min_value=1,
    max_value=20,
    value=5
)

criterion = st.sidebar.selectbox(
    "Criterion",
    ["gini", "entropy"]
)

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.1,
    max_value=0.4,
    value=0.2
)

# ======================================================
# TRAIN MODEL
# ======================================================

X = df_encoded.drop("y", axis=1)
y = df_encoded["y"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=42,
    stratify=y
)

model = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ======================================================
# METRICS
# ======================================================

st.subheader("Model Performance")

col1, col2 = st.columns(2)

with col1:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col2:
    st.metric("Test Samples", len(X_test))

# ======================================================
# CLASSIFICATION REPORT
# ======================================================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)

# ======================================================
# CONFUSION MATRIX
# ======================================================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(6, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)

# ======================================================
# FEATURE IMPORTANCE
# ======================================================

st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig2, ax2 = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    ax=ax2
)

st.pyplot(fig2)

# ======================================================
# DECISION TREE VISUALIZATION
# ======================================================

st.subheader("Decision Tree Visualization")

fig3, ax3 = plt.subplots(figsize=(25, 10))

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["No", "Yes"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax3
)

st.pyplot(fig3)

# ======================================================
# INTERACTIVE PREDICTION
# ======================================================

st.subheader("Customer Purchase Prediction")

sample_input = {}

for col in X.columns:
    sample_input[col] = st.number_input(
        col,
        value=float(X[col].mean())
    )

input_df = pd.DataFrame([sample_input])

if st.button("Predict"):

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success(
            "Customer WILL purchase/subscribed"
        )
    else:
        st.error(
            "Customer WILL NOT purchase"
        )