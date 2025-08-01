import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

st.title("🎓 Student Performance Predictor")

uploaded_file = st.file_uploader("Upload student_mat.csv", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(data.head())

    df = data.copy()
    df['pass'] = df['G3'].apply(lambda x: 1 if x >= 10 else 0)
    df.drop(columns=['G3'], inplace=True)

    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    X = df.drop('pass', axis=1)
    y = df['pass']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📈 Model Evaluation")
    st.write(f"Accuracy: **{accuracy_score(y_test, y_pred):.2f}**")
    st.code(classification_report(y_test, y_pred))

    st.subheader("🔮 Predict for New Student")
    input_data = {}
    for col in X.columns:
        if df[col].nunique() <= 10:
            input_data[col] = st.selectbox(col, sorted(df[col].unique()))
        else:
            input_data[col] = st.slider(col, int(df[col].min()), int(df[col].max()), int(df[col].mean()))

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success("Prediction: ✅ Pass" if prediction else "Prediction: ❌ Fail")
