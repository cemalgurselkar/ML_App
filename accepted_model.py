import streamlit as st
import pandas as pd
from io import BytesIO
import joblib
import base64
import os

st.set_page_config(page_title="Accepted Model Testing", layout="centered")
st.title("Accepted Model Testing")

def load_file(file):
    filename = file.name
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if ext == '.csv':
        return pd.read_csv(file)
    elif ext == '.xlsx':
        return pd.read_excel(file)
    else:
        st.error("Unsupported file type!")
        return None

if "training_results" not in st.session_state:
    st.error("No trained model found. Please train a model first from the main page.")
    st.stop()

results = st.session_state.training_results

st.write("Trained model loaded successfully.")

test_file = st.file_uploader("Upload your test file", type=["csv", "xlsx"])
if test_file is not None:
    test_data = load_file(test_file)
    if test_data is not None:
        st.write("Test data loaded successfully!")
        
        if st.button("Test Model"):
            with st.spinner("Testing the accepted model..."):
                try:
                    model_pickle_b64 = results.get("model_pickle")
                    model_bytes = base64.b64decode(model_pickle_b64)
                    model = joblib.load(BytesIO(model_bytes))

                    features = st.session_state.get("features_cols", None)
                    if features is None:
                        st.error("No feature columns info found from the training session.")
                    else:
                        try:
                            X_test = test_data[features]
                        except Exception as e:
                            st.error("Test data does not contain the required feature columns: " + str(e))
                            X_test = test_data
                        
                        predictions = model.predict(X_test)
                        test_data["Predictions"] = predictions
                        st.write("Prediction results:")
                        st.dataframe(test_data)
                except Exception as e:
                    st.error("An error occurred while testing the model: " + str(e))
