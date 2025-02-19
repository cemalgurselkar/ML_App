import streamlit as st
import pandas as pd
import os
import requests
import time
from streamlit_modal import Modal
from io import BytesIO
import joblib
import base64

st.set_page_config(page_title="Model Training App", layout="centered")
st.title("Model Training App")

st.markdown(
    """
    <style>
    /* Modal pencerenin container'ını hedefleyin */
    div[data-testid="stModalContainer"] > div {
        width: 900px !important;
        height: 800px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.sidebar.title("Uploading File")
uploaded_file = st.sidebar.file_uploader("Please upload here", type=["csv", "xlsx"])

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

if uploaded_file is not None:
    data = load_file(uploaded_file)
    if data is not None:
        st.session_state['data'] = data
        st.write("Veri başarıyla yüklendi!")

        if "training_results" not in st.session_state:
            st.session_state.training_results = None

        modal = Modal("model_train_modal", "Model Training Options")

        if st.button("Model Train"):
            modal.open()

        if modal.is_open():
            with modal.container():
                st.header("Model Training Options")
                select_model = st.selectbox('Select Model:', ['SVR', 'MLPRegression', 'XGBoostRegression'])
                scaler_method = st.selectbox('Select Scaler:', ['None', 'Standart Scaler', 'Min-Max Scaler'])
                split_method = st.selectbox('Select Split Method:', ['Cross Validation', 'Random Split', 'Leave One Out'])
                if split_method == 'Random Split':
                    test_size = st.slider("Select Test Size:", min_value=0.1, max_value=0.5, step=0.05)
                elif split_method == 'Cross Validation':
                    cv_fold = st.slider("Select number of folds for Cross Validation", min_value=3
                    ,max_value=10, step=1)
                tuning_method = st.selectbox("Select Hyperparameter Tuning Method", ['None', 'Manuel', 'Grid Search'])
                
                manual_params = {}
                if tuning_method == "Manuel":
                    if select_model == "SVR":
                        C_value = st.number_input("C:", value=1.0)
                        epsilon_value = st.number_input("Epsilon:", value=0.1)
                        kernel_value = st.selectbox("Kernel:", ["rbf", "linear", "poly", "sigmoid"])
                        manual_params = {"C": C_value, "epsilon": epsilon_value, "kernel": kernel_value}
                    elif select_model == "MLPRegression":
                        hidden_layers = st.text_input("Hidden layer sizes (comma separated)", "100")
                        activation = st.selectbox("Activation:", ["relu", "tanh", "logistic"])
                        solver = st.selectbox("Solver:", ["adam", "sgd"])
                        hidden_layers_tuple = tuple(int(x.strip()) for x in hidden_layers.split(",") if x.strip().isdigit())
                        manual_params = {"hidden_layer_sizes": hidden_layers_tuple, "activation": activation, "solver": solver, "random_state": 42}
                    elif select_model == "XGBoostRegression":
                        n_estimators = st.number_input("n_estimators:", value=100, step=10)
                        max_depth = st.number_input("max_depth:", value=3, step=1)
                        learning_rate = st.number_input("learning_rate:", value=0.1, step=0.01, format="%.2f")
                        manual_params = {"n_estimators": n_estimators, "max_depth": max_depth, "learning_rate": learning_rate, "random_state": 42}

                numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
                target_col = st.selectbox('Select Target Column:', numeric_cols)
                all_features = [col for col in numeric_cols if col != target_col]
                excluded_features = st.multiselect("Select Feature Columns to EXCLUDE:", 
                                                    options=all_features, default=[],
                                                    help="Choose any columns you do NOT want to use in the model.")
                features_cols = [col for col in all_features if col not in excluded_features]
                st.write("Final Features (to be used in model):", features_cols)
                # Özellik sütunlarını session_state'e kaydediyoruz.
                st.session_state["features_cols"] = features_cols

                if st.button('Train', key="train_button"):
                    with st.spinner("The model is being trained..."):
                        time.sleep(5)
                        payload = {
                            'model_choice': select_model,
                            'scaler_choice': scaler_method,
                            'target_col': target_col,
                            'features_col': features_cols,
                            'tuning_method': tuning_method,
                            'split_method': split_method,
                            'data': data.to_json(orient='split')
                        }
                        if tuning_method == "Manuel":
                            payload["hyperparameters"] = manual_params
                        
                        if split_method == 'Cross Validation':
                            payload['cv_folds'] = cv_fold
                        
                        if split_method == 'Random Split':
                            payload["test_size"] = test_size
                        
                        try:
                            response = requests.post("http://127.0.0.1:5000/train_model", json=payload)
                        except Exception as e:
                            st.error("API bağlantı hatası: " + str(e))
                            response = None
                        
                        if response is not None and response.status_code == 200:
                            results = response.json()
                            st.session_state.training_results = results
                        else:
                            st.error("Training failed: " + (response.text if response else "No response from API"))
                        modal.close()

        if st.session_state.get("training_results") is not None:
            results = st.session_state.training_results
            st.subheader("Model Training Results")
            if 'CV_results' in results:
                cv_results = results['CV_results']
                st.write("### Cross Validation Evaluation Metrics")
                st.write(f"CV MAPE: {cv_results.get('CV_MAPE', 'N/A')}")
                st.write(f"CV MAE: {cv_results.get('CV_MAE', 'N/A')}")
                st.write(f"CV MSE: {cv_results.get('CV_MSE', 'N/A')}")
                st.write(f"CV R2: {cv_results.get('CV_R2', 'N/A')}")
            else:
                st.write("### Test Set Evaluation Metrics")
                st.write(f"MAPE: {results.get('MAPE', 'N/A')}")
                st.write(f"MAE: {results.get('MAE', 'N/A')}")
                st.write(f"MSE: {results.get('MSE', 'N/A')}")
                st.write(f"R2: {results.get('R2', 'N/A')}")
            
            st.info("Model Training is ready to test!!")
            
            if "accepted_model_open" not in st.session_state:
                st.session_state["accepted_model_open"] = False

            if st.button("Accepted Model", key="accepted_model_button"):
                st.session_state["accepted_model_open"] = not st.session_state["accepted_model_open"]

            if st.session_state["accepted_model_open"]:
                with st.container():
                    st.header("Test Your Accepted Model")
                    test_file = st.file_uploader("Upload your test file", type=["csv", "xlsx"], key="test_file")
                    if test_file is not None:
                        test_data = load_file(test_file)
                        st.write("Test data loaded successfully!")
                        if st.button("Test Model", key="test_model_button"):
                            with st.spinner("Testing the accepted model..."):
                                try:
                                    model_pickle_b64 = results.get("model_pickle")
                                    model_bytes = base64.b64decode(model_pickle_b64)
                                    model = joblib.load(BytesIO(model_bytes))
                                    
                                    features = st.session_state.get("features_cols", None)
                                    if features is None:
                                        st.error("No feature columns info found from training.")
                                    else:
                                        X_test = test_data.copy()
                                        missing_cols = [col for col in features if col not in X_test.columns]
                                        if missing_cols:
                                            st.error("Test data is missing the following required columns: " + ", ".join(missing_cols))
                                        else:
                                            X_test = X_test[features]
                                            predictions = model.predict(X_test)
                                            test_data["Predictions"] = predictions
                                            st.write("Prediction results:")
                                            st.dataframe(test_data["Predictions"])
                                except Exception as e:
                                    st.error("Error testing the model: " + str(e))
