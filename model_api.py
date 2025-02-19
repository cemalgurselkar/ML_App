from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import base64
from sklearn.model_selection import train_test_split, LeaveOneOut, cross_val_predict, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

app = Flask(__name__)

def calculate_mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100

@app.route('/train_model', methods=['POST'])
def model_train():
    try:
        data_json = request.get_json()

        model_choice = data_json.get('model_choice', 'SVR')
        scaler_choice = data_json.get('scaler_choice', 'None')
        split_method = data_json.get('split_method', 'Random Split')
        target_col = data_json.get('target_col')
        features_col = data_json.get('features_col')
        test_size = data_json.get('test_size', 0.2)
        tuning_method = data_json.get('tuning_method', 'Grid Search')
        hyperparameters = data_json.get('hyperparameters', {})

        data_str = data_json.get("data")
        data = pd.read_json(data_str, orient="split")

        if target_col is None or not features_col:
            return jsonify({"error": "Target column or feature columns not provided."}), 400

        X = data[features_col]
        y = data[target_col]

        if scaler_choice in ["Standart Scaler","Min-Max Scaler"]:
            if scaler_choice == "Standart Scaler":
                scaler = StandardScaler()
            elif scaler_choice == "Min-Max Scaler":
                scaler = MinMaxScaler()
            X = scaler.fit_transform(X)

        if model_choice == "SVR":
            model = SVR()
            if tuning_method == "Grid Search":
                param_grid = {
                    'C': hyperparameters.get('C_range', [0.1, 1, 10]),
                    'epsilon': hyperparameters.get('epsilon_range', [0.01, 0.1, 1]),
                    'kernel': hyperparameters.get('kernel_options', ['rbf', 'linear'])
                }
                model = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
            else:
                model = SVR(**hyperparameters)
        elif model_choice == "MLPRegression":
            model = MLPRegressor()
            if tuning_method == "Grid Search":
                param_grid = {
                    'hidden_layer_sizes': hyperparameters.get('hidden_layer_sizes_range', [(50,), (100,), (150,)]),
                    'activation': hyperparameters.get('activation_options', ['relu', 'tanh']),
                    'solver': hyperparameters.get('solver_options', ['adam'])
                }
                model = GridSearchCV(MLPRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
            else:
                model = MLPRegressor(**hyperparameters)
        elif model_choice == "XGBoostRegression":
            model = XGBRegressor()
            if tuning_method == "Grid Search":
                param_grid = {
                    'n_estimators': hyperparameters.get('n_estimators_range', [50, 100, 150]),
                    'max_depth': hyperparameters.get('max_depth_range', [3, 5, 7]),
                    'learning_rate': hyperparameters.get('learning_rate_range', [0.01, 0.1, 0.2])
                }
                model = GridSearchCV(XGBRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
            else:
                model = XGBRegressor(**hyperparameters)
        else:
            return jsonify({"error": "Unknown model choice"}), 400

        result = {}
        if split_method == "Cross Validation":
            cv_folds = data_json.get("cv_folds",5)
            y_cv_preds = cross_val_predict(model, X,y,cv=cv_folds)
            cv_mape = calculate_mape(y,y_cv_preds)
            cv_mae = mean_absolute_error(y, y_cv_preds)
            cv_mse = mean_squared_error(y, y_cv_preds)
            cv_r2 = r2_score(y, y_cv_preds)
            model.fit(X,y)

            result["CV_results"] = {
                "CV_MAPE": cv_mape,
                "CV_MAE": cv_mae,
                "CV_MSE": cv_mse,
                "CV_R2": cv_r2
            }
        else:
            if split_method == "Random Split":
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            elif split_method == "Leave One Out":
                loo = LeaveOneOut()
                splits = list(loo.split(X))
                train_idx, test_idx = splits[0]
                
                if isinstance(X, pd.DataFrame):
                    X_train,X_test = X.iloc[train_idx], X.iloc[test_idx]
                else:
                    X_train, X_test = X[train_idx], X[test_idx]
                
                if isinstance(y, (pd.Series, pd.DataFrame)):
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                else:
                    y_train, y_test = y[train_idx], y[test_idx]

            else:
                return jsonify({"error": "Unknown split method."}), 400
            
            model.fit(X_train, y_train)

            if tuning_method == "Grid Search":
                used_params = model.best_params_
                estimator = model.best_estimator_
            else:
                used_params = hyperparameters
                estimator = model
            
            y_pred = model.predict(X_test)
            mape = calculate_mape(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            result = {
                "MAPE": mape,
                "MAE": mae,
                "MSE": mse,
                "R2": r2,
                "used_hyperparameters": used_params,
            }

        model_filename = "trained_model.pkl"
        joblib.dump(model, model_filename)

        with open(model_filename, "rb") as f:
            model_file = f.read()
        result["model_pickle"] = base64.b64encode(model_file).decode("utf-8")

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
