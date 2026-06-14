import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models



# CONFIG CLASS
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# MODEL TRAINER CLASS

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and testing data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

           
            # MODELS (KEYS FIXED)
            
            models = {
                "DecisionTree": DecisionTreeRegressor(),
                "RandomForest": RandomForestRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "KNN": KNeighborsRegressor(),
                "XGBoost": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor(),
            }

           
            # PARAMS (KEYS MATCH MODELS)
            
            params = {
                "DecisionTree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"]
                },

                "RandomForest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "GradientBoosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "LinearRegression": {},

                "KNN": {
                    "n_neighbors": [3, 5, 7, 9]
                },

                "XGBoost": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },

                "CatBoost": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },

                "AdaBoost": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

           
            # MODEL EVALUATION
            
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )

            logging.info(f"Model Report: {model_report}")

           
            # BEST MODEL SELECTION
           
            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            logging.info(
                f"Best Model: {best_model_name}, Score: {best_model_score}"
            )

            
            # THRESHOLD CHECK
            
            if best_model_score < 0.6:
                raise CustomException("No good model found (R2 < 0.6)")

            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

           
            # FINAL TEST SCORE
            
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)