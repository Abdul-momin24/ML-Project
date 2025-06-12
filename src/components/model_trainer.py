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
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_model, save_object


@dataclass
class ModelTrainerConfig:
    train_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Initiating the model training process")

            x_train, y_train = train_array[:, :-1], train_array[:, -1]
            x_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGBoost Regressor": XGBRegressor(verbosity=0),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "splitter": ["best", "random"],
                    "max_features": ["sqrt", "log2"],
                },
                "Random Forest": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "max_features": ["sqrt", "log2", None],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Gradient Boosting": {
                    "loss": ["squared_error", "huber", "absolute_error", "quantile"],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_features": ["sqrt", "log2"],
                },
                "Linear Regression": {},
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["ball_tree", "kd_tree", "brute"]
                },
                "XGBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [3, 5, 7, 9]
                },
                "CatBoost Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100]
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.01, 0.05, 0.1],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                }
            }

            model_report, best_model, best_model_name = evaluate_model(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            best_model_score = max(model_report.values())

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R² score >= 0.6")

            logging.info(f"Best model found: {best_model_name} with R² score: {best_model_score}")

            os.makedirs(os.path.dirname(self.model_trainer_config.train_model_file_path), exist_ok=True)

            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
