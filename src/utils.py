import os
import sys
import dill
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import CustomException
import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as obj_file:
            dill.dump(obj, obj_file)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        best_model = None
        best_model_name = None
        best_score = float("-inf")

        for name, model in models.items():
            logging.info(f"Training {name}...")

            param = params.get(name, {})
            gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
            gs.fit(x_train, y_train)

            best_estimator = gs.best_estimator_
            y_test_pred = best_estimator.predict(x_test)
            score = r2_score(y_test, y_test_pred)
            report[name] = score

            logging.info(f"{name} R² score: {score:.4f}")

            if score > best_score:
                best_score = score
                best_model = best_estimator
                best_model_name = name

        return report, best_model, best_model_name

    except Exception as e:
        raise CustomException(e, sys)
