import os
import sys
import dill
from src.exception import CustomException

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)



def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name, model in models.items():

            # get params safely
            params = param.get(model_name, {})

            
            if len(params) > 0:
                gs = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=3,
                    n_jobs=-1,
                    scoring="r2"
                )

                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_

            else:
                best_model = model
                best_model.fit(X_train, y_train)

           
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)