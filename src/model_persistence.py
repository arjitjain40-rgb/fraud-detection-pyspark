
import os
import json

from pyspark.ml.classification import (
    RandomForestClassificationModel,
    GBTClassificationModel,
    LogisticRegressionModel,
    DecisionTreeClassificationModel
)


MODEL_PATHS = {
    "Random Forest": "models/random_forest",
    "Gradient-Boosted Trees": "models/gradient_boosted_trees",
    "Weighted Logistic Regression": "models/weighted_logistic_regression",
    "Decision Tree": "models/decision_tree"
}


MODEL_CLASSES = {
    "Random Forest": RandomForestClassificationModel,
    "Gradient-Boosted Trees": GBTClassificationModel,
    "Weighted Logistic Regression": LogisticRegressionModel,
    "Decision Tree": DecisionTreeClassificationModel
}


CONFIG_PATH = "models/ensemble_config.json"


def saved_models_available():

    if not os.path.isfile(CONFIG_PATH):
        return False

    for path in MODEL_PATHS.values():
        if not os.path.isdir(path):
            return False

    return True


def save_models(models):

    os.makedirs("models", exist_ok=True)

    for model_name, model in models:

        path = MODEL_PATHS[model_name]

        model.write().overwrite().save(path)

        print(f"{model_name} saved successfully.")


def load_models():

    models = []

    for model_name, model_class in MODEL_CLASSES.items():

        path = MODEL_PATHS[model_name]

        model = model_class.load(path)

        models.append((model_name, model))

        print(f"{model_name} loaded successfully.")

    return models


def save_ensemble_config(weights, threshold):

    os.makedirs("models", exist_ok=True)

    config = {
        "weights": {
            name: float(weight)
            for name, weight in weights.items()
        },
        "threshold": float(threshold)
    }

    with open(CONFIG_PATH, "w") as file:
        json.dump(config, file, indent=4)

    print("Ensemble configuration saved successfully.")


def load_ensemble_config():

    with open(CONFIG_PATH, "r") as file:
        config = json.load(file)

    return config["weights"], config["threshold"]
