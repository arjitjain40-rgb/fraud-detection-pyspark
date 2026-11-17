
from pyspark.sql.functions import col, lit
from pyspark.ml.functions import vector_to_array



def generate_ensemble_predictions(validation_df, models, weights):

    predictions = validation_df

    for model_name, model in models:

        probability_column = (
            "prob_" +
            model_name.lower().replace(" ", "_").replace("-", "_")
        )

        model_copy = model.copy({
            model.probabilityCol: probability_column
        })

        predictions = model_copy.transform(predictions)

        # Extract probability of class 1 (fraud) as a numeric value
        predictions = predictions.withColumn(
            probability_column,
            vector_to_array(col(probability_column))[1]
        )

        predictions = predictions.drop(
            "rawPrediction",
            "prediction"
        )

    return predictions



def calculate_ensemble_probability(predictions, weights):

    ensemble_expression = lit(0.0)

    for model_name, weight in weights.items():

        probability_column = (
            "prob_" +
            model_name.lower().replace(" ", "_").replace("-", "_")
        )

        ensemble_expression += (
            col(probability_column) * float(weight)
        )

    predictions = predictions.withColumn(
        "ensemble_probability",
        ensemble_expression
    )

    return predictions
