
from src.data_loader import create_spark_session, load_fraud_data

from src.preprocessing import (
    remove_duplicates,
    prepare_features,
    split_data,
    split_training_validation
)

from src.train import (
    train_random_forest,
    train_weighted_logistic_regression,
    train_decision_tree,
    train_gradient_boosted_trees
)

from src.evaluate import (
    evaluate_model,
    select_ensemble_threshold
)

from src.ensemble import (
    generate_ensemble_predictions,
    calculate_ensemble_probability
)

from pyspark.sql.functions import (
    col, when, array, sum as spark_sum
)

from pyspark.ml.functions import array_to_vector

from pyspark.ml.evaluation import BinaryClassificationEvaluator

from src.model_persistence import (
    saved_models_available,
    save_models,
    load_models,
    save_ensemble_config,
    load_ensemble_config
)

import pandas as pd



def main():

    spark = create_spark_session()

    try:
        # 1. Load dataset
        df = load_fraud_data(spark)

        # 2. Remove duplicate transactions
        df_clean = remove_duplicates(df)

        # 3. Prepare features
        df_ml = prepare_features(df_clean)

        # 4. Split into training and test datasets
        train_df, test_df = split_data(df_ml)

        # 5. Split training data into model training and validation
        model_train_df, validation_df = split_training_validation(
            train_df
        )

        
        # ==========================================
        # MODEL TRAINING OR LOADING
        # ==========================================

        if saved_models_available():

            print("\n========== LOADING SAVED MODELS ==========")

            models = load_models()

            weights, ensemble_threshold = load_ensemble_config()

            print("\nSaved ensemble weights:")
            for model_name, weight in weights.items():
                print(f"{model_name}: {weight:.4f}")

            print("Loaded ensemble threshold:", ensemble_threshold)


        else:

            print("\n========== TRAINING FOUR MODELS ==========")

            rf_model = train_random_forest(model_train_df)

            gbt_model = train_gradient_boosted_trees(model_train_df)

            lr_model = train_weighted_logistic_regression(model_train_df)

            dt_model = train_decision_tree(model_train_df)

            models = [
                ("Random Forest", rf_model),
                ("Gradient-Boosted Trees", gbt_model),
                ("Weighted Logistic Regression", lr_model),
                ("Decision Tree", dt_model)
            ]

            # ==========================================
            # VALIDATION AND MODEL WEIGHTS
            # ==========================================

            results = []

            print("\n========== VALIDATION RESULTS ==========")

            for model_name, model in models:

                print(f"\nEvaluating {model_name}...")

                result = evaluate_model(
                    model,
                    validation_df,
                    model_name
                )

                results.append(result)

            results_df = pd.DataFrame(results)

            results_df["Ensemble Score"] = (
                0.7 * results_df["PR-AUC"]
                + 0.3 * results_df["ROC-AUC"]
            )

            total_score = results_df["Ensemble Score"].sum()

            results_df["Weight"] = (
                results_df["Ensemble Score"] / total_score
            )

            weights = dict(
                zip(results_df["Model"], results_df["Weight"])
            )

            print("\n========== MODEL WEIGHTS ==========")

            for model_name, weight in weights.items():
                print(f"{model_name}: {weight:.4f}")

            results_df.to_csv(
                "validation_model_comparison.csv",
                index=False
            )

            print("\nModel comparison saved.")

            # ==========================================
            # VALIDATION ENSEMBLE
            # ==========================================

            print("\nGenerating validation ensemble predictions...")

            ensemble_predictions = generate_ensemble_predictions(
                validation_df, models, weights
            )

            ensemble_predictions = calculate_ensemble_probability(
                ensemble_predictions, weights
            )

            validation_count = validation_df.count()
            ensemble_count = ensemble_predictions.count()

            print("Validation rows:", validation_count)
            print("Ensemble rows:", ensemble_count)

            if validation_count != ensemble_count:
                raise ValueError(
                    "Ensemble row count does not match validation data."
                )

            # ==========================================
            # THRESHOLD TUNING
            # ==========================================

            ensemble_threshold = select_ensemble_threshold(
                ensemble_predictions
            )

            # ==========================================
            # SAVE MODELS AND CONFIGURATION
            # ==========================================
            try:
                print("\n========== SAVING MODELS ==========")
                save_models(models)
                save_ensemble_config(
                    weights,
                    ensemble_threshold
                )
            except Exception as e:
                print(f"\n[Warning] Could not persist models to disk: {e}")
                print("Continuing with test set evaluation and CSV generation...")

        # ==========================================
        # FINAL TEST SET EVALUATION
        # ==========================================

        print("\nGenerating test ensemble predictions...")

        test_ensemble = generate_ensemble_predictions(
            test_df, models, weights
        )

        test_ensemble = calculate_ensemble_probability(
            test_ensemble, weights
        )

        test_ensemble = test_ensemble.withColumn(
            "ensemble_prediction",
            (col("ensemble_probability") >= ensemble_threshold).cast("double")
        )

        counts = test_ensemble.agg(
        spark_sum(
            ((col("label") == 1) &
            (col("ensemble_prediction") == 1)).cast("int")
        ).alias("TP"),

        spark_sum(
            ((col("label") == 0) &
            (col("ensemble_prediction") == 1)).cast("int")
        ).alias("FP"),

        spark_sum(
            ((col("label") == 1) &
            (col("ensemble_prediction") == 0)).cast("int")
        ).alias("FN"),

        spark_sum(
            ((col("label") == 0) &
            (col("ensemble_prediction") == 0)).cast("int")
        ).alias("TN")
    ).first()

        tp = counts["TP"] or 0
        fp = counts["FP"] or 0
        fn = counts["FN"] or 0
        tn = counts["TN"] or 0

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        # Convert scalar ensemble probability to a Spark vector for AUC evaluation
        test_ensemble = test_ensemble.withColumn(
            "ensemble_raw_prediction",
            array_to_vector(
                array(
                    1.0 - col("ensemble_probability"),
                    col("ensemble_probability")
                )
            )
        )

        evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="ensemble_raw_prediction"
        )

        roc_auc = evaluator.evaluate(
            test_ensemble,
            {evaluator.metricName: "areaUnderROC"}
        )

        pr_auc = evaluator.evaluate(
            test_ensemble,
            {evaluator.metricName: "areaUnderPR"}
        )

        print("\n========== FINAL ENSEMBLE TEST RESULTS ==========")
        print("Threshold:", ensemble_threshold)
        print("TP:", tp, "FP:", fp, "FN:", fn, "TN:", tn)
        print(f"Precision: {precision:.4%}")
        print(f"Recall: {recall:.4%}")
        print(f"F1-score: {f1:.4%}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")

        # ==========================================
        # SAVE FINAL RISK SCORES
        # ==========================================

        test_ensemble = test_ensemble.withColumn(
            "risk_score",
            col("ensemble_probability") * 100
        ).withColumn(
            "risk_category",
            when(col("risk_score") < 30, "Low")
            .when(col("risk_score") < 60, "Medium")
            .when(col("risk_score") < 80, "High")
            .otherwise("Critical")
        )

        test_ensemble.select(
            "Time",
            "Amount",
            "risk_score",
            "risk_category",
            "label",
            "ensemble_prediction"
        ).toPandas().to_csv(
            "ensemble_risk_predictions.csv",
            index=False
        )

        print("\nEnsemble predictions saved to ensemble_risk_predictions.csv")


    finally:
        spark.stop()


if __name__ == "__main__":
    main()
