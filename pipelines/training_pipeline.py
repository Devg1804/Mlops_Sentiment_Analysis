
from steps.data_ingestion_step import ingest_data
from steps.data_preprocessing_step import data_preprocessing_step
from steps.data_sampling_step import data_sampling_step
from steps.data_splitter_step import data_splitter_step
from steps.vectorization_step import vectorization_step
from steps.model_building_step import model_building_step
from steps.model_evaluation_step import model_evaluation_step
from steps.collect_models_step import collect_models_step
from steps.model_selector_step import model_selector_step
from zenml import Model, pipeline

@pipeline(
    model=Model(
        # The name uniquely identifies this model
        name="customer_reviews_predictor"
    ),
)
def ml_pipeline():

    """Define an end-to-end machine learning pipeline."""

    # Data Ingestion Step
    raw_data = ingest_data("customer_reviews")

    df_preprocessed = data_preprocessing_step(raw_data)
    
    df_sampled = data_sampling_step(df_preprocessed)
    
    X_train, X_test, y_train, y_test = data_splitter_step(df_sampled, target_column="label")

    tf_X_train, tf_X_test = vectorization_step(X_train, X_test)


    # Train models
    svm_model = model_building_step(X_train=tf_X_train, y_train=y_train, method="svc",fine_tuning=False)
    logistic_model = model_building_step(X_train=tf_X_train, y_train=y_train, method="logistic_regression")
    rf_model = model_building_step(X_train=tf_X_train, y_train=y_train, method="random_forest")
    nb_model = model_building_step(X_train=tf_X_train, y_train=y_train, method="naive_bayes")
    xg_model = model_building_step(X_train=tf_X_train, y_train=y_train, method="xgboost")

    # Evaluate models (return model + metrics)
    svm_eval = model_evaluation_step(model=svm_model, X_test=tf_X_test, y_test=y_test)
    logistic_eval = model_evaluation_step(model=logistic_model, X_test=tf_X_test, y_test=y_test)
    rf_eval = model_evaluation_step(model=rf_model, X_test=tf_X_test, y_test=y_test)
    nb_eval = model_evaluation_step(model=nb_model, X_test=tf_X_test, y_test=y_test)
    xg_eval = model_evaluation_step(model=xg_model, X_test=tf_X_test, y_test=y_test)

    # ✅ Collect all evaluation results
    model_results = collect_models_step(
        svm_eval=svm_eval,
        logistic_eval=logistic_eval,
        rf_eval=rf_eval,
        nb_eval=nb_eval,
        xg_eval=xg_eval,
    )

    # ✅ Pass that output to model selector step
    best_metrics = model_selector_step(
        model_results=model_results,
        primary_metric="F1 Score"
    )

    return best_metrics

if __name__=="__main__":
    run = ml_pipeline()