import logging
import joblib
from typing import Tuple
import pandas as pd
from scipy.sparse import csr_matrix
from src.vectorization import TfidfVectorization
from zenml import step

@step
def vectorization_step(
    X_train:pd.DataFrame, X_test:pd.DataFrame
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Transforms the training and testing text data using TF-IDF vectorization.

    Parameters:
        X_train (pd.Series): Training data features as text.
        X_test (pd.Series): Testing data features as text.

    Returns:
        Tuple[csr_matrix, csr_matrix]: TF-IDF transformed training and testing data.
    """
    logging.info("Started TF-IDF vectorization step")
    vectorizer = TfidfVectorization()
    tf_X_train = vectorizer.fit_transform(X_train["review_text"])
    tf_X_test = vectorizer.transform(X_test["review_text"])

    # ✅ Save vectorizer to file (for Streamlit or inference later)
    joblib.dump(vectorizer, "vectorizer.pkl")  # you can also use a full path like "artifacts/vectorizer.pkl"
    logging.info("Vectorizer saved to vectorizer.pkl")
    return tf_X_train, tf_X_test
    