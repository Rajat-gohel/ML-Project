# ============================================================
# LINGO AI - MULTI MODEL LANGUAGE CLASSIFICATION
# ============================================================
#
# Dataset:
#   language.csv
#
# Columns:
#   Headline | Language
#
# Models:
#   1. Logistic Regression
#   2. KNN
#   3. Naive Bayes
#   4. Decision Tree
#   5. Random Forest
#   6. SVM
#   7. Gradient Boosting
#   8. AdaBoost
#   9. XGBoost
#   10. LightGBM
#   11. CatBoost
#
# Features:
#   Character-level TF-IDF
#
# Extra:
#   Unknown / Unsupported language detection
#   Model comparison
#   Confusion matrices
#   Classification reports
#   Saved models
#   Model metadata
#
# ============================================================

import os
import re
import time
import warnings
import joblib

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================
# SKLEARN
# ============================================================

from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import MultinomialNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# BOOSTING LIBRARIES
# ============================================================

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "language.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "model_results"
)

MODELS_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

CONFUSION_DIR = os.path.join(
    OUTPUT_DIR,
    "confusion_matrices"
)

TEST_SIZE = 0.20

RANDOM_STATE = 42

MAX_FEATURES = 20000

UNKNOWN_THRESHOLD = 0.55


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    CONFUSION_DIR,
    exist_ok=True
)


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    text = str(text)

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    # Keep Unicode letters/numbers
    # Very important for multilingual detection.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# HEADER
# ============================================================

print("\n")
print("=" * 75)
print("LINGO AI - MULTI LANGUAGE DETECTION")
print("=" * 75)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}\n"
        "\nMake sure language.csv is in the same folder as model.py."
    )

df = pd.read_csv(
    DATASET_PATH
)

print("\nDataset columns:")
print(
    df.columns.tolist()
)


# ============================================================
# FIND COLUMNS
# ============================================================

headline_column = None

language_column = None


for column in df.columns:

    normalized = str(column).strip().lower()

    if normalized in [
        "headline",
        "text",
        "sentence",
        "content"
    ]:

        headline_column = column

    if normalized in [
        "language",
        "lang",
        "label",
        "target"
    ]:

        language_column = column


if headline_column is None:

    raise ValueError(
        "\nCould not find text column.\n"
        "Expected: Headline"
    )


if language_column is None:

    raise ValueError(
        "\nCould not find language column.\n"
        "Expected: Language"
    )


print(
    f"\nText column     : {headline_column}"
)

print(
    f"Language column : {language_column}"
)


# ============================================================
# KEEP REQUIRED COLUMNS
# ============================================================

df = df[
    [
        headline_column,
        language_column
    ]
].copy()


df.columns = [
    "Headline",
    "Language"
]


# ============================================================
# REMOVE NULLS
# ============================================================

before = len(df)

df = df.dropna(
    subset=[
        "Headline",
        "Language"
    ]
)

print(
    f"\nRemoved null rows: {before - len(df)}"
)


# ============================================================
# CLEAN TEXT
# ============================================================

print(
    "\nCleaning text..."
)

df["Headline"] = (
    df["Headline"]
    .astype(str)
    .apply(clean_text)
)


df["Language"] = (
    df["Language"]
    .astype(str)
    .str.strip()
)


# ============================================================
# REMOVE EMPTY TEXT
# ============================================================

df = df[
    df["Headline"].str.len() > 0
]


df = df[
    df["Language"].str.len() > 0
]


# ============================================================
# REMOVE DUPLICATES
# ============================================================

before = len(df)

df = df.drop_duplicates(
    subset=[
        "Headline",
        "Language"
    ]
)

print(
    f"Removed duplicates: {before - len(df)}"
)


# ============================================================
# REMOVE VERY RARE LANGUAGES
# ============================================================
#
# IMPORTANT:
#
# If a language has only 1 or 2 examples, the model cannot
# properly learn that language.
#
# We keep languages having at least 2 samples so that
# stratified train/test split can work.
#
# ============================================================

language_counts = (
    df["Language"]
    .value_counts()
)


valid_languages = language_counts[
    language_counts >= 2
].index


removed_languages = language_counts[
    language_counts < 2
]


if len(removed_languages) > 0:

    print("\nLanguages with less than 2 samples:")

    print(
        removed_languages
    )

    df = df[
        df["Language"].isin(
            valid_languages
        )
    ]


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n")
print("-" * 75)
print("DATASET INFORMATION")
print("-" * 75)

print(
    f"Total samples       : {len(df):,}"
)

print(
    f"Number of languages : {df['Language'].nunique()}"
)

print(
    f"Average text length : "
    f"{df['Headline'].str.len().mean():.2f}"
)

print(
    f"Minimum text length : "
    f"{df['Headline'].str.len().min()}"
)

print(
    f"Maximum text length : "
    f"{df['Headline'].str.len().max()}"
)


print("\nLanguage distribution:")

print(
    df["Language"].value_counts()
)


# ============================================================
# SAVE CLEAN DATASET
# ============================================================

clean_dataset_path = os.path.join(
    OUTPUT_DIR,
    "cleaned_dataset.csv"
)

df.to_csv(
    clean_dataset_path,
    index=False
)


# ============================================================
# INPUT
# ============================================================

X = df["Headline"]

y = df["Language"]


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

print("\n")
print("Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y
)


print(
    f"Training samples : {len(X_train):,}"
)

print(
    f"Testing samples  : {len(X_test):,}"
)


# ============================================================
# TF-IDF
# ============================================================

print("\n")
print("Creating character-level TF-IDF...")


vectorizer = TfidfVectorizer(

    analyzer="char",

    ngram_range=(2, 5),

    min_df=2,

    max_features=MAX_FEATURES,

    sublinear_tf=True,

    lowercase=False
)


# ============================================================
# FIT TF-IDF ONLY ON TRAINING DATA
# ============================================================

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    f"TF-IDF training shape : "
    f"{X_train_tfidf.shape}"
)

print(
    f"TF-IDF testing shape  : "
    f"{X_test_tfidf.shape}"
)


# ============================================================
# SAVE VECTORIZER
# ============================================================

joblib.dump(

    vectorizer,

    os.path.join(
        OUTPUT_DIR,
        "tfidf_vectorizer.pkl"
    )
)


# ============================================================
# LABEL ENCODER
# ============================================================

print(
    "\nEncoding language labels..."
)


label_encoder = LabelEncoder()


y_train_encoded = (
    label_encoder.fit_transform(
        y_train
    )
)


y_test_encoded = (
    label_encoder.transform(
        y_test
    )
)


joblib.dump(

    label_encoder,

    os.path.join(
        OUTPUT_DIR,
        "label_encoder.pkl"
    )
)


print(
    f"Classes: {list(label_encoder.classes_)}"
)


# ============================================================
# NUMBER OF CLASSES
# ============================================================

NUMBER_OF_CLASSES = len(
    label_encoder.classes_
)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

models = {

    # ========================================================
    # 1. LOGISTIC REGRESSION
    # ========================================================

    "Logistic Regression":

        LogisticRegression(

            max_iter=3000,

            C=5,

            solver="liblinear",

            random_state=RANDOM_STATE
        ),


    # ========================================================
    # 2. KNN
    # ========================================================

    "KNN":

        KNeighborsClassifier(

            n_neighbors=5,

            weights="distance",

            metric="cosine",

            n_jobs=-1
        ),


    # ========================================================
    # 3. NAIVE BAYES
    # ========================================================

    "Naive Bayes":

        MultinomialNB(

            alpha=0.1
        ),


    # ========================================================
    # 4. DECISION TREE
    # ========================================================

    "Decision Tree":

        DecisionTreeClassifier(

            max_depth=100,

            min_samples_split=2,

            random_state=RANDOM_STATE
        ),


    # ========================================================
    # 5. RANDOM FOREST
    # ========================================================

    "Random Forest":

        RandomForestClassifier(

            n_estimators=100,

            max_depth=20,

            min_samples_split=2,

            n_jobs=-1,

            random_state=RANDOM_STATE
        ),


    # ========================================================
    # 6. SVM
    # ========================================================

    "SVM":

        LinearSVC(

            C=2,

            max_iter=5000,

            random_state=RANDOM_STATE
        ),


    # ========================================================
    # 7. GRADIENT BOOSTING
    # ========================================================

    # "Gradient Boosting":

    #     GradientBoostingClassifier(

    #         n_estimators=100,

    #         learning_rate=0.1,

    #         max_depth=3,

    #         random_state=RANDOM_STATE
    #     ),


    # ========================================================
    # 8. ADABOOST
    # ========================================================

    "AdaBoost":

        AdaBoostClassifier(

            n_estimators=100,

            learning_rate=0.5,

            random_state=RANDOM_STATE
        ),


    # ========================================================
    # 9. XGBOOST
    # ========================================================

    # "XGBoost":

    #     XGBClassifier(

    #         n_estimators=150,

    #         max_depth=6,

    #         learning_rate=0.1,

    #         subsample=0.8,

    #         colsample_bytree=0.8,

    #         objective="multi:softprob",

    #         num_class=NUMBER_OF_CLASSES,

    #         eval_metric="mlogloss",

    #         tree_method="hist",

    #         n_jobs=-1,

    #         random_state=RANDOM_STATE
    #     ),


    # ========================================================
    # 10. LIGHTGBM
    # ========================================================

    # "LightGBM":

    #     LGBMClassifier(

    #         n_estimators=150,

    #         learning_rate=0.1,

    #         num_leaves=31,

    #         objective="multiclass",

    #         n_jobs=-1,

    #         verbosity=-1,

    #         random_state=RANDOM_STATE
    #     ),


    # ========================================================
    # 11. CATBOOST
    # ========================================================

    # "CatBoost":

    #     CatBoostClassifier(

    #         iterations=150,

    #         depth=6,

    #         learning_rate=0.1,

    #         loss_function="MultiClass",

    #         verbose=False,

    #         random_seed=RANDOM_STATE
    #     )

}


# ============================================================
# RESULTS
# ============================================================

results = []

trained_models = {}


# ============================================================
# TRAINING
# ============================================================

print("\n")
print("=" * 75)
print("TRAINING ALL 11 MODELS")
print("=" * 75)


for model_name, model in models.items():

    print("\n")
    print("-" * 75)

    print(
        f"Training: {model_name}"
    )

    print("-" * 75)

    start_time = time.time()

    try:

        # ====================================================
        # BOOSTING MODELS
        # ====================================================

        if model_name in [
            "XGBoost",
            "LightGBM",
            "CatBoost"
        ]:

            model.fit(

                X_train_tfidf,

                y_train_encoded
            )


            raw_predictions = (
                model.predict(
                    X_test_tfidf
                )
            )


            raw_predictions = (
                np.asarray(
                    raw_predictions
                ).reshape(-1)
            )


            predictions = (
                label_encoder.inverse_transform(
                    raw_predictions.astype(int)
                )
            )


        # ====================================================
        # NORMAL MODELS
        # ====================================================

        else:

            model.fit(

                X_train_tfidf,

                y_train
            )


            predictions = (
                model.predict(
                    X_test_tfidf
                )
            )


        # ====================================================
        # METRICS
        # ====================================================

        accuracy = accuracy_score(

            y_test,

            predictions
        )


        precision = precision_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0
        )


        recall = recall_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0
        )


        f1 = f1_score(

            y_test,

            predictions,

            average="weighted",

            zero_division=0
        )


        training_time = (
            time.time() - start_time
        )


        # ====================================================
        # SAVE MODEL IN MEMORY
        # ====================================================

        trained_models[
            model_name
        ] = model


        # ====================================================
        # SAFE MODEL NAME
        # ====================================================

        safe_name = (

            model_name

            .lower()

            .replace(" ", "_")

            .replace("-", "_")
        )


        # ====================================================
        # SAVE MODEL
        # ====================================================

        model_path = os.path.join(

            MODELS_DIR,

            f"{safe_name}.pkl"
        )


        joblib.dump(

            model,

            model_path
        )


        # ====================================================
        # RESULTS
        # ====================================================

        results.append({

            "Model": model_name,

            "Accuracy": accuracy,

            "Precision": precision,

            "Recall": recall,

            "F1 Score": f1,

            "Training Time (sec)":
                training_time
        })


        # ====================================================
        # PRINT METRICS
        # ====================================================

        print(
            f"Accuracy  : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            f"Precision : "
            f"{precision * 100:.2f}%"
        )

        print(
            f"Recall    : "
            f"{recall * 100:.2f}%"
        )

        print(
            f"F1 Score  : "
            f"{f1 * 100:.2f}%"
        )

        print(
            f"Time      : "
            f"{training_time:.2f} sec"
        )


        # ====================================================
        # CLASSIFICATION REPORT
        # ====================================================

        report = classification_report(

            y_test,

            predictions,

            zero_division=0
        )


        report_path = os.path.join(

            OUTPUT_DIR,

            f"{safe_name}_classification_report.txt"
        )


        with open(

            report_path,

            "w",

            encoding="utf-8"

        ) as file:

            file.write(report)


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        labels = sorted(
            y_test.unique()
        )


        cm = confusion_matrix(

            y_test,

            predictions,

            labels=labels
        )


        plt.figure(

            figsize=(12, 10)
        )


        sns.heatmap(

            cm,

            annot=True,

            fmt="d",

            cmap="Blues",

            xticklabels=labels,

            yticklabels=labels
        )


        plt.title(
            f"{model_name} - Confusion Matrix"
        )


        plt.xlabel(
            "Predicted Language"
        )


        plt.ylabel(
            "Actual Language"
        )


        plt.xticks(

            rotation=45,

            ha="right"
        )


        plt.tight_layout()


        matrix_path = os.path.join(

            CONFUSION_DIR,

            f"{safe_name}.png"
        )


        plt.savefig(

            matrix_path,

            dpi=200,

            bbox_inches="tight"
        )


        plt.close()


    except Exception as error:

        print(
            f"\nERROR in {model_name}"
        )

        print(
            str(error)
        )


# ============================================================
# CHECK RESULTS
# ============================================================

if len(results) == 0:

    raise RuntimeError(
        "\nNo model trained successfully."
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# SORT BY ACCURACY
# ============================================================

results_df = (
    results_df
    .sort_values(
        by="Accuracy",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 75)
print("FINAL MODEL COMPARISON")
print("=" * 75)


display_df = results_df.copy()


display_df["Accuracy"] = (
    display_df["Accuracy"] * 100
)


display_df["Precision"] = (
    display_df["Precision"] * 100
)


display_df["Recall"] = (
    display_df["Recall"] * 100
)


display_df["F1 Score"] = (
    display_df["F1 Score"] * 100
)


print(
    display_df.to_string(
        index=False,
        formatters={

            "Accuracy":
                "{:.2f}%".format,

            "Precision":
                "{:.2f}%".format,

            "Recall":
                "{:.2f}%".format,

            "F1 Score":
                "{:.2f}%".format,

            "Training Time (sec)":
                "{:.2f}".format
        }
    )
)


# ============================================================
# SAVE COMPARISON
# ============================================================

comparison_path = os.path.join(

    OUTPUT_DIR,

    "model_comparison.csv"
)


results_df.to_csv(

    comparison_path,

    index=False
)


# ============================================================
# BEST MODEL
# ============================================================

best_model_name = (
    results_df.iloc[0]["Model"]
)


best_accuracy = (
    results_df.iloc[0]["Accuracy"]
)


best_model = (
    trained_models[
        best_model_name
    ]
)


print("\n")
print(
    f"Best Model: "
    f"{best_model_name}"
)


print(
    f"Best Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

joblib.dump(

    best_model,

    os.path.join(
        OUTPUT_DIR,
        "best_model.pkl"
    )
)


# ============================================================
# ============================================================
# UNKNOWN LANGUAGE DETECTION
# ============================================================
# ============================================================

print("\n")
print("=" * 75)
print("CREATING UNKNOWN LANGUAGE DETECTION CONFIGURATION")
print("=" * 75)


# ============================================================
# MODEL CONFIDENCE SUPPORT
# ============================================================

confidence_support = {}


for model_name, model in trained_models.items():

    if hasattr(
        model,
        "predict_proba"
    ):

        confidence_support[
            model_name
        ] = True

    elif hasattr(
        model,
        "decision_function"
    ):

        confidence_support[
            model_name
        ] = "decision_function"

    else:

        confidence_support[
            model_name
        ] = False


# ============================================================
# SAVE UNKNOWN CONFIG
# ============================================================

unknown_config = {

    "unknown_threshold":
        UNKNOWN_THRESHOLD,

    "known_languages":
        list(
            label_encoder.classes_
        ),

    "confidence_support":
        confidence_support,

    "note":
        "Predictions below the configured confidence "
        "threshold can be treated as Unknown/Unsupported. "
        "Threshold is configurable in the Streamlit app."
}


joblib.dump(

    unknown_config,

    os.path.join(
        OUTPUT_DIR,
        "unknown_language_config.pkl"
    )
)


# ============================================================
# MODEL INFORMATION
# ============================================================

model_info = {

    "best_model":
        best_model_name,

    "accuracy":
        float(best_accuracy),

    "accuracy_percent":
        float(
            best_accuracy * 100
        ),

    "test_size":
        TEST_SIZE,

    "random_state":
        RANDOM_STATE,

    "tfidf_analyzer":
        "char",

    "tfidf_ngram_range":
        (2, 5),

    "tfidf_max_features":
        MAX_FEATURES,

    "number_of_languages":
        int(
            df["Language"].nunique()
        ),

    "number_of_samples":
        int(
            len(df)
        ),

    "languages":
        list(
            label_encoder.classes_
        ),

    "number_of_models":
        len(
            trained_models
        ),

    "unknown_language_detection":
        True,

    "unknown_threshold":
        UNKNOWN_THRESHOLD
}


joblib.dump(

    model_info,

    os.path.join(
        OUTPUT_DIR,
        "model_info.pkl"
    )
)


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(14, 8)
)


plt.bar(

    display_df["Model"],

    display_df["Accuracy"]
)


plt.title(
    "Lingo AI - Model Accuracy Comparison"
)


plt.xlabel(
    "Machine Learning Algorithm"
)


plt.ylabel(
    "Accuracy (%)"
)


plt.xticks(

    rotation=45,

    ha="right"
)


plt.ylim(

    max(
        0,
        display_df["Accuracy"].min() - 5
    ),

    100
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        OUTPUT_DIR,
        "model_accuracy_comparison.png"
    ),

    dpi=200,

    bbox_inches="tight"
)


plt.close()


# ============================================================
# ALL METRICS GRAPH
# ============================================================

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1 Score"
]


for metric in metrics:

    plt.figure(
        figsize=(14, 7)
    )


    plt.bar(

        display_df["Model"],

        display_df[metric]
    )


    plt.title(
        f"{metric} Comparison"
    )


    plt.xlabel(
        "Machine Learning Algorithm"
    )


    plt.ylabel(
        f"{metric} (%)"
    )


    plt.xticks(

        rotation=45,

        ha="right"
    )


    plt.tight_layout()


    file_name = (

        metric

        .lower()

        .replace(
            " ",
            "_"
        )

        + "_comparison.png"
    )


    plt.savefig(

        os.path.join(
            OUTPUT_DIR,
            file_name
        ),

        dpi=200,

        bbox_inches="tight"
    )


    plt.close()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 75)
print("LINGO AI TRAINING COMPLETED")
print("=" * 75)


print(
    f"\nModels trained successfully : "
    f"{len(trained_models)} / 11"
)


print(
    f"Languages detected in dataset : "
    f"{df['Language'].nunique()}"
)


print(
    f"Dataset samples : "
    f"{len(df):,}"
)


print(
    f"Best model : "
    f"{best_model_name}"
)


print(
    f"Best accuracy : "
    f"{best_accuracy * 100:.2f}%"
)


print("\nKnown languages:")

for language in label_encoder.classes_:

    print(
        f"  ✓ {language}"
    )


print("\n")
print("Files generated:")

print(
    "  ✓ tfidf_vectorizer.pkl"
)

print(
    "  ✓ label_encoder.pkl"
)

print(
    "  ✓ best_model.pkl"
)

print(
    "  ✓ model_info.pkl"
)

print(
    "  ✓ unknown_language_config.pkl"
)

print(
    "  ✓ model_comparison.csv"
)

print(
    "  ✓ cleaned_dataset.csv"
)

print(
    "  ✓ models/"
)

print(
    "  ✓ confusion_matrices/"
)

print("\n")
print("=" * 75)
print("DONE")
print("=" * 75)