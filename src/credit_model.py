import pandas as pd
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def run_credit_model():

    engine = create_engine(
        "postgresql+psycopg2://postgres@localhost:5432/capalloc",
        connect_args={"password": os.getenv("DB_PASSWORD")}
    )

    df_loans = pd.read_sql("SELECT * FROM raw.loans", engine)

    df_loans["default"] = df_loans["loan_status"].isin(
        ["Charged Off", "Default", "Late (31-120 days)", "Late (16-30 days)"]
    ).astype(int)

    credit_df = df_loans[[
        "default",
        "loan_amnt",
        "int_rate",
        "annual_inc",
        "dti",
        "installment",
        "revol_bal",
        "revol_util"
    ]].dropna()

    X = credit_df.drop(columns=["default"])
    y = credit_df["default"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_prob)

    credit_df["pd"] = model.predict_proba(X)[:, 1]
    credit_df["expected_return"] = credit_df["loan_amnt"] * credit_df["int_rate"] / 100
    credit_df["expected_loss"] = credit_df["pd"] * credit_df["loan_amnt"]
    credit_df["risk_adjusted_return"] = (
        credit_df["expected_return"] - credit_df["expected_loss"]
    )

    print("Credit AUC:", auc)

    return credit_df
