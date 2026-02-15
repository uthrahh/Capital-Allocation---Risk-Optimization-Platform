import pandas as pd
import os
from sqlalchemy import create_engine
from lifelines import CoxPHFitter


def run_vacancy_model():

    # ---------------- DATABASE CONNECTION ----------------
    engine = create_engine(
        "postgresql+psycopg2://postgres@localhost:5432/capalloc",
        connect_args={"password": os.getenv("DB_PASSWORD")}
    )

    # ---------------- LOAD DATA ----------------
    df = pd.read_sql("SELECT * FROM stg.vacancy_model_base", engine)

    # Ensure correct ordering for survival modeling
    df = df.sort_values(["listing_id", "month"])

    # ---------------- SURVIVAL STRUCTURE ----------------
    df["duration"] = df.groupby("listing_id").cumcount() + 1
    df["event"] = df["fully_vacant_flag"]

    model_df = df[[
        "duration",
        "event",
        "accommodates",
        "bedrooms",
        "beds",
        "price",
        "minimum_nights",
        "maximum_nights",
        "availability_365",
        "number_of_reviews",
        "review_scores_rating",
        "reviews_per_month"
    ]].dropna()

    if len(model_df) == 0:
        raise ValueError("Vacancy model dataset is empty")

    # ---------------- FIT COX MODEL ----------------
    cph = CoxPHFitter()
    cph.fit(model_df, duration_col="duration", event_col="event")

    print("Vacancy C-index:", cph.concordance_index_)

    # ---------------- PREDICT RISK ----------------
    df["predicted_risk"] = cph.predict_partial_hazard(df)

    # ---------------- AGGREGATE TO LISTING LEVEL ----------------
    listing_returns = df.groupby("listing_id").agg({
        "occupancy_rate": "mean",
        "price": "mean",
        "predicted_risk": "mean"
    }).reset_index()

    listing_returns = listing_returns.replace([float("inf"), float("-inf")], pd.NA)
    listing_returns = listing_returns.dropna().reset_index(drop=True)

    # ---------------- EXPECTED REVENUE ----------------
    listing_returns["expected_revenue"] = (
        listing_returns["occupancy_rate"] *
        listing_returns["price"] * 365
    )

    listing_returns["risk_adjusted_return"] = (
        listing_returns["expected_revenue"] *
        (1 - listing_returns["predicted_risk"])
    )

    return listing_returns
