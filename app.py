from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover - optional dependency
    LGBMClassifier = None

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
RAW_DATASET = DATA_DIR / "tiktok_accounts.csv"

APP_TITLE = "Hybrid XGBoost and LightGBM Driven Model for Classifying AI-Generated vs. Real Human Accounts on TikTok"

MODEL_FEATURES = [
    "diggCount",
    "followerCount",
    "followingCount",
    "heartCount",
    "videoCount",
    "downloadSetting",
    "duetSetting",
    "openFavorite",
    "stitchSetting",
    "verified",
    "signatureLength",
    "nicknameLength",
    "nicknameNumSpecialCharacters",
    "uniqueIdNumDigits",
    "uniqueIdLength",
]


@dataclass(frozen=True)
class ModelBundle:
    pipeline: Pipeline
    metrics: dict[str, float]
    engine_label: str
    dataset_rows: int


def extract_tiktok_username(profile_link: str) -> str:
    text = profile_link.strip()
    if not text:
        return ""

    match = re.search(r"tiktok\.com/@([^/?#\s]+)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip("@")

    if text.startswith("@"):
        return text[1:].split("/")[0].strip()

    return text.rstrip("/").split("/")[-1].strip("@")


def count_digits(value: str) -> int:
    return min(len(re.findall(r"\d", value or "")), 3)


def count_special_characters(value: str) -> int:
    count = len(re.findall(r"[^a-zA-Z\s]", value or ""))
    return min(count, 3)


def normalize_raw_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["signature"] = out["signature"].fillna("").replace("No bio yet", "")
    out["nickname"] = out["nickname"].fillna("")
    out["uniqueId"] = out["uniqueId"].fillna("")
    out["signatureLength"] = out["signature"].str.len()
    out["nicknameLength"] = out["nickname"].str.len()
    out["nicknameNumSpecialCharacters"] = out["nickname"].apply(count_special_characters)
    out["uniqueIdNumDigits"] = out["uniqueId"].apply(count_digits)
    out["uniqueIdLength"] = out["uniqueId"].str.len()
    out["openFavorite"] = out["openFavorite"].astype(bool).astype(int)
    out["verified"] = out["verified"].astype(bool).astype(int)
    out["fake"] = out["fake"].astype(bool).astype(int)

    for col in ["downloadSetting", "duetSetting", "stitchSetting"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out


def make_hybrid_classifier() -> tuple[VotingClassifier, str]:
    if XGBClassifier is not None and LGBMClassifier is not None:
        xgb = XGBClassifier(
            n_estimators=220,
            max_depth=4,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        lgbm = LGBMClassifier(
            n_estimators=220,
            learning_rate=0.06,
            num_leaves=31,
            random_state=42,
        )
        return (
            VotingClassifier(
                estimators=[("xgboost", xgb), ("lightgbm", lgbm)],
                voting="soft",
                weights=[0.52, 0.48],
            ),
            "Hybrid XGBoost + LightGBM",
        )

    return (
        VotingClassifier(
            estimators=[
                ("gradient_boosting", GradientBoostingClassifier(random_state=42)),
                ("hist_gradient_boosting", HistGradientBoostingClassifier(random_state=42)),
            ],
            voting="soft",
            weights=[0.5, 0.5],
        ),
        "Fallback sklearn hybrid gradient boosting",
    )


@st.cache_data
def load_raw_data() -> pd.DataFrame:
    return normalize_raw_dataset(pd.read_csv(RAW_DATASET))


@st.cache_resource
def train_model() -> ModelBundle:
    df = load_raw_data()
    x = df[MODEL_FEATURES]
    y = df["fake"]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    classifier, engine_label = make_hybrid_classifier()
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }
    return ModelBundle(pipeline=pipeline, metrics=metrics, engine_label=engine_label, dataset_rows=len(df))


def find_profile_row(df: pd.DataFrame, username: str) -> pd.Series | None:
    if not username:
        return None
    matches = df[df["uniqueId"].str.lower() == username.lower()]
    if matches.empty:
        return None
    return matches.iloc[0]


def build_manual_features(username: str) -> pd.DataFrame:
    with st.form("manual_profile_features"):
        st.caption("TikTok does not expose these model features from a URL alone in this local repo, so enter the visible profile metrics here.")
        col1, col2, col3 = st.columns(3)
        digg_count = col1.number_input("Digg count", min_value=0, value=0, step=1)
        follower_count = col2.number_input("Follower count", min_value=0, value=0, step=1)
        following_count = col3.number_input("Following count", min_value=0, value=0, step=1)

        col4, col5, col6 = st.columns(3)
        heart_count = col4.number_input("Heart count", min_value=0, value=0, step=1)
        video_count = col5.number_input("Video count", min_value=0, value=0, step=1)
        verified = col6.checkbox("Verified account")

        nickname = st.text_input("Display name / nickname", value=username or "")
        signature = st.text_area("Bio / signature", value="", height=90)

        col7, col8, col9 = st.columns(3)
        open_favorite = col7.checkbox("Favorites are open")
        download_setting = col8.selectbox("Download setting", [0, 1, 2, 3], index=0)
        duet_setting = col9.selectbox("Duet setting", [0, 1, 2, 3], index=0)
        stitch_setting = st.selectbox("Stitch setting", [0, 1, 2, 3], index=0)

        submitted = st.form_submit_button("Classify Account", type="primary")

    if not submitted:
        return pd.DataFrame()

    features = {
        "diggCount": digg_count,
        "followerCount": follower_count,
        "followingCount": following_count,
        "heartCount": heart_count,
        "videoCount": video_count,
        "downloadSetting": download_setting,
        "duetSetting": duet_setting,
        "openFavorite": int(open_favorite),
        "stitchSetting": stitch_setting,
        "verified": int(verified),
        "signatureLength": len(signature.replace("No bio yet", "")),
        "nicknameLength": len(nickname),
        "nicknameNumSpecialCharacters": count_special_characters(nickname),
        "uniqueIdNumDigits": count_digits(username),
        "uniqueIdLength": len(username),
    }
    return pd.DataFrame([features], columns=MODEL_FEATURES)


def predict_account(bundle: ModelBundle, features: pd.DataFrame) -> tuple[int, float]:
    prediction = int(bundle.pipeline.predict(features)[0])
    probability = float(bundle.pipeline.predict_proba(features)[0][prediction])
    return prediction, probability


def render_prediction(prediction: int, probability: float) -> None:
    if prediction == 1:
        st.error(f"Predicted class: AI-generated / fake account ({probability:.1%} confidence)")
    else:
        st.success(f"Predicted class: real human account ({probability:.1%} confidence)")


def main() -> None:
    st.set_page_config(page_title="TikTok Account Classifier", page_icon="T", layout="wide")
    st.title(APP_TITLE)

    df = load_raw_data()
    bundle = train_model()

    metric_cols = st.columns(5)
    metric_cols[0].metric("Dataset rows", f"{bundle.dataset_rows:,}")
    metric_cols[1].metric("Model engine", bundle.engine_label)
    metric_cols[2].metric("Accuracy", f"{bundle.metrics['accuracy']:.3f}")
    metric_cols[3].metric("F1 score", f"{bundle.metrics['f1']:.3f}")
    metric_cols[4].metric("Recall", f"{bundle.metrics['recall']:.3f}")

    st.divider()

    profile_link = st.text_input(
        "Paste TikTok profile link",
        placeholder="https://www.tiktok.com/@username",
    )
    username = extract_tiktok_username(profile_link)

    if username:
        st.caption(f"Parsed username: @{username}")
        row = find_profile_row(df, username)
        if row is not None:
            st.info("This username exists in the local dataset, so the app used its stored profile features.")
            features = pd.DataFrame([row[MODEL_FEATURES].to_dict()], columns=MODEL_FEATURES)
            prediction, probability = predict_account(bundle, features)
            render_prediction(prediction, probability)

            with st.expander("Profile features used"):
                st.dataframe(features, use_container_width=True)
        else:
            st.warning("This username is not in the local dataset. Enter the profile metrics below to classify it.")
            features = build_manual_features(username)
            if not features.empty:
                prediction, probability = predict_account(bundle, features)
                render_prediction(prediction, probability)

                with st.expander("Profile features used"):
                    st.dataframe(features, use_container_width=True)
    else:
        st.info("Paste a TikTok profile link to begin.")

    st.divider()
    with st.expander("Dataset preview"):
        st.dataframe(df.head(30), use_container_width=True)


if __name__ == "__main__":
    main()
