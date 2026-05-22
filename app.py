from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
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

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


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


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def find_user_info(payload: object) -> dict[str, object] | None:
    if isinstance(payload, dict):
        user_info = payload.get("userInfo")
        if isinstance(user_info, dict) and "user" in user_info and "stats" in user_info:
            return user_info

        user_module = payload.get("UserModule")
        if isinstance(user_module, dict):
            users = user_module.get("users")
            stats = user_module.get("stats")
            if isinstance(users, dict) and isinstance(stats, dict) and users:
                unique_id, user = next(iter(users.items()))
                return {
                    "user": user,
                    "stats": stats.get(unique_id, {}),
                }

        for value in payload.values():
            found = find_user_info(value)
            if found is not None:
                return found

    if isinstance(payload, list):
        for value in payload:
            found = find_user_info(value)
            if found is not None:
                return found

    return None


def extract_embedded_json(page_html: str) -> list[dict[str, object]]:
    payloads = []
    script_patterns = [
        r'<script[^>]+id="__UNIVERSAL_DATA_FOR_REHYDRATION__"[^>]*>(.*?)</script>',
        r'<script[^>]+id="SIGI_STATE"[^>]*>(.*?)</script>',
    ]
    for pattern in script_patterns:
        match = re.search(pattern, page_html, flags=re.DOTALL)
        if not match:
            continue
        raw_json = html.unescape(match.group(1)).strip()
        try:
            payloads.append(json.loads(raw_json))
        except json.JSONDecodeError:
            continue
    return payloads


@st.cache_data(ttl=600, show_spinner=False)
def fetch_profile_features(profile_link: str, username: str) -> tuple[pd.DataFrame, dict[str, object]]:
    url = f"https://www.tiktok.com/@{username}"
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()

    for payload in extract_embedded_json(response.text):
        user_info = find_user_info(payload)
        if user_info is None:
            continue

        user = user_info.get("user", {})
        stats = user_info.get("stats", {})
        if not isinstance(user, dict) or not isinstance(stats, dict):
            continue

        unique_id = str(user.get("uniqueId") or username)
        nickname = str(user.get("nickname") or unique_id)
        signature = str(user.get("signature") or "").replace("No bio yet", "")

        features = {
            "diggCount": safe_int(stats.get("diggCount")),
            "followerCount": safe_int(stats.get("followerCount")),
            "followingCount": safe_int(stats.get("followingCount")),
            "heartCount": safe_int(stats.get("heartCount") or stats.get("heart")),
            "videoCount": safe_int(stats.get("videoCount")),
            "downloadSetting": safe_int(user.get("downloadSetting")),
            "duetSetting": safe_int(user.get("duetSetting")),
            "openFavorite": int(bool(user.get("openFavorite", False))),
            "stitchSetting": safe_int(user.get("stitchSetting")),
            "verified": int(bool(user.get("verified", False))),
            "signatureLength": len(signature),
            "nicknameLength": len(nickname),
            "nicknameNumSpecialCharacters": count_special_characters(nickname),
            "uniqueIdNumDigits": count_digits(unique_id),
            "uniqueIdLength": len(unique_id),
        }
        profile = {
            "uniqueId": unique_id,
            "nickname": nickname,
            "signature": signature,
            "profileUrl": profile_link,
        }
        return pd.DataFrame([features], columns=MODEL_FEATURES), profile

    raise ValueError("No public TikTok profile metrics were found in the page data.")


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


def build_manual_features(username: str) -> pd.DataFrame:
    with st.form("manual_profile_features"):
        st.caption("Enter the visible profile metrics, then the trained hybrid model will predict the account class. The app does not search for the username in the training dataset.")
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
        try:
            with st.spinner("Fetching public TikTok profile metrics..."):
                features, profile = fetch_profile_features(profile_link, username)

            prediction, probability = predict_account(bundle, features)
            render_prediction(prediction, probability)

            st.subheader("Fetched Profile Metrics")
            profile_cols = st.columns(3)
            profile_cols[0].metric("Username", f"@{profile['uniqueId']}")
            profile_cols[1].metric("Display name length", int(features["nicknameLength"].iloc[0]))
            profile_cols[2].metric("Bio length", int(features["signatureLength"].iloc[0]))

            with st.expander("Profile features used"):
                st.dataframe(features, use_container_width=True)

        except Exception as exc:
            st.error("The app could not automatically fetch this profile's public metrics.")
            st.caption(f"Fetch error: {exc}")
            st.warning("TikTok sometimes blocks automated page reads or hides profile data. You can still run a model prediction by entering the visible metrics below.")
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
