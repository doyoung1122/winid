"""
vfs_data_csv → 발화원인(cause_main) 다중 분류 모델 학습 및 비교

실행: python ml/train.py
옵션:
  --skip-rf    RandomForest 제외 (느릴 때)
  --quick      샘플 20K건만 사용 (빠른 테스트)
"""

import os
import sys
import time
import warnings
import argparse
import joblib
import numpy as np
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
load_dotenv()

# ================================================================
# 설정
# ================================================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

FEATURES = [
    "fire_type",
    "building_type",
    "building_structure",
    "location_main",
    "location_mid",
    "location_sub",
    "region",
    "temperature",
    "wind_speed",
    "humidity",
]

CATEGORICAL_FEATURES = [
    "fire_type",
    "building_type",
    "building_structure",
    "location_main",
    "location_mid",
    "location_sub",
    "region",
    "wind_speed",
]

TARGET = "cause_main"
EXCLUDE_TARGETS = ["미상"]

# ================================================================
# 데이터 로드
# ================================================================
def load_data():
    print("=" * 60)
    print("📥  MySQL → vfs_data_csv 로드")
    print("=" * 60)

    conn = mysql.connector.connect(
        host=os.getenv("MY_HOST", "218.158.62.155"),
        port=int(os.getenv("MY_PORT", 33307)),
        user=os.getenv("MY_USER", "vfs"),
        password=os.getenv("MY_PASS"),
        database=os.getenv("MY_DB", "vfs"),
        charset="utf8mb4",
    )

    cols = ", ".join(FEATURES + [TARGET])
    placeholders = ", ".join(["%s"] * len(EXCLUDE_TARGETS))
    query = f"SELECT {cols} FROM vfs_data_csv WHERE {TARGET} NOT IN ({placeholders})"
    df = pd.read_sql(query, conn, params=EXCLUDE_TARGETS)
    conn.close()

    # 빈 문자열 → NaN
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].replace("", np.nan)

    print(f"✅  로드 완료: {len(df):,}건\n")
    return df


# ================================================================
# 전처리 A: 레이블 인코딩 (LightGBM / XGBoost / RandomForest용)
# ================================================================
def encode_for_tree(df, label_encoders=None, fit=True):
    df = df.copy()

    if label_encoders is None:
        label_encoders = {}

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            le.fit(df[col].dropna().astype(str))
            label_encoders[col] = le

        le = label_encoders[col]
        df[col] = df[col].apply(
            lambda v: float(le.transform([str(v)])[0])
            if pd.notna(v) and str(v) in le.classes_
            else np.nan
        )

    if "temperature" in df.columns:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").clip(-40, 50)
    if "humidity" in df.columns:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce").clip(0, 100)

    return df, label_encoders


# ================================================================
# 전처리 B: CatBoost용 (원본 문자열 유지, NaN → None)
# ================================================================
def prepare_for_catboost(df):
    df = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # NaN → None (CatBoost가 None을 결측값으로 처리)
            df[col] = df[col].where(df[col].notna(), None)
    if "temperature" in df.columns:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").clip(-40, 50)
    if "humidity" in df.columns:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce").clip(0, 100)
    return df


# ================================================================
# 모델 정의
# ================================================================
def get_models(n_classes, cat_indices, skip_rf=False):
    models = {}

    models["LightGBM"] = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=63,
        objective="multiclass",
        num_class=n_classes,
        class_weight="balanced",
        categorical_feature=cat_indices,
        min_child_samples=20,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    # CatBoost는 cat_indices 별도 전달 (Pool 사용)
    models["CatBoost"] = CatBoostClassifier(
        iterations=500,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
    )

    models["XGBoost"] = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
        verbosity=0,
    )

    if not skip_rf:
        models["RandomForest"] = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )

    return models


# ================================================================
# 학습 + 평가
# ================================================================
def train_and_evaluate(
    models,
    X_train_enc, X_test_enc,   # 인코딩 버전 (LightGBM/XGBoost/RF)
    X_train_cat, X_test_cat,   # 원본 DataFrame (CatBoost)
    y_train, y_test,
    cat_indices,
):
    results = []

    for name, model in models.items():
        print(f"\n🔄  [{name}] 학습 중...")
        t0 = time.time()

        if name == "CatBoost":
            train_pool = Pool(X_train_cat, y_train, cat_features=CATEGORICAL_FEATURES)
            test_pool  = Pool(X_test_cat,  y_test,  cat_features=CATEGORICAL_FEATURES)
            model.fit(train_pool)
            y_pred = model.predict(test_pool).flatten().astype(int)
            y_prob = model.predict_proba(test_pool)
        else:
            # XGBoost는 NaN을 자체 처리, LightGBM/RF도 동일
            model.fit(X_train_enc, y_train)
            y_pred = model.predict(X_test_enc)
            y_prob = model.predict_proba(X_test_enc)

        train_time = time.time() - t0

        t1 = time.time()
        infer_time = (time.time() - t1) * 1000

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)
        ll  = log_loss(y_test, y_prob)

        print(f"   ✅  학습 완료 ({train_time:.1f}초)")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   Macro F1 : {f1:.4f}")
        print(f"   Log Loss : {ll:.4f}  ← 낮을수록 확률 보정 좋음")

        path = os.path.join(MODEL_DIR, f"{name.lower()}.pkl")
        joblib.dump(model, path)
        print(f"   💾  저장: {path}")

        results.append({
            "model": name,
            "accuracy": acc,
            "macro_f1": f1,
            "log_loss": ll,
            "train_sec": round(train_time, 1),
        })

    return results


# ================================================================
# 결과 요약
# ================================================================
def print_summary(results):
    print("\n" + "=" * 60)
    print("📊  모델 비교 결과")
    print("=" * 60)

    df = pd.DataFrame(results).sort_values("macro_f1", ascending=False)

    print(f"\n{'모델':<15} {'Accuracy':>10} {'Macro F1':>10} {'Log Loss':>10} {'학습(s)':>9}")
    print("-" * 60)
    for _, r in df.iterrows():
        print(
            f"{r['model']:<15} {r['accuracy']:>10.4f} {r['macro_f1']:>10.4f} "
            f"{r['log_loss']:>10.4f} {r['train_sec']:>9.1f}"
        )

    best = df.iloc[0]
    print(f"\n🏆  최고 성능: {best['model']} (Macro F1 = {best['macro_f1']:.4f})")


# ================================================================
# 메인
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-rf", action="store_true")
    parser.add_argument("--quick",   action="store_true")
    args = parser.parse_args()

    # 1. 데이터 로드
    df = load_data()

    if args.quick:
        df = df.sample(20_000, random_state=42)
        print(f"⚡  Quick 모드: {len(df):,}건만 사용\n")

    # 2. 타겟 인코딩
    target_le = LabelEncoder()
    y = target_le.fit_transform(df[TARGET])
    n_classes = len(target_le.classes_)
    print(f"🎯  타겟 클래스 ({n_classes}개): {list(target_le.classes_)}\n")

    feature_cols = [f for f in FEATURES if f in df.columns]
    cat_indices  = [i for i, f in enumerate(feature_cols) if f in CATEGORICAL_FEATURES]

    # 3A. 인코딩 버전 (LightGBM/XGBoost/RF)
    df_enc, label_encoders = encode_for_tree(df[feature_cols], fit=True)
    X_enc = df_enc[feature_cols].values.astype(float)

    # 3B. CatBoost용 원본 DataFrame
    df_cat = prepare_for_catboost(df[feature_cols])

    # 4. Train/Test split (동일 인덱스 사용)
    idx = np.arange(len(df))
    idx_train, idx_test = train_test_split(idx, test_size=0.2, random_state=42, stratify=y)

    X_train_enc, X_test_enc = X_enc[idx_train], X_enc[idx_test]
    X_train_cat = df_cat.iloc[idx_train][feature_cols]
    X_test_cat  = df_cat.iloc[idx_test][feature_cols]
    y_train, y_test = y[idx_train], y[idx_test]

    print(f"📂  Train: {len(idx_train):,}건 / Test: {len(idx_test):,}건\n")

    # 5. 메타데이터 저장
    meta = {
        "target_le": target_le,
        "label_encoders": label_encoders,
        "feature_cols": feature_cols,
        "cat_indices": cat_indices,
        "n_classes": n_classes,
        "classes": list(target_le.classes_),
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "meta.pkl"))
    print("💾  메타데이터 저장: ml/models/meta.pkl\n")

    # 6. 학습 + 평가
    models = get_models(n_classes, cat_indices, skip_rf=args.skip_rf)
    results = train_and_evaluate(
        models,
        X_train_enc, X_test_enc,
        X_train_cat, X_test_cat,
        y_train, y_test,
        cat_indices,
    )

    # 7. 요약
    print_summary(results)


if __name__ == "__main__":
    main()
