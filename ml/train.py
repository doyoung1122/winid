"""
2단계 계층 분류 모델 학습

  Stage 1: cause_main → 6그룹 분류
           (부주의 / 전기적요인 / 기계적요인 / 방화 / 기타특수)
           미상·기타는 학습에서 제외

  Stage 2: cause_sub 전용 분류기 3종
           전기적요인(10) / 부주의(13→합산) / 기계적요인(8→합산)

실행 (프로젝트 루트에서):
  python ml/train.py              # 전체 학습
  python ml/train.py --quick      # 30K 샘플 빠른 테스트
  python ml/train.py --stage1     # Stage 1만
"""

import os, sys, time, warnings, argparse
import numpy as np
import pandas as pd
import pymysql
import joblib
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# .env 로드 (프로젝트 루트)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(ROOT, ".env"))

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# ================================================================
# 설정
# ================================================================

# 감식 완료 후 입력 가능한 모든 feature
ALL_FEATURES = [
    # 건물 / 위치
    "fire_type", "building_type", "building_structure",
    "location_main", "location_mid", "location_sub", "region",
    # 현장 증거 (조사 중 확인 가능)
    "first_object_main", "first_object_sub",
    "heat_source", "device_main", "device_sub",
    # 피해 규모
    "damage_amount", "death_count", "injury_count", "suppression_time",
    # 기상 / 시간
    "temperature", "humidity", "wind_speed",
    "month", "hour",
    # 건물 규모
    "building_area", "floor_area",
]

# 레이블 인코딩 대상 (문자열 범주형)
CAT_COLS = [
    "fire_type", "building_type", "building_structure",
    "location_main", "location_mid", "location_sub", "region",
    "first_object_main", "first_object_sub",
    "heat_source", "device_main", "device_sub",
]

# wind_speed → 순서형 정수 (별도 처리)
WIND_ORDER = {"0~4 m/s": 1, "5~8 m/s": 2, "9~12 m/s": 3, "13~17 m/s": 4, "18 m/s 이상": 5}

# 미상/빈값으로 간주할 값들
NAN_VALS = {"미상", "미상(발화원인)", "기타(발화원인)", ""}

# cause_main → Stage 1 그룹 매핑 (미상·기타는 None → 학습 제외)
CAUSE_GROUP = {
    "부주의":          "부주의",
    "전기적 요인":     "전기적 요인",
    "기계적 요인":     "기계적 요인",
    "방화":            "방화",
    "방화의심":        "방화",       # 방화와 통합
    "화학적 요인":     "기타특수",
    "가스누출(폭발)":  "기타특수",
    "자연적인 요인":   "기타특수",
    "교통사고":        "기타특수",
    "제품결함":        "기타특수",
    # "미상", "기타" → 제외
}

# Stage 2 학습 대상 그룹
STAGE2_GROUPS = ["전기적 요인", "부주의", "기계적 요인"]

# 이 건수 미만인 sub 클래스는 기타로 합산
SUB_MIN = 200

# LightGBM 공통 파라미터
LGBM_BASE = dict(
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)


# ================================================================
# DB 로드
# ================================================================
def load_data() -> pd.DataFrame:
    print("=" * 60)
    print("MySQL → vfs_data_csv 로드")
    print("=" * 60)

    conn = pymysql.connect(
        host=os.getenv("MY_HOST", "218.158.62.155"),
        port=int(os.getenv("MY_PORT", 33307)),
        user=os.getenv("MY_USER", "vfs"),
        password=os.getenv("MY_PASS"),
        database=os.getenv("MY_DB", "vfs"),
        charset="utf8mb4",
    )
    cols = ", ".join(ALL_FEATURES + ["cause_main", "cause_sub"])
    df = pd.read_sql(f"SELECT {cols} FROM vfs_data_csv", conn)
    conn.close()

    print(f"로드 완료: {len(df):,}건\n")
    return df


# ================================================================
# 전처리
# ================================================================
def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    """
    df      : 원본 DataFrame
    encoders: 기존 인코더 dict (fit=False 시 재사용)
    fit     : True면 인코더를 새로 학습
    반환    : (전처리된 df, encoders)
    """
    df = df.copy()

    # 1. 문자열 범주형: 미상·빈값 → NaN, 공백 제거
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].apply(
            lambda v: np.nan if (pd.isna(v) or str(v).strip() in NAN_VALS)
            else str(v).strip()
        )

    # 2. wind_speed → 순서형 정수 (1~5, 미상=NaN)
    if "wind_speed" in df.columns:
        df["wind_speed"] = df["wind_speed"].map(WIND_ORDER)

    # 3. 수치형 이상치 처리
    if "temperature" in df.columns:
        df["temperature"] = pd.to_numeric(df["temperature"], errors="coerce").clip(-40, 50)
    if "humidity" in df.columns:
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce").clip(0, 100)
    if "building_area" in df.columns:
        df["building_area"] = df["building_area"].replace(0, np.nan)
        df["building_area"] = np.log1p(pd.to_numeric(df["building_area"], errors="coerce").clip(lower=0))
    if "floor_area" in df.columns:
        df["floor_area"] = df["floor_area"].replace(0, np.nan)
        df["floor_area"] = np.log1p(pd.to_numeric(df["floor_area"], errors="coerce").clip(lower=0))
    if "damage_amount" in df.columns:
        df["damage_amount"] = np.log1p(pd.to_numeric(df["damage_amount"], errors="coerce").clip(lower=0))
    if "suppression_time" in df.columns:
        df["suppression_time"] = df["suppression_time"].replace(0, np.nan)
        df["suppression_time"] = pd.to_numeric(df["suppression_time"], errors="coerce").clip(upper=86000)
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce")

    # 4. 범주형 LabelEncoding (미상=NaN 유지, NaN은 LightGBM이 자체 처리)
    if encoders is None:
        encoders = {}
    for col in CAT_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            le.fit(df[col].dropna().astype(str))
            encoders[col] = le
        le = encoders[col]
        df[col] = df[col].apply(
            lambda v: float(le.transform([str(v)])[0])
            if pd.notna(v) and str(v) in le.classes_
            else np.nan
        )

    return df, encoders


# ================================================================
# LightGBM 학습 + 평가
# ================================================================
def get_cat_feature_indices(feature_cols: list) -> list:
    return [i for i, c in enumerate(feature_cols) if c in CAT_COLS]


def train_lgbm(X_train, X_test, y_train, y_test, classes: list, tag: str = "") -> tuple:
    n_classes = len(classes)
    params = dict(**LGBM_BASE)

    if n_classes == 2:
        params["objective"] = "binary"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes

    cat_indices = get_cat_feature_indices(X_train.columns.tolist())

    model = LGBMClassifier(**params)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        categorical_feature=cat_indices,
        eval_set=[(X_test, y_test)],
        callbacks=[],
    )
    elapsed = time.time() - t0

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n[{tag}] 학습시간={elapsed:.1f}s  Accuracy={acc:.4f}  Macro-F1={f1:.4f}")
    print(classification_report(
        y_test, y_pred,
        target_names=[str(c) for c in classes],
        zero_division=0,
    ))

    return model, acc, f1


# ================================================================
# Stage 1: cause_main 6그룹 분류
# ================================================================
def run_stage1(df_proc: pd.DataFrame, df_raw: pd.DataFrame,
               encoders: dict, quick: bool):
    print("\n" + "=" * 60)
    print("STAGE 1: cause_main 6그룹 분류기")
    print("=" * 60)

    # 그룹 매핑 적용, 미상·기타 제외
    group_series = df_raw["cause_main"].map(CAUSE_GROUP)
    mask = group_series.notna()
    X_all = df_proc[mask].copy()
    y_labels = group_series[mask]

    print(f"학습 데이터: {len(X_all):,}건 (미상·기타 {(~mask).sum():,}건 제외)")
    print("클래스 분포:")
    for cls, cnt in y_labels.value_counts().items():
        print(f"  {cls:<15} {cnt:>7,}건")
    print()

    target_le = LabelEncoder()
    y = target_le.fit_transform(y_labels)
    classes = list(target_le.classes_)

    feature_cols = [c for c in ALL_FEATURES if c in X_all.columns]
    X = X_all[feature_cols]

    if quick:
        idx = np.random.RandomState(42).choice(len(X), min(30_000, len(X)), replace=False)
        X, y = X.iloc[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}건 / Test: {len(X_test):,}건")

    model, acc, f1 = train_lgbm(X_train, X_test, y_train, y_test, classes, tag="Stage1")

    # 저장
    joblib.dump(model, os.path.join(MODEL_DIR, "stage1.pkl"))
    meta = {
        "feature_cols": feature_cols,
        "encoders": encoders,
        "target_le": target_le,
        "classes": classes,
        "cause_group_map": CAUSE_GROUP,
        "wind_order": WIND_ORDER,
        "nan_vals": NAN_VALS,
        "accuracy": acc,
        "macro_f1": f1,
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "stage1_meta.pkl"))
    print("저장 완료: models/stage1.pkl, stage1_meta.pkl")

    return model, meta


# ================================================================
# Stage 2: cause_sub 전용 분류기
# ================================================================
def run_stage2(df_proc: pd.DataFrame, df_raw: pd.DataFrame,
               encoders: dict, group_name: str, quick: bool):
    print(f"\n{'=' * 60}")
    print(f"STAGE 2: [{group_name}] cause_sub 분류기")
    print("=" * 60)

    # 해당 그룹 데이터만 추출
    mask = df_raw["cause_main"].map(CAUSE_GROUP) == group_name
    X_g = df_proc[mask].copy()
    sub_raw = df_raw.loc[mask, "cause_sub"].copy()

    # 소수 클래스 합산: 기존 "기타(...)" 레이블에 흡수
    sub_counts = sub_raw.value_counts()
    rare = sub_counts[sub_counts < SUB_MIN].index.tolist()

    existing_other = [s for s in sub_counts.index if str(s).startswith("기타(")]
    other_label = existing_other[0] if existing_other else f"기타({group_name})"

    sub_series = sub_raw.copy()
    sub_series[sub_series.isin(rare)] = other_label

    print(f"전체 건수: {len(X_g):,}건")
    if rare:
        print(f"소수 클래스(<{SUB_MIN}건) → '{other_label}' 합산: {rare}")
    print("최종 클래스 분포:")
    for cls, cnt in sub_series.value_counts().items():
        print(f"  {str(cls):<35} {cnt:>6,}건")
    print()

    target_le = LabelEncoder()
    y = target_le.fit_transform(sub_series)
    classes = list(target_le.classes_)

    feature_cols = [c for c in ALL_FEATURES if c in X_g.columns]
    X = X_g[feature_cols]

    if quick:
        n = min(15_000, len(X))
        idx = np.random.RandomState(42).choice(len(X), n, replace=False)
        X, y = X.iloc[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train):,}건 / Test: {len(X_test):,}건")

    model, acc, f1 = train_lgbm(X_train, X_test, y_train, y_test, classes, tag=f"Stage2-{group_name}")

    # 파일명용 태그 (공백·괄호 제거)
    safe = group_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
    model_path = os.path.join(MODEL_DIR, f"stage2_{safe}.pkl")
    meta_path  = os.path.join(MODEL_DIR, f"stage2_{safe}_meta.pkl")

    joblib.dump(model, model_path)
    meta = {
        "group_name": group_name,
        "feature_cols": feature_cols,
        "encoders": encoders,
        "target_le": target_le,
        "classes": classes,
        "other_label": other_label,
        "rare_labels": rare,
        "wind_order": WIND_ORDER,
        "nan_vals": NAN_VALS,
        "accuracy": acc,
        "macro_f1": f1,
    }
    joblib.dump(meta, meta_path)
    print(f"저장 완료: {os.path.basename(model_path)}, {os.path.basename(meta_path)}")

    return model, meta


# ================================================================
# 최종 요약
# ================================================================
def print_summary(results: list):
    print("\n" + "=" * 60)
    print("전체 학습 결과 요약")
    print("=" * 60)
    print(f"{'모델':<35} {'Accuracy':>10} {'Macro-F1':>10}")
    print("-" * 57)
    for r in results:
        print(f"  {r['tag']:<33} {r['acc']:>10.4f} {r['f1']:>10.4f}")


# ================================================================
# 메인
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",  action="store_true", help="샘플링으로 빠른 테스트")
    parser.add_argument("--stage1", action="store_true", help="Stage 1만 학습")
    args = parser.parse_args()

    if args.quick:
        print("⚡ Quick 모드 (샘플링)\n")

    # 1. 데이터 로드
    df_raw = load_data()

    # 2. 전처리 (전체 데이터 기준으로 encoder fit → Stage 1·2 공유)
    print("전처리 중...")
    df_proc, encoders = preprocess(df_raw, fit=True)
    print("전처리 완료\n")

    results = []

    # 3. Stage 1
    _, s1_meta = run_stage1(df_proc, df_raw, encoders, quick=args.quick)
    results.append({"tag": "Stage1  cause_main (6그룹)", "acc": s1_meta["accuracy"], "f1": s1_meta["macro_f1"]})

    if args.stage1:
        print_summary(results)
        return

    # 4. Stage 2 (3종)
    for group in STAGE2_GROUPS:
        _, s2_meta = run_stage2(df_proc, df_raw, encoders, group_name=group, quick=args.quick)
        results.append({
            "tag": f"Stage2  cause_sub [{group}]",
            "acc": s2_meta["accuracy"],
            "f1": s2_meta["macro_f1"],
        })

    print_summary(results)
    print(f"\n모델 저장 위치: {MODEL_DIR}")


if __name__ == "__main__":
    main()
