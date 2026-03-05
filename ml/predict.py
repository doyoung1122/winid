"""
2단계 계층 분류 모델 추론

사용법:
  from ml.predict import FireCausePredictor

  predictor = FireCausePredictor()
  result = predictor.predict({
      "fire_type":        "건축,구조물",
      "location_main":    "주거",
      "location_mid":     "공동주택",
      "first_object_main": "전기,전자",
      "heat_source":      "작동기기",
      "damage_amount":    5000,
      "month":            7,
      "hour":             14,
  })
  # 반환 예시:
  # {
  #   "stage1": [{"cause": "전기적 요인", "prob": 0.72, "pct": "72.0%"}, ...],
  #   "stage2": [{"sub": "절연열화에 의한 단락", "prob": 0.41, "pct": "41.0%"}, ...],
  #   "top_cause": "전기적 요인",
  #   "top_sub": "절연열화에 의한 단락",
  # }
"""

import os
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Stage 2 모델 파일명 규칙 (group_name → safe tag)
def _safe_tag(group_name: str) -> str:
    return group_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")

# Stage 2 모델이 존재하는 그룹 목록
STAGE2_GROUPS = ["전기적 요인", "부주의", "기계적 요인"]


class FireCausePredictor:
    def __init__(self):
        # Stage 1 로드
        s1_model_path = os.path.join(MODEL_DIR, "stage1.pkl")
        s1_meta_path  = os.path.join(MODEL_DIR, "stage1_meta.pkl")
        if not os.path.exists(s1_model_path):
            raise FileNotFoundError("stage1.pkl 없음. python ml/train.py 먼저 실행하세요.")

        self.s1_model = joblib.load(s1_model_path)
        self.s1_meta  = joblib.load(s1_meta_path)

        # Stage 2 로드 (있는 것만)
        self.s2_models = {}
        self.s2_metas  = {}
        for group in STAGE2_GROUPS:
            tag = _safe_tag(group)
            mp = os.path.join(MODEL_DIR, f"stage2_{tag}.pkl")
            ep = os.path.join(MODEL_DIR, f"stage2_{tag}_meta.pkl")
            if os.path.exists(mp) and os.path.exists(ep):
                self.s2_models[group] = joblib.load(mp)
                self.s2_metas[group]  = joblib.load(ep)

    # ------------------------------------------------------------------
    # 내부 전처리 (train.py preprocess()와 동일 로직)
    # ------------------------------------------------------------------
    def _build_row(self, input_dict: dict, meta: dict) -> pd.DataFrame:
        feature_cols = meta["feature_cols"]
        encoders     = meta["encoders"]
        wind_order   = meta["wind_order"]
        nan_vals     = meta.get("nan_vals", {"미상", "미상(발화원인)", "기타(발화원인)", ""})

        CAT_COLS = [
            "fire_type", "building_type", "building_structure",
            "location_main", "location_mid", "location_sub", "region",
            "first_object_main", "first_object_sub",
            "heat_source", "device_main", "device_sub",
        ]

        row = {}
        for col in feature_cols:
            val = input_dict.get(col, None)

            # wind_speed 별도 처리
            if col == "wind_speed":
                row[col] = wind_order.get(str(val), np.nan) if val is not None else np.nan
                continue

            # 범주형
            if col in CAT_COLS:
                if val is None or str(val).strip() in nan_vals:
                    row[col] = np.nan
                elif col in encoders:
                    le = encoders[col]
                    sv = str(val).strip()
                    row[col] = float(le.transform([sv])[0]) if sv in le.classes_ else np.nan
                else:
                    row[col] = np.nan
                continue

            # 수치형 특별 처리
            try:
                fval = float(val) if val is not None else np.nan
            except (ValueError, TypeError):
                fval = np.nan

            if col == "temperature":
                fval = np.clip(fval, -40, 50)
            elif col == "humidity":
                fval = np.clip(fval, 0, 100)
            elif col in ("building_area", "floor_area"):
                fval = np.nan if (np.isnan(fval) or fval <= 0) else np.log1p(fval)
            elif col == "damage_amount":
                fval = np.nan if (np.isnan(fval) or fval < 0) else np.log1p(fval)
            elif col == "suppression_time":
                fval = np.nan if (np.isnan(fval) or fval <= 0 or fval > 86000) else fval

            row[col] = fval

        return pd.DataFrame([row], columns=feature_cols)

    # ------------------------------------------------------------------
    # 예측
    # ------------------------------------------------------------------
    def predict(self, input_dict: dict, top_k: int = 5) -> dict:
        """
        input_dict: 알고 있는 필드만 입력 (나머지 NaN 처리)
        top_k     : 반환할 상위 클래스 수

        반환:
          stage1   : cause_main 확률 상위 k개
          stage2   : cause_sub  확률 상위 k개 (해당 그룹이 있을 때만)
          top_cause: 1위 cause_main 문자열
          top_sub  : 1위 cause_sub 문자열 (없으면 None)
        """
        # ── Stage 1 ──────────────────────────────
        X1 = self._build_row(input_dict, self.s1_meta)
        proba1 = self.s1_model.predict_proba(X1)[0]
        classes1 = self.s1_meta["classes"]

        top_idx1 = np.argsort(proba1)[::-1][:top_k]
        stage1 = [
            {
                "cause": classes1[i],
                "prob":  round(float(proba1[i]), 4),
                "pct":   f"{proba1[i]*100:.1f}%",
            }
            for i in top_idx1
            if proba1[i] > 0.01
        ]

        top_cause = stage1[0]["cause"] if stage1 else None

        # ── Stage 2 (top cause에 전용 모델 있을 때만) ──
        stage2   = []
        top_sub  = None

        if top_cause and top_cause in self.s2_models:
            meta2  = self.s2_metas[top_cause]
            model2 = self.s2_models[top_cause]

            X2 = self._build_row(input_dict, meta2)
            proba2 = model2.predict_proba(X2)[0]
            classes2 = meta2["classes"]

            top_idx2 = np.argsort(proba2)[::-1][:top_k]
            stage2 = [
                {
                    "sub":  classes2[i],
                    "prob": round(float(proba2[i]), 4),
                    "pct":  f"{proba2[i]*100:.1f}%",
                }
                for i in top_idx2
                if proba2[i] > 0.01
            ]
            top_sub = stage2[0]["sub"] if stage2 else None

        return {
            "stage1":    stage1,
            "stage2":    stage2,
            "top_cause": top_cause,
            "top_sub":   top_sub,
        }

    def predict_text(self, input_dict: dict, top_k: int = 5) -> str:
        """예측 결과를 자연어로 반환"""
        r = self.predict(input_dict, top_k)
        lines = ["[발화원인 예측]"]
        for i, s in enumerate(r["stage1"], 1):
            lines.append(f"  {i}. {s['cause']}: {s['pct']}")

        if r["stage2"]:
            lines.append(f"\n[세부원인 예측 — {r['top_cause']}]")
            for i, s in enumerate(r["stage2"], 1):
                lines.append(f"  {i}. {s['sub']}: {s['pct']}")

        return "\n".join(lines)


# ================================================================
# 단독 실행 테스트
# ================================================================
if __name__ == "__main__":
    predictor = FireCausePredictor()

    cases = [
        {
            "name": "아파트 주방 / 가스렌지 근처 (부주의 예상)",
            "input": {
                "fire_type": "건축,구조물",
                "location_main": "주거",
                "location_mid": "공동주택",
                "first_object_main": "식품",
                "heat_source": "담뱃불, 라이터불",
                "damage_amount": 500,
                "month": 12,
                "hour": 19,
                "temperature": 2.0,
                "humidity": 55.0,
            },
        },
        {
            "name": "공장 전기실 / 배전반 근처 (전기 예상)",
            "input": {
                "fire_type": "건축,구조물",
                "location_main": "산업시설",
                "location_mid": "공장시설",
                "first_object_main": "전기,전자",
                "first_object_sub": "전선피복",
                "heat_source": "작동기기",
                "device_main": "배선/배선기구",
                "device_sub": "배전반/분전반",
                "damage_amount": 80000,
                "month": 8,
                "hour": 2,
                "temperature": 30.0,
                "humidity": 75.0,
            },
        },
        {
            "name": "야외 쓰레기 / 정보 최소",
            "input": {
                "fire_type": "기타(쓰레기 화재등)",
                "location_main": "기타",
                "first_object_main": "쓰레기류",
                "month": 3,
            },
        },
    ]

    for case in cases:
        print(f"\n{'='*55}")
        print(f"사례: {case['name']}")
        print(f"입력: {case['input']}")
        print(predictor.predict_text(case["input"]))
