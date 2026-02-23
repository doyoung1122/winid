"""
저장된 모델로 발화원인 확률 예측

사용법:
  from ml.predict import FireCausePredictor

  predictor = FireCausePredictor("lightgbm")  # or "catboost", "xgboost"
  result = predictor.predict({
      "fire_type": "건축,구조물",
      "location_main": "위락시설",
      "location_sub": "전기실",
      "region": "서울특별시",
      "temperature": 15.0,
  })
  # → [{"cause": "전기적 요인", "probability": 0.72}, ...]
"""

import os
import numpy as np
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")


class FireCausePredictor:
    def __init__(self, model_name="lightgbm"):
        """
        model_name: "lightgbm" | "catboost" | "xgboost" | "randomforest"
        """
        meta_path = os.path.join(MODEL_DIR, "meta.pkl")
        model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

        if not os.path.exists(meta_path):
            raise FileNotFoundError("meta.pkl 없음. python ml/train.py 먼저 실행하세요.")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_name}.pkl 없음. python ml/train.py 먼저 실행하세요.")

        self.meta = joblib.load(meta_path)
        self.model = joblib.load(model_path)
        self.model_name = model_name

    def predict(self, input_dict: dict, top_k: int = 5) -> list[dict]:
        """
        input_dict: 알고 있는 필드만 넣으면 됨 (나머지는 NaN 처리)
        top_k: 상위 몇 개 원인까지 반환할지

        반환:
          [{"cause": "전기적 요인", "probability": 0.72}, ...]
        """
        feature_cols = self.meta["feature_cols"]
        label_encoders = self.meta["label_encoders"]
        classes = self.meta["classes"]

        # 1. 피처 벡터 구성 (없는 필드 → NaN)
        row = {}
        for col in feature_cols:
            val = input_dict.get(col, None)

            if val is None or val == "" or val == "미상":
                row[col] = np.nan
            elif col in label_encoders:
                le = label_encoders[col]
                if val in le.classes_:
                    row[col] = float(le.transform([val])[0])
                else:
                    row[col] = np.nan  # 학습 때 없던 카테고리
            else:
                try:
                    row[col] = float(val)
                except (ValueError, TypeError):
                    row[col] = np.nan

        X = np.array([[row[col] for col in feature_cols]], dtype=float)

        # 2. 예측
        proba = self.model.predict_proba(X)[0]

        # 3. 상위 k개 반환
        top_indices = np.argsort(proba)[::-1][:top_k]
        results = [
            {
                "cause": classes[i],
                "probability": round(float(proba[i]), 4),
                "probability_pct": f"{proba[i]*100:.1f}%",
            }
            for i in top_indices
            if proba[i] > 0.01  # 1% 미만은 제외
        ]

        return results

    def predict_text(self, input_dict: dict, top_k: int = 5) -> str:
        """예측 결과를 자연어 문장으로 반환"""
        results = self.predict(input_dict, top_k)
        if not results:
            return "예측 결과가 없습니다."

        lines = ["발화원인 예측 결과:"]
        for i, r in enumerate(results, 1):
            lines.append(f"  {i}. {r['cause']}: {r['probability_pct']}")

        return "\n".join(lines)


# ================================================================
# 단독 실행 테스트
# ================================================================
if __name__ == "__main__":
    import sys

    model_name = sys.argv[1] if len(sys.argv) > 1 else "lightgbm"

    print(f"🔍  모델: {model_name}")
    predictor = FireCausePredictor(model_name)

    test_cases = [
        {
            "name": "서울 위락시설 전기실",
            "input": {
                "fire_type": "건축,구조물",
                "location_main": "위락시설",
                "location_sub": "전기실",
                "region": "서울특별시",
                "temperature": 15.0,
                "humidity": 60.0,
            },
        },
        {
            "name": "경기 주거 주방 (정보 최소)",
            "input": {
                "location_main": "주거",
                "location_mid": "공동주택",
                "location_sub": "주방",
                "region": "경기도",
            },
        },
        {
            "name": "부산 산업시설 창고",
            "input": {
                "fire_type": "건축,구조물",
                "location_main": "산업시설",
                "location_sub": "창고",
                "region": "부산광역시",
                "temperature": 25.0,
                "humidity": 70.0,
                "wind_speed": "0~4 m/s",
            },
        },
    ]

    print()
    for case in test_cases:
        print(f"📋  사례: {case['name']}")
        print(f"    입력: {case['input']}")
        print(f"    {predictor.predict_text(case['input'])}")
        print()
