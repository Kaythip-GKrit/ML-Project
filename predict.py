# -*- coding: utf-8 -*-
"""
predict.py
==========
ใช้โมเดลที่ฝึกแล้ว (rf_model.joblib) ทำนายเหตุการณ์ใหม่

ใช้เป็นไลบรารี:
    from predict import FatalityPredictor
    p = FatalityPredictor("rf_model.joblib")
    p.predict_one({
        "incident_datetime": "2022-11-19 23:10",
        "province_en": "Surin",
        "agency": "department of rural roads",
        "vehicle_type": "motorcycle",
        "accident_type": "head-on collision (not overtaking)",
        "number_of_vehicles_involved": 2,
        "weather_condition": "clear",
        "road_description": "straight road",
        "latitude": 14.88, "longitude": 103.49,   # ไม่บังคับ ถ้าไม่มีจะใช้ค่ากลางของจังหวัด
    })

ใช้จาก command line (ทำนายทั้งไฟล์ CSV):
    python predict.py --csv new_accidents.csv --out predictions.csv
    python predict.py --demo
"""
import argparse

import joblib
import numpy as np
import pandas as pd

import rf_features as rf  # ต้อง import ก่อน joblib.load เพื่อให้หาคลาส GeoImputer เจอ

REQUIRED = ["incident_datetime", "province_en", "agency", "vehicle_type", "accident_type",
            "number_of_vehicles_involved", "weather_condition", "road_description"]


class FatalityPredictor:
    def __init__(self, model_path="rf_model.joblib", threshold=None):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.threshold = float(threshold if threshold is not None else bundle["threshold"])
        self.known = bundle["known_values"]
        self.info = {k: v for k, v in bundle.items() if k != "model"}

    def _check(self, df: pd.DataFrame):
        missing = [c for c in REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"ขาดคอลัมน์ที่จำเป็น: {missing}")
        warnings = []
        for col, key in [("province_en", "province"), ("agency", "agency"),
                         ("vehicle_type", "vehicle_type"), ("accident_type", "accident_type"),
                         ("weather_condition", "weather_condition"), ("road_description", "road_description")]:
            if key not in self.known:
                continue
            vals = df[col].dropna()
            if key in ("weather_condition", "road_description"):
                vals = vals.astype(str).str.strip().str.lower()
            bad = sorted(set(vals) - set(self.known[key]) - {"buogkan", "unknown"})
            if bad:
                warnings.append(f"{col} มีค่าที่โมเดลไม่เคยเห็น {bad} (โมเดลจะถือเป็นหมวดที่พบน้อย)")
        return warnings

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """รับ DataFrame หลายแถว คืน DataFrame ที่เพิ่มคอลัมน์ prob_fatal และ prediction"""
        df = df.copy()
        for c in ["latitude", "longitude"]:
            if c not in df.columns:
                df[c] = np.nan
        for w in self._check(df):
            print("คำเตือน:", w)
        proba = self.model.predict_proba(rf.prepare(df))[:, 1]
        df["prob_fatal"] = proba.round(4)
        df["prediction"] = np.where(proba >= self.threshold, "Fatal", "Non-fatal")
        return df

    def predict_one(self, record: dict) -> dict:
        out = self.predict(pd.DataFrame([record])).iloc[0]
        return {"prob_fatal": float(out["prob_fatal"]), "prediction": out["prediction"],
                "threshold": self.threshold}


DEMO = [
    {"incident_datetime": "2022-11-19 23:10", "province_en": "Surin", "agency": "department of rural roads",
     "vehicle_type": "motorcycle", "accident_type": "head-on collision (not overtaking)",
     "number_of_vehicles_involved": 2, "weather_condition": "clear", "road_description": "wide curve",
     "latitude": 14.88, "longitude": 103.49},
    {"incident_datetime": "2022-06-14 13:20", "province_en": "Chon Buri", "agency": "department of highways",
     "vehicle_type": "private/passenger car", "accident_type": "rear-end collision",
     "number_of_vehicles_involved": 2, "weather_condition": "rainy", "road_description": "straight road",
     "latitude": 13.36, "longitude": 100.98},
    {"incident_datetime": "2022-09-03 19:40", "province_en": "Nakhon Ratchasima", "agency": "department of highways",
     "vehicle_type": "pedestrian", "accident_type": "pedestrian collision",
     "number_of_vehicles_involved": 1, "weather_condition": "dark",
     "road_description": "t-intersection"},  # ไม่ใส่พิกัด -> ใช้ค่ากลางของจังหวัด
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf_model.joblib")
    ap.add_argument("--csv")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    p = FatalityPredictor(args.model)
    if args.csv:
        result = p.predict(pd.read_csv(args.csv))
        result.to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"ทำนาย {len(result):,} แถว บันทึกที่ {args.out}")
        print(result["prediction"].value_counts())
    else:
        for rec in DEMO:
            r = p.predict_one(rec)
            print(f"{rec['vehicle_type']:<25} {rec['accident_type']:<36} "
                  f"P(Fatal)={r['prob_fatal']:.3f} -> {r['prediction']}")
