# -*- coding: utf-8 -*-
"""
train_model.py
==============
ฝึกโมเดล Random Forest ตัวสุดท้าย แล้วบันทึกเป็นไฟล์พร้อมใช้งาน (rf_model.joblib)

ขั้นตอน:
  1. โหลด CSV -> ตัดเหตุชนท้ายที่มียานพาหนะ 0-1 คัน -> ทำความสะอาด -> สร้าง 21 Features
  2. แบ่ง Train/Test แบบ Stratified 80/20 (seed 42 เหมือนในรายงาน)
  3. Tuning ด้วย RandomizedSearchCV (16 ชุด, 5-fold CV, เกณฑ์ PR-AUC) บนชุดฝึก
  4. หา threshold ที่ให้ F1 สูงสุด จาก 5-fold Cross-validation บนชุดฝึก
  5. ฝึกโมเดลด้วยชุดฝึก วัดผลบนชุดทดสอบ และคำนวณ Permutation Importance
  6. (ถ้าใส่ --full) ฝึกใหม่ด้วยข้อมูลทั้งหมดก่อนบันทึก เพื่อใช้งานจริง
  7. บันทึกโมเดล + threshold + รายการค่าที่รู้จัก ลง rf_model.joblib

รัน:
  python train_model.py --csv thai_road_accident_2019_2022.csv
  python train_model.py --csv thai_road_accident_2019_2022.csv --full
  python train_model.py --csv thai_road_accident_2019_2022.csv --no-tune   # ข้าม Tuning ใช้ BEST_RF_PARAMS
"""
import argparse
import json
import os
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold, cross_val_predict,
                                     train_test_split)

import rf_features as rf

# ช่วงค่าที่ค้นหา (เหมือนรอบก่อนใน model evaluation summary.md)
PARAM_SPACE = {
    "n_estimators": [150, 250, 350],
    "max_depth": [12, 18, 25, None],
    "min_samples_leaf": [1, 3, 5, 10, 20],
    "max_features": ["sqrt", 0.1, 0.2, 0.3],
    "class_weight": [None, "balanced_subsample"],
}
N_ITER = 16


def evaluate(y, proba, thr):
    yhat = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat).ravel()
    return {"accuracy": accuracy_score(y, yhat), "precision": precision_score(y, yhat, zero_division=0),
            "recall": recall_score(y, yhat), "f1": f1_score(y, yhat),
            "roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="thai_road_accident_2019_2022.csv")
    ap.add_argument("--out", default="rf_model.joblib")
    ap.add_argument("--full", action="store_true", help="ฝึกใหม่ด้วยข้อมูลทั้งหมดหลังประเมินผล")
    ap.add_argument("--no-tune", action="store_true", help="ข้าม RandomizedSearchCV ใช้ BEST_RF_PARAMS")
    ap.add_argument("--compare", default="model_info_v2_22feat.json",
                    help="ไฟล์ model_info ของโมเดลเดิม สำหรับพิมพ์ผลเทียบ")
    args = ap.parse_args()
    t0 = time.time()

    # 1. โหลดและเตรียมข้อมูล
    raw = pd.read_csv(args.csv)
    # ตัดเหตุชนท้ายที่บันทึกว่ามียานพาหนะ 0-1 คัน (ชนท้ายต้องมีอย่างน้อย 2 คัน จึงน่าจะบันทึกผิด)
    # ทำตรงนี้ก่อน clean() เพราะ clean() จะปรับ 0 คันเป็น 1 คัน
    bad_rear = ((raw["accident_type"] == "rear-end collision")
                & (pd.to_numeric(raw["number_of_vehicles_involved"], errors="coerce") < 2))
    raw = raw.loc[~bad_rear].reset_index(drop=True)
    print(f"ตัดเหตุชนท้ายที่มียานพาหนะ 0-1 คัน ออก {bad_rear.sum():,} แถว")
    y = (raw["number_of_fatalities"] > 0).astype(int)
    X = rf.prepare(raw)
    print(f"ข้อมูล {len(X):,} แถว | Features {X.shape[1]} ตัว | สัดส่วน Fatal {y.mean():.4f}")

    # 2. แบ่ง Train/Test
    tr_idx, te_idx = train_test_split(X.index, test_size=0.2, stratify=y, random_state=rf.SEED)
    Xtr, ytr, Xte, yte = X.loc[tr_idx], y.loc[tr_idx], X.loc[te_idx], y.loc[te_idx]
    print(f"Train {len(Xtr):,} แถว | Test {len(Xte):,} แถว")
    cv = StratifiedKFold(5, shuffle=True, random_state=rf.SEED)

    # 3. Tuning ด้วย RandomizedSearchCV บนชุดฝึก (ไม่แตะชุดทดสอบ)
    tuning = None
    if args.no_tune:
        params = dict(rf.BEST_RF_PARAMS)
        print("ข้าม Tuning ใช้ BEST_RF_PARAMS:", params)
    else:
        search = RandomizedSearchCV(
            rf.build_pipeline(), {f"clf__{k}": v for k, v in PARAM_SPACE.items()},
            n_iter=N_ITER, scoring="average_precision", cv=cv, random_state=rf.SEED,
            n_jobs=-1, refit=False, verbose=1)
        search.fit(Xtr, ytr)
        params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}
        tuning = {"method": "RandomizedSearchCV", "n_iter": N_ITER, "cv": 5, "scoring": "average_precision",
                  "best_cv_pr_auc": float(search.best_score_), "param_space": PARAM_SPACE}
        print(f"Best params: {params} | CV PR-AUC = {search.best_score_:.4f} ({time.time() - t0:.0f}s)")

    # 4. หา threshold จาก 5-fold CV บนชุดฝึก
    oof = cross_val_predict(rf.build_pipeline(**params), Xtr, ytr, cv=cv, method="predict_proba",
                            n_jobs=-1)[:, 1]
    pr, rc, th = precision_recall_curve(ytr, oof)
    f1 = 2 * pr * rc / (pr + rc + 1e-12)
    threshold = float(th[np.nanargmax(f1[:-1])])
    print(f"threshold ที่ให้ F1 สูงสุดบน CV = {threshold:.4f} ({time.time() - t0:.0f}s)")

    # 5. ฝึกด้วยชุดฝึก และประเมินบนชุดทดสอบ
    model = rf.build_pipeline(**params).fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:, 1]
    test_metrics = evaluate(yte, proba, threshold)
    print("ผลบนชุดทดสอบ:", {k: round(v, 4) if isinstance(v, float) else v for k, v in test_metrics.items()})

    if os.path.exists(args.compare):
        old_info = json.load(open(args.compare, encoding="utf-8"))
        old, n_old = old_info["test_metrics"], len(old_info["features"])
        print(f"\n{'metric':<10}{f'เดิม ({n_old})':>12}{f'ใหม่ ({X.shape[1]})':>12}{'ต่าง':>10}")
        for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
            print(f"{k:<10}{old[k]:>12.4f}{test_metrics[k]:>12.4f}{test_metrics[k] - old[k]:>+10.4f}")

    # Permutation Importance บนชุดทดสอบ (สลับค่าทีละ Feature แล้ววัด PR-AUC ที่ลดลง)
    # หมายเหตุ: คู่ที่ซ้ำซ้อนกัน (เช่น road_description / road_group) จะได้ค่าต่ำกว่าความจริง
    # เพราะตอนสลับตัวหนึ่ง โมเดลยังใช้อีกตัวแทนได้
    pi = permutation_importance(model, Xte, yte, scoring="average_precision", n_repeats=5,
                                random_state=rf.SEED, n_jobs=-1)
    perm_imp = pd.Series(pi.importances_mean, index=Xte.columns).sort_values(ascending=False)
    print("\nPermutation Importance (PR-AUC ที่ลดลง):")
    for f, v in perm_imp.items():
        mark = "  <- ใหม่" if f in {"weather_condition", "weather_group", "road_description", "road_group"} else ""
        print(f"  {f:<20}{v:>9.4f}{mark}")

    # 6. (ทางเลือก) ฝึกใหม่ด้วยข้อมูลทั้งหมด
    if args.full:
        model = rf.build_pipeline(**params).fit(X, y)
        print("ฝึกใหม่ด้วยข้อมูลทั้งหมดแล้ว (ตัวเลขผลประเมินข้างบนมาจากโมเดลที่ฝึกด้วย 80%)")

    # 7. บันทึก
    known = {
        "province": sorted(raw["province_en"].replace({"buogkan": "Bueng Kan"}).loc[lambda s: s != "unknown"].unique().tolist()),
        "vehicle_type": sorted(raw["vehicle_type"].unique().tolist()),
        "accident_type": sorted(raw["accident_type"].unique().tolist()),
        "weather_condition": sorted(raw["weather_condition"].dropna().str.strip().str.lower().unique().tolist()),
        "road_description": sorted(raw["road_description"].dropna().str.strip().str.lower().unique().tolist()),
    }
    bundle = {
        "model": model,
        "threshold": threshold,
        "features": rf.FEATURES,
        "params": params,
        "tuning": tuning,
        "known_values": known,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "permutation_importance": {k: float(v) for k, v in perm_imp.items()},
        "trained_on": "all data" if args.full else "80% train split",
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(bundle, args.out, compress=3)
    with open("model_info.json", "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"}, f, ensure_ascii=False, indent=1)
    print(f"บันทึก {args.out} และ model_info.json แล้ว ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
