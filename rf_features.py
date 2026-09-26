# -*- coding: utf-8 -*-
"""
rf_features.py
==============
โมดูลกลางที่ทั้ง train_model.py, predict.py และ app.py ใช้ร่วมกัน

สำคัญ: ไฟล์นี้ต้องอยู่ในโฟลเดอร์เดียวกับไฟล์โมเดล (.joblib) เสมอ
เพราะโมเดลที่บันทึกไว้มีคลาส GeoImputer อยู่ข้างใน ตอนโหลดโมเดล Python
ต้อง import คลาสนี้จากไฟล์ชื่อ rf_features.py ให้ได้
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

SEED = 42

# ---------------------------------------------------------------------------
# Features 21 ตัวที่โมเดลใช้
#   - 17 ตัวจากชุดเดิม (คัดจาก Permutation Importance ของ Random Forest)
#     ตัด agency ออกแล้ว เพราะไม่ให้ผู้ใช้เลือกหน่วยงานในเว็บแอป
#   - 4 ตัวด้านสภาพแวดล้อม: weather_condition, weather_group, road_description, road_group
# ---------------------------------------------------------------------------
NUM_FEATURES = ["is_vulnerable_user", "is_newyear", "hour", "is_songkran", "lon", "lat",
                "night_motorcycle", "month", "n_vehicles", "is_bkk_metro", "is_night", "is_weekend"]
CAT_FEATURES = ["vehicle_type", "vehicle_group", "province", "region", "accident_group",
                "weather_condition", "weather_group", "road_description", "road_group"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# Hyperparameters ที่ได้จากการ Tuning (Randomized Search, 5-fold CV, เกณฑ์ PR-AUC)
# ค่านี้มาจากชุด 18 Features เดิม ใช้เป็นค่าตั้งต้นเมื่อรัน train_model.py --no-tune
# (ค่าที่ Tuning ได้จริงของแต่ละรอบจะถูกบันทึกไว้ใน rf_model.joblib / model_info.json ส่วน "params")
BEST_RF_PARAMS = dict(n_estimators=150, max_depth=18, min_samples_leaf=3,
                      max_features=0.2, class_weight=None)

# ---------------------------------------------------------------------------
# ตารางจับคู่สำหรับสร้าง Features
# ---------------------------------------------------------------------------
REGION = {}
for _region, _provinces in {
    "North": ["Chiang Mai", "Chiang Rai", "Lampang", "Lamphun", "Mae Hong Son", "Nan", "Phayao",
              "Phrae", "Uttaradit"],
    "Northeast": ["Amnat Charoen", "Bueng Kan", "Buri Ram", "Chaiyaphum", "Kalasin", "Khon Kaen", "Loei",
                  "Maha Sarakham", "Mukdahan", "Nakhon Phanom", "Nakhon Ratchasima", "Nong Bua Lam Phu",
                  "Nong Khai", "Roi Et", "Sakon Nakhon", "Si Sa Ket", "Surin", "Ubon Ratchathani",
                  "Udon Thani", "Yasothon"],
    "Central": ["Bangkok", "Kamphaeng Phet", "Chai Nat", "Nakhon Nayok", "Nakhon Pathom", "Nakhon Sawan",
                "Nonthaburi", "Pathum Thani", "Phra Nakhon Si Ayutthaya", "Phichit", "Phitsanulok",
                "Phetchabun", "Loburi", "Samut Prakan", "Samut Songkhram", "Samut Sakhon", "Sing Buri",
                "Sukhothai", "Suphan Buri", "Saraburi", "Ang Thong", "Uthai Thani"],
    "East": ["Chanthaburi", "Chachoengsao", "Chon Buri", "Trat", "Prachin Buri", "Rayong", "Sa Kaeo"],
    "West": ["Kanchanaburi", "Tak", "Prachuap Khiri Khan", "Phetchaburi", "Ratchaburi"],
    "South": ["Chumphon", "Krabi", "Nakhon Si Thammarat", "Narathiwat", "Pattani", "Phangnga",
              "Phatthalung", "Phuket", "Ranong", "Satun", "Songkhla", "Surat Thani", "Trang", "Yala"],
}.items():
    for _p in _provinces:
        REGION[_p] = _region

BKK_METRO = {"Bangkok", "Nonthaburi", "Pathum Thani", "Samut Prakan", "Samut Sakhon", "Nakhon Pathom"}

VEHICLE_GROUP = {
    "motorcycle": "motorcycle/3-wheeler", "motorized tricycle": "motorcycle/3-wheeler",
    "three-wheeled vehicle": "motorcycle/3-wheeler",
    "bicycle": "bicycle/pedestrian", "pedestrian": "bicycle/pedestrian",
    "private/passenger car": "car/van", "van": "car/van", "passenger pickup truck": "car/van",
    "4-wheel pickup truck": "pickup 4-wheel",
    "6-wheel truck": "heavy truck", "7-10-wheel truck": "heavy truck", "large truck with trailer": "heavy truck",
    "large passenger vehicle": "bus/large passenger",
    "other": "other", "tractor/agricultural vehicle": "other",
}

ACCIDENT_GROUP = {"side collision": "other collision", "collision during overtaking": "other collision",
                  "turning/retreating collision": "other collision"}

# สภาพอากาศ: "dark" (มืด/แสงไม่พอ) กับ "foggy" รวมเป็นกลุ่มทัศนวิสัยต่ำ
WEATHER_GROUP = {
    "clear": "clear", "rainy": "rain",
    "foggy": "low visibility", "dark": "low visibility",
    "natural disaster": "other", "land slide": "other", "other": "other",
}

# ลักษณะถนน: 19 หมวด -> 6 กลุ่ม (หลายหมวดมีไม่ถึง 30 เหตุการณ์)
ROAD_GROUP = {
    "straight road": "straight",
    "wide curve": "curve", "sharp curve": "curve",
    "t-intersection": "intersection", "four-way intersection": "intersection",
    "y-intersection": "intersection", "roundabout": "intersection", "u-turn area": "intersection",
    "connecting to public/commercial area": "access point", "connecting to private area": "access point",
    "connecting to school area": "access point",
    "grade-separated intersection/ramps": "ramp/merge", "merge lane": "ramp/merge",
    "lane-changing area": "ramp/merge",
}


# ---------------------------------------------------------------------------
# 1. ทำความสะอาดข้อมูล (ใช้ทั้งตอนฝึกและตอนทำนาย เพื่อให้ข้อมูลผ่านขั้นตอนเดียวกัน)
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    รับ DataFrame ที่มีคอลัมน์แบบไฟล์ CSV ต้นฉบับ อย่างน้อย:
      incident_datetime, province_en, vehicle_type, accident_type,
      number_of_vehicles_involved, weather_condition, road_description, latitude, longitude
    """
    d = df.copy()
    d["inc"] = pd.to_datetime(d["incident_datetime"])
    d["province"] = d["province_en"].replace({"buogkan": "Bueng Kan", "unknown": "Unknown"})

    # สภาพอากาศ / ลักษณะถนน: ตัวพิมพ์เล็ก ตัดช่องว่าง ค่าว่าง -> "other"
    for c in ["weather_condition", "road_description"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = d[c].fillna("other").astype(str).str.strip().str.lower()

    # พิกัดนอกประเทศไทย -> ค่าว่าง (จะถูกเติมด้วยค่ากลางของจังหวัดใน GeoImputer)
    for c in ["latitude", "longitude"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
    outside = ((d.latitude < 5.5) | (d.latitude > 20.6) | (d.longitude < 97.3) | (d.longitude > 105.7))
    d.loc[outside, ["latitude", "longitude"]] = np.nan

    # จำนวนยานพาหนะ: 0 -> 1, มากกว่า 5 -> 5
    d["n_vehicles"] = pd.to_numeric(d["number_of_vehicles_involved"]).clip(lower=1, upper=5)
    return d


# ---------------------------------------------------------------------------
# 2. สร้าง Features
# ---------------------------------------------------------------------------
def engineer(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    inc = d["inc"]
    d["hour"] = inc.dt.hour
    d["month"] = inc.dt.month
    d["is_weekend"] = (inc.dt.dayofweek >= 5).astype(int)
    d["is_night"] = ((d.hour >= 18) | (d.hour < 6)).astype(int)
    month_day = inc.dt.month * 100 + inc.dt.day
    d["is_songkran"] = ((month_day >= 411) & (month_day <= 417)).astype(int)
    d["is_newyear"] = ((month_day >= 1229) | (month_day <= 104)).astype(int)

    d["region"] = d["province"].map(REGION).fillna("Unknown")
    d["is_bkk_metro"] = d["province"].isin(BKK_METRO).astype(int)
    d["lat"] = d["latitude"]
    d["lon"] = d["longitude"]

    d["vehicle_group"] = d["vehicle_type"].map(VEHICLE_GROUP).fillna("other")
    d["accident_group"] = d["accident_type"].replace(ACCIDENT_GROUP)
    d["is_vulnerable_user"] = d["vehicle_group"].isin(["motorcycle/3-wheeler", "bicycle/pedestrian"]).astype(int)
    d["night_motorcycle"] = ((d.is_night == 1) & (d.vehicle_group == "motorcycle/3-wheeler")).astype(int)

    d["weather_group"] = d["weather_condition"].map(WEATHER_GROUP).fillna("other")
    d["road_group"] = d["road_description"].map(ROAD_GROUP).fillna("other")
    return d


def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """ข้อมูลดิบ -> ตาราง 21 Features พร้อมส่งเข้าโมเดล"""
    return engineer(clean(df))[FEATURES]


# ---------------------------------------------------------------------------
# 3. ส่วนประกอบของ Pipeline
# ---------------------------------------------------------------------------
class GeoImputer(BaseEstimator, TransformerMixin):
    """เติมพิกัดที่ขาดด้วยค่ามัธยฐานของจังหวัด (เรียนรู้จากชุดฝึกเท่านั้น)"""

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.prov_ = X.groupby("province")[["lat", "lon"]].median()
        self.glob_ = X[["lat", "lon"]].median()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for c in ["lat", "lon"]:
            fill = X["province"].map(self.prov_[c]).fillna(self.glob_[c])
            X[c] = X[c].fillna(fill)
        return X


def build_pipeline(**rf_params) -> Pipeline:
    """GeoImputer -> (เติมค่าตัวเลข + One-hot ตัวแปรกลุ่ม) -> Random Forest"""
    prep = ColumnTransformer(
        [("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), NUM_FEATURES),
         ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                               sparse_output=False), CAT_FEATURES)],
        verbose_feature_names_out=False)
    prep.set_output(transform="pandas")
    params = {**BEST_RF_PARAMS, **rf_params}
    return Pipeline([
        ("geo", GeoImputer()),
        ("prep", prep),
        ("clf", RandomForestClassifier(random_state=SEED, **params)),
    ])
