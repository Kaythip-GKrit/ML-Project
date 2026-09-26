# ระบบประเมินความเสี่ยงการเสียชีวิตจากอุบัติเหตุ (Random Forest) — ชุดสำหรับ Deploy

## ไฟล์ในชุดนี้

| ไฟล์ | หน้าที่ |
|---|---|
| `rf_features.py` | ทำความสะอาดข้อมูล สร้าง 21 Features และประกอบ Pipeline (**ต้องอยู่คู่กับไฟล์โมเดลเสมอ**) |
| `train_model.py` | Tuning ด้วย RandomizedSearchCV, เลือก threshold จาก CV, ประเมินผลชุดทดสอบ + Permutation Importance แล้วบันทึก `rf_model.joblib` |
| `rf_model.joblib` | โมเดลที่ฝึกแล้ว พร้อม threshold และรายการค่าที่โมเดลรู้จัก |
| `rf_model_v1_18feat.joblib`, `model_info_v1_18feat.json` | โมเดลรุ่นแรก (18 Features ไม่มีสภาพอากาศ/ลักษณะถนน) เก็บไว้เทียบผล |
| `rf_model_v2_22feat.joblib`, `model_info_v2_22feat.json` | โมเดลรุ่นที่ 2 (22 Features มี agency) เก็บไว้เทียบผล |
| `model_info.json` | ข้อมูลโมเดลและผลประเมินแบบอ่านได้ |
| `predict.py` | คลาส `FatalityPredictor` สำหรับเรียกใช้จากโค้ดอื่น และใช้ทำนายไฟล์ CSV จาก command line |
| `app.py` | เว็บแอป Streamlit (กรอกทีละเหตุการณ์ / อัปโหลด CSV) |
| `requirements.txt` | ไลบรารีที่ต้องติดตั้ง |

## ติดตั้ง

```bash
pip install -r requirements.txt
```

`scikit-learn` ต้องเป็นเวอร์ชัน **1.8.0** เท่ากับตอนฝึก ถ้าใช้เวอร์ชันอื่นให้รัน `train_model.py` ใหม่บนเครื่องนั้น

## วิธีใช้

**1) ฝึกโมเดลใหม่ (ไม่จำเป็น ถ้าใช้ `rf_model.joblib` ที่แนบมา)**
```bash
python train_model.py --csv thai_road_accident_2019_2022.csv          # ฝึกด้วย 80% (ตรงกับรายงาน)
python train_model.py --csv thai_road_accident_2019_2022.csv --full   # ฝึกด้วยข้อมูลทั้งหมด สำหรับใช้งานจริง
python train_model.py --csv thai_road_accident_2019_2022.csv --no-tune  # ข้าม Tuning (เร็วขึ้น) ใช้ BEST_RF_PARAMS
```
Tuning: RandomizedSearchCV 16 ชุด, 5-fold CV, เกณฑ์ PR-AUC ช่วงค่า `n_estimators` ∈ {150, 250, 350}, `max_depth` ∈ {12, 18, 25, None}, `min_samples_leaf` ∈ {1, 3, 5, 10, 20}, `max_features` ∈ {sqrt, 0.1, 0.2, 0.3}, `class_weight` ∈ {None, balanced_subsample}
ก่อนฝึกจะตัดเหตุชนท้ายที่บันทึกว่ามียานพาหนะ 0–1 คัน (98 แถว) ออก เพราะน่าจะบันทึกผิด
ผลบนชุดทดสอบ ค่าที่ Tuning ได้ และ Permutation Importance จะถูกพิมพ์ออกมาและบันทึกใน `model_info.json`
(ผลของโมเดลเดิม 18 Features: Accuracy 0.868, Precision 0.471, Recall 0.539, F1 0.503, ROC-AUC 0.844, PR-AUC 0.498)

**2) ทำนายจาก command line**
```bash
python predict.py --demo
python predict.py --csv new_accidents.csv --out predictions.csv
```

**3) เรียกใช้ในโค้ด Python**
```python
from predict import FatalityPredictor
p = FatalityPredictor("rf_model.joblib")
p.predict_one({
    "incident_datetime": "2022-11-19 23:10",
    "province_en": "Surin",
    "vehicle_type": "motorcycle",
    "accident_type": "head-on collision (not overtaking)",
    "number_of_vehicles_involved": 2,
    "weather_condition": "clear",
    "road_description": "straight road",
    "latitude": 14.88, "longitude": 103.49,   # ไม่บังคับ
})
# -> {'prob_fatal': ..., 'prediction': 'Fatal' / 'Non-fatal', 'threshold': ...}
```

**4) เปิดเว็บแอป**
```bash
streamlit run app.py
```

## ข้อมูลที่ต้องใช้ในการทำนาย

| คอลัมน์ | ตัวอย่าง | บังคับ |
|---|---|---|
| `incident_datetime` | `2022-11-19 23:10` | ใช่ |
| `province_en` | `Surin` | ใช่ |
| `vehicle_type` | `motorcycle` | ใช่ |
| `accident_type` | `rear-end collision` | ใช่ |
| `number_of_vehicles_involved` | `2` (ถ้าเป็นชนท้าย เว็บแอปบังคับ ≥ 2) | ใช่ |
| `weather_condition` | `clear`, `rainy`, `foggy`, `dark` | ใช่ |
| `road_description` | `straight road`, `wide curve`, `t-intersection` | ใช่ |
| `latitude`, `longitude` | `14.88`, `103.49` | ไม่ (ถ้าไม่มีจะใช้ค่ากลางของจังหวัด) |

ค่าที่ใช้ได้ของแต่ละคอลัมน์ดูได้ใน `model_info.json` ส่วน `known_values`
ถ้าส่งค่าที่โมเดลไม่เคยเห็น ระบบจะแจ้งเตือนและถือเป็นหมวดที่พบน้อย

## นำขึ้นออนไลน์ (Streamlit Community Cloud)

1. อัปโหลดทุกไฟล์ยกเว้นไฟล์ CSV ขึ้น GitHub repository
2. เข้า share.streamlit.io แล้วเลือก repository และไฟล์ `app.py`
3. กด Deploy

## ข้อจำกัด

- ใช้ได้หลังมีรายงานเหตุแล้ว ไม่ใช่การทำนายก่อนเกิดเหตุ
- ข้อมูลฝึกมาจาก 3 หน่วยงานและครอบคลุมเพียงส่วนหนึ่งของอุบัติเหตุทั้งประเทศ
- ไม่มีข้อมูลความเร็ว การสวมหมวก/คาดเข็มขัด หรืออายุผู้ประสบเหตุ จึงพลาดเหตุ Fatal ประมาณ 46%
- ควรใช้ประกอบการตัดสินใจของเจ้าหน้าที่ ไม่ใช่ตัดสินแทน
