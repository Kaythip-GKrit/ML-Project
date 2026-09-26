# -*- coding: utf-8 -*-
"""
app.py — เว็บแอปทำนายความเสี่ยงการเสียชีวิตจากอุบัติเหตุ (Streamlit)

รันบนเครื่อง:   streamlit run app.py
เปิดเบราว์เซอร์ที่ http://localhost:8501
"""
import datetime as dt

import pandas as pd
import streamlit as st

from predict import FatalityPredictor

st.set_page_config(page_title="ประเมินความเสี่ยงอุบัติเหตุ", page_icon="🚦", layout="centered")

VEHICLE_TH = {
    "motorcycle": "รถจักรยานยนต์", "motorized tricycle": "รถสามล้อเครื่อง",
    "three-wheeled vehicle": "รถสามล้อ", "bicycle": "จักรยาน", "pedestrian": "คนเดินเท้า",
    "private/passenger car": "รถยนต์นั่งส่วนบุคคล", "van": "รถตู้",
    "passenger pickup truck": "รถกระบะโดยสาร", "4-wheel pickup truck": "รถกระบะ 4 ล้อ",
    "6-wheel truck": "รถบรรทุก 6 ล้อ", "7-10-wheel truck": "รถบรรทุก 7-10 ล้อ",
    "large truck with trailer": "รถบรรทุกพ่วง", "large passenger vehicle": "รถโดยสารขนาดใหญ่",
    "tractor/agricultural vehicle": "รถไถ/รถการเกษตร", "other": "อื่น ๆ",
}
AGENCY_TH = {
    "department of highways": "กรมทางหลวง",
    "department of rural roads": "กรมทางหลวงชนบท",
    "expressway authority of thailand": "การทางพิเศษแห่งประเทศไทย",
}
ACCIDENT_TH = {
    "rollover/fallen on straight road": "พลิกคว่ำ/ล้มบนทางตรง",
    "rollover/fallen on curved road": "พลิกคว่ำ/ล้มบนทางโค้ง",
    "rear-end collision": "ชนท้าย", "head-on collision (not overtaking)": "ชนประสานงา",
    "collision at intersection corner": "ชนบริเวณทางแยก", "pedestrian collision": "ชนคนเดินเท้า",
    "collision with obstruction (on road surface)": "ชนสิ่งกีดขวางบนถนน",
    "side collision": "ชนด้านข้าง", "collision during overtaking": "ชนขณะแซง",
    "turning/retreating collision": "ชนขณะเลี้ยว/ถอยหลัง", "other": "อื่น ๆ",
}
WEATHER_TH = {
    "clear": "แจ่มใส", "rainy": "ฝนตก", "foggy": "หมอก", "dark": "มืด/แสงสว่างไม่เพียงพอ",
    "natural disaster": "ภัยธรรมชาติ", "land slide": "ดินถล่ม", "other": "อื่น ๆ",
}
ROAD_TH = {
    "straight road": "ทางตรง", "wide curve": "ทางโค้งกว้าง", "sharp curve": "ทางโค้งหักศอก",
    "t-intersection": "ทางแยกรูปตัว T", "four-way intersection": "ทางแยก 4 แยก",
    "y-intersection": "ทางแยกรูปตัว Y", "roundabout": "วงเวียน", "u-turn area": "จุดกลับรถ",
    "grade-separated intersection/ramps": "ทางแยกต่างระดับ/ทางขึ้นลง", "merge lane": "ช่องทางเบี่ยง/รวมรถ",
    "lane-changing area": "จุดเปลี่ยนช่องจราจร", "bridge (across river/canal)": "สะพานข้ามแม่น้ำ/คลอง",
    "connecting to public/commercial area": "ทางเชื่อมเข้าพื้นที่สาธารณะ/ชุมชน/การค้า",
    "connecting to private area": "ทางเชื่อมเข้าพื้นที่ส่วนบุคคล",
    "connecting to school area": "ทางเชื่อมเข้าพื้นที่โรงเรียน",
    "motorcycle lane": "ช่องทางรถจักรยานยนต์", "pedestrian path": "ทางเท้า",
    "zebra crossing/pedestrian crossing": "ทางม้าลาย/ทางข้าม", "other": "อื่น ๆ",
}
REAR_END = "rear-end collision"


@st.cache_resource
def load_predictor():
    return FatalityPredictor("rf_model.joblib")


p = load_predictor()
k = p.known
if "weather_condition" not in k or "road_description" not in k:
    st.error("rf_model.joblib เป็นโมเดลรุ่นเก่า (ไม่มี weather_condition / road_description) "
             "กรุณารัน `python train_model.py` ใหม่ก่อน")
    st.stop()

st.title("🚦 ประเมินความเสี่ยงการเสียชีวิตจากอุบัติเหตุทางถนน")
st.caption("โมเดล Random Forest ฝึกจากข้อมูลอุบัติเหตุ พ.ศ. 2562–2565 (81,637 เหตุการณ์) "
           "ใช้ประกอบการคัดกรองเท่านั้น ไม่ใช่การตัดสินแทนเจ้าหน้าที่")

tab1, tab2, tab3 = st.tabs(["ทำนายทีละเหตุการณ์", "ทำนายจากไฟล์ CSV", "เกี่ยวกับโมเดล"])

# --------------------------------------------------------------------------- tab 1
with tab1:
    # ไม่ใช้ st.form เพราะจำนวนยานพาหนะต้องเปลี่ยนเงื่อนไขทันทีที่เลือกลักษณะการเกิดอุบัติเหตุ
    c1, c2 = st.columns(2)
    date = c1.date_input("วันที่เกิดเหตุ", dt.date(2022, 11, 19))
    time = c2.time_input("เวลาที่เกิดเหตุ", dt.time(23, 10))
    vehicle = st.selectbox("ประเภทยานพาหนะของผู้ประสบเหตุ", k["vehicle_type"],
                           format_func=lambda v: VEHICLE_TH.get(v, v),
                           index=k["vehicle_type"].index("motorcycle"))
    accident = st.selectbox("ลักษณะการเกิดอุบัติเหตุ", k["accident_type"],
                            format_func=lambda v: ACCIDENT_TH.get(v, v))

    # ชนท้ายต้องมียานพาหนะอย่างน้อย 2 คัน
    min_veh = 2 if accident == REAR_END else 1
    st.session_state.setdefault("n_veh", 2)
    if st.session_state["n_veh"] < min_veh:
        st.session_state["n_veh"] = min_veh
    n_veh = st.number_input("จำนวนยานพาหนะที่เกี่ยวข้อง", min_value=min_veh, max_value=30, key="n_veh",
                            help="การชนท้ายต้องมียานพาหนะอย่างน้อย 2 คัน" if accident == REAR_END else None)
    if accident == REAR_END:
        st.caption("⚠️ เลือก “ชนท้าย” แล้ว จำนวนยานพาหนะต้องเป็น 2 คันขึ้นไป")

    c7, c8 = st.columns(2)
    weather = c7.selectbox("สภาพอากาศ", k["weather_condition"], format_func=lambda v: WEATHER_TH.get(v, v),
                           index=k["weather_condition"].index("clear"))
    road = c8.selectbox("ลักษณะถนน", k["road_description"], format_func=lambda v: ROAD_TH.get(v, v),
                        index=k["road_description"].index("straight road"))
    c3, c4 = st.columns(2)
    province = c3.selectbox("จังหวัด", k["province"], index=k["province"].index("Surin"))
    agency = c4.selectbox("หน่วยงานที่รับผิดชอบถนน", k["agency"], format_func=lambda v: AGENCY_TH.get(v, v))
    use_coord = st.checkbox("ระบุพิกัด (ถ้าไม่ระบุ จะใช้ค่ากลางของจังหวัด)")
    c5, c6 = st.columns(2)
    lat = c5.number_input("Latitude", value=14.88, format="%.5f", disabled=not use_coord)
    lon = c6.number_input("Longitude", value=103.49, format="%.5f", disabled=not use_coord)
    submitted = st.button("ประเมินความเสี่ยง", type="primary")

    if submitted and accident == REAR_END and n_veh < 2:
        st.error("การชนท้ายต้องมียานพาหนะที่เกี่ยวข้องอย่างน้อย 2 คัน")
    elif submitted:
        record = {
            "incident_datetime": f"{date} {time.strftime('%H:%M')}",
            "province_en": province, "agency": agency, "vehicle_type": vehicle,
            "accident_type": accident, "number_of_vehicles_involved": int(n_veh),
            "weather_condition": weather, "road_description": road,
            "latitude": lat if use_coord else None, "longitude": lon if use_coord else None,
        }
        r = p.predict_one(record)
        prob = r["prob_fatal"]
        if r["prediction"] == "Fatal":
            st.error(f"### ความเสี่ยงสูง — มีโอกาสมีผู้เสียชีวิต {prob:.1%}")
        else:
            st.success(f"### ความเสี่ยงต่ำกว่าเกณฑ์ — มีโอกาสมีผู้เสียชีวิต {prob:.1%}")
        st.progress(min(prob, 1.0))
        st.caption(f"เกณฑ์ตัดสิน (threshold) = {r['threshold']:.3f} | "
                   f"ค่าเฉลี่ยของทั้งชุดข้อมูล = 12.4%")

# --------------------------------------------------------------------------- tab 2
with tab2:
    st.write("อัปโหลดไฟล์ CSV ที่มีคอลัมน์ `incident_datetime, province_en, agency, vehicle_type, "
             "accident_type, number_of_vehicles_involved, weather_condition, road_description` "
             "(ใส่ `latitude, longitude` ด้วยได้)")
    up = st.file_uploader("เลือกไฟล์ CSV", type="csv")
    if up is not None:
        try:
            result = p.predict(pd.read_csv(up))
            st.dataframe(result, use_container_width=True)
            st.write(result["prediction"].value_counts())
            st.download_button("ดาวน์โหลดผลลัพธ์", result.to_csv(index=False).encode("utf-8-sig"),
                               "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"อ่านไฟล์ไม่สำเร็จ: {e}")

# --------------------------------------------------------------------------- tab 3
with tab3:
    m = p.info["test_metrics"]
    st.write(f"**อัลกอริทึม:** Random Forest ({p.info['params']['n_estimators']} ต้น) | "
             f"**ฝึกด้วย:** {p.info['trained_on']}")
    c = st.columns(4)
    c[0].metric("Recall", f"{m['recall']:.3f}")
    c[1].metric("Precision", f"{m['precision']:.3f}")
    c[2].metric("F1", f"{m['f1']:.3f}")
    c[3].metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    st.info("ข้อจำกัด: โมเดลเรียนจากข้อมูลที่บันทึกโดย 3 หน่วยงาน ไม่มีข้อมูลความเร็ว การสวมหมวก/คาดเข็มขัด "
            "หรืออายุผู้ประสบเหตุ จึงพลาดเหตุ Fatal ได้เกือบครึ่งหนึ่ง และผลเป็นความสัมพันธ์ทางสถิติ ไม่ใช่เหตุและผล")
