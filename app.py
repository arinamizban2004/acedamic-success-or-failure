import json
import os

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCENARIOS = {
    "A": ["Admission grade", "Previous qualification (grade)", "Age at enrollment",
          "Gender", "Displaced", "Scholarship holder"],
}
SCENARIOS["B"] = SCENARIOS["A"] + [
    "Debtor", "Tuition fees up to date",
    "Curricular units 1st sem (enrolled)", "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
]
SCENARIOS["C"] = SCENARIOS["B"] + [
    "Curricular units 2nd sem (enrolled)", "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
]
SCENARIO_NAMES = {
    "A": "Entry data only",
    "B": "Entry + 1st semester",
    "C": "Entry + 1st and 2nd semester",
}


def path(*parts):
    return os.path.join(BASE_DIR, *parts)


def positive_shap(values):
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    if values.ndim == 3:
        return values[:, :, 1]
    return values

st.set_page_config(page_title="پیش‌بینی وضعیت تحصیلی", page_icon="🎓")

YES_NO = {0: "خیر", 1: "بله"}

NUMERIC = {
    "Admission grade": ("نمره پذیرش (۰ تا ۲۰۰)", 0.0, 200.0, 125.0, 0.5),
    "Previous qualification (grade)": ("نمره مدرک قبلی (۰ تا ۲۰۰)", 0.0, 200.0, 130.0, 0.5),
    "Age at enrollment": ("سن هنگام ورود", 16, 70, 19, 1),
    "Curricular units 1st sem (enrolled)": ("ترم اول: واحدهای ثبت‌نام‌شده", 0, 30, 6, 1),
    "Curricular units 1st sem (approved)": ("ترم اول: واحدهای قبول‌شده", 0, 30, 5, 1),
    "Curricular units 1st sem (grade)": ("ترم اول: معدل (۰ تا ۲۰)", 0.0, 20.0, 12.0, 0.1),
    "Curricular units 2nd sem (enrolled)": ("ترم دوم: واحدهای ثبت‌نام‌شده", 0, 30, 6, 1),
    "Curricular units 2nd sem (approved)": ("ترم دوم: واحدهای قبول‌شده", 0, 30, 5, 1),
    "Curricular units 2nd sem (grade)": ("ترم دوم: معدل (۰ تا ۲۰)", 0.0, 20.0, 12.0, 0.1),
}

BINARY = {
    "Gender": ("جنسیت", {0: "زن", 1: "مرد"}, 0),
    "Displaced": ("غیربومی (جابه‌جاشده)؟", YES_NO, 0),
    "Scholarship holder": ("بورسیه؟", YES_NO, 0),
    "Debtor": ("بدهکار؟", YES_NO, 0),
    "Tuition fees up to date": ("شهریه به‌روز پرداخت شده؟", YES_NO, 1),
}

MODES = {
    "success": "موفقیت تحصیلی (فارغ‌التحصیلی در برابر ترک تحصیل)",
    "probation": "مشروطی ترم دوم",
}


@st.cache_resource
def load_json(name):
    with open(path(name), encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource
def load_model(file_name):
    return joblib.load(path("models", file_name))


@st.cache_resource
def load_explainer(key):
    import shap
    rf = joblib.load(path("models", f"explainer_rf_{key}.pkl"))
    return shap.TreeExplainer(rf)


st.title("🎓 پیش‌بینی وضعیت تحصیلی دانشجو")

mode = st.radio("نوع پیش‌بینی", list(MODES.keys()), format_func=lambda m: MODES[m], horizontal=True)

if mode == "success":
    if not os.path.exists(path("model_info.json")):
        st.error("مدل پیدا نشد. ابتدا tune.py را اجرا کنید.")
        st.stop()
    info = load_json("model_info.json")
    available = [k for k in SCENARIOS if k in info]
    tag = st.radio("سناریوی پیش‌بینی", available,
                   format_func=lambda k: f"{k}: {SCENARIO_NAMES[k]}", horizontal=True)
    feats = SCENARIOS[tag]
    meta = info[tag]
    model_file = f"best_{tag}.pkl"
else:
    if not os.path.exists(path("probation_info.json")):
        st.error("مدل مشروطی پیدا نشد. ابتدا probation.py را اجرا کنید.")
        st.stop()
    meta = load_json("probation_info.json")
    tag = "P"
    feats = meta["features"]
    model_file = "probation_model.pkl"
    st.info(f"مشروطی یعنی معدل ترم دوم کمتر از {meta['threshold']:g}. "
            "مدل با اطلاعات ورود و ترم اول، مشروطی ترم دوم را پیش‌بینی می‌کند.")

st.caption(f"Model: {meta['model']} | Accuracy: {meta['accuracy']:.1%} | F1: {meta['f1']:.3f}")

values = {}
columns = st.columns(2)
for n, feat in enumerate(feats):
    box = columns[n % 2]
    widget_key = f"{mode}|{tag}|{feat}"
    if feat in NUMERIC:
        label, lo, hi, default, step = NUMERIC[feat]
        values[feat] = box.number_input(
            label, min_value=lo, max_value=hi, value=default, step=step, key=widget_key)
    else:
        label, options, default = BINARY[feat]
        option_list = list(options.keys())
        values[feat] = box.selectbox(
            label, option_list, index=option_list.index(default),
            format_func=lambda v, o=options: o[v], key=widget_key)

if st.button("🔮 پیش‌بینی", type="primary"):
    for sem in ("1st", "2nd"):
        enrolled = f"Curricular units {sem} sem (enrolled)"
        approved = f"Curricular units {sem} sem (approved)"
        if enrolled in values and values[approved] > values[enrolled]:
            st.warning("تعداد واحدهای قبول‌شده نمی‌تواند از واحدهای ثبت‌نام‌شده بیشتر باشد.")
            st.stop()

    x = pd.DataFrame([values])[feats]
    model = load_model(model_file)
    pred = int(model.predict(x)[0])
    idx = list(model.classes_).index(1)
    p = float(model.predict_proba(x)[0][idx])

    if mode == "success":
        if pred == 1:
            st.success("✅ پیش‌بینی: دانشجو موفق خواهد بود")
        else:
            st.error("⚠️ پیش‌بینی: دانشجو در معرض خطر ناموفقی است")
        st.metric("احتمال موفقیت", f"{p:.1%}")
    else:
        if pred == 1:
            st.error("⚠️ پیش‌بینی: دانشجو در ترم دوم در معرض مشروطی است")
        else:
            st.success("✅ پیش‌بینی: دانشجو در ترم دوم مشروط نخواهد شد")
        st.metric("احتمال مشروطی", f"{p:.1%}")
    st.progress(min(max(p, 0.0), 1.0))

    if mode == "success":
        try:
            explainer = load_explainer(tag)
            contrib = positive_shap(explainer.shap_values(x, check_additivity=False))[0]
            series = pd.Series(contrib, index=feats).sort_values()
            fig, ax = plt.subplots(figsize=(7, max(3.0, 0.4 * len(feats))))
            ax.barh(series.index, series.values,
                    color=["tomato" if v < 0 else "seagreen" for v in series.values])
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title("Feature contributions (SHAP, Random Forest surrogate)")
            ax.set_xlabel("Effect on success probability (green = increases)")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            st.caption("SHAP explanation not available. Run explain.py first.")

st.divider()
st.caption("این پیش‌بینی یک تخمین آماری است و جایگزین ارزیابی آموزشی نیست.")
