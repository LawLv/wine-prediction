import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "artifacts/xgb_price_class_model.joblib"

@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)

artifact = load_artifact()
model = artifact["model"]
ui_features = artifact["ui_features"]
class_names = artifact.get("class_names", {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"})
price_bins = artifact.get("price_bins", None)

st.markdown(
    """
    <style>
    .stApp {
        background-color: #CEEAD6;
    }

    /* 主内容区域卡片感 */
    section.main > div {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.image("logo.png", width=160)
st.title("Wine Price Tier Predictor (Systembolaget)")

st.caption("Enter bottle-label style info and predict the price tier (Q1-Q4) within the filtered mainstream range.")

# ---- Build dropdown options from the model's training data categories ----
# We can extract categories from the OneHotEncoder learned categories_.
# But easiest & robust: let the user type if dropdown too big.
# Here: create dropdowns from encoder categories if available.

country = None
cat1 = None
cat2 = None

# try to get fitted encoder categories
try:
    preprocess = model.named_steps["preprocess"]
    # ColumnTransformer: find the "cat" pipeline then onehot
    cat_pipeline = preprocess.named_transformers_["cat"]
    ohe = cat_pipeline.named_steps["onehot"]
    cat_cols = preprocess.transformers_[1][2]  # assumes second transformer is ("cat", ..., cat_cols)
    categories = dict(zip(cat_cols, ohe.categories_))

    if "country" in categories:
        country = st.selectbox("Country", options=list(categories["country"]))
    else:
        country = st.text_input("Country", value="Unknown")

    if "categoryLevel1" in categories:
        cat1 = st.selectbox("Category Level 1", options=list(categories["categoryLevel1"]))
    else:
        cat1 = st.text_input("Category Level 1", value="Unknown")

    if "categoryLevel2" in categories:
        cat2 = st.selectbox("Category Level 2", options=list(categories["categoryLevel2"]))
    else:
        cat2 = st.text_input("Category Level 2", value="Unknown")

except Exception:
    # fallback if we can't access fitted encoder
    country = st.text_input("Country", value="Unknown")
    cat1 = st.text_input("Category Level 1", value="Unknown")
    cat2 = st.text_input("Category Level 2", value="Unknown")

alcohol = st.number_input("Alcohol Percentage", min_value=0.0, max_value=100.0, value=13.0, step=0.1)
volume = st.number_input("Volume (ml)", min_value=50, max_value=3000, value=750, step=50)

# vintage can be missing in data; allow empty
vintage_input = st.text_input("Vintage (year, optional)", value="")
is_organic = st.checkbox("Organic", value=False)

def parse_vintage(s: str):
    s = s.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        return None

vintage = parse_vintage(vintage_input)

if st.button("Predict price tier"):
    user_row = {
        "country": country,
        "categoryLevel1": cat1,
        "categoryLevel2": cat2,
        "alcoholPercentage": float(alcohol),
        "volume": int(volume),
        "vintage": vintage,
        "isOrganic": int(is_organic),
    }

    user_df = pd.DataFrame([user_row])

    # ensure expected columns exist and order matches training
    for col in ui_features:
        if col not in user_df.columns:
            user_df[col] = None
    user_df = user_df[ui_features]

    pred_class = int(model.predict(user_df)[0])

    label = class_names.get(pred_class, f"Class {pred_class}")

    if price_bins and 0 <= pred_class < len(price_bins):
        lo, hi = price_bins[pred_class]
        st.success(f"Predicted tier: {label}")
        st.info(f"Estimated price range: **{lo:.0f} – {hi:.0f} SEK** (within mainstream filtered products)")
    else:
        st.success(f"Predicted tier: {label}")

