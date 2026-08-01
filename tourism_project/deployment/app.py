"""
Streamlit application for Wellness Tourism Package purchase prediction.

Loads the model committed to the repository by the GitHub Actions pipeline,
collects customer and interaction details, and returns a purchase probability.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Resolve the model relative to this file so the app works regardless of the
# working directory Streamlit Cloud starts from
MODEL_PATH = Path(__file__).parent / "best_tourism_model_v1.joblib"

st.set_page_config(
    page_title="Wellness Tourism Package Prediction",
    page_icon="✈️",
    layout="centered",
)


@st.cache_resource
def load_model():
    """Load the trained pipeline once per session."""
    return joblib.load(MODEL_PATH)


st.title("Wellness Tourism Package — Purchase Prediction")
st.write(
    "Predicts how likely a customer is to purchase the Wellness Tourism Package, "
    "so the marketing team can prioritise follow-up effort. Enter the customer and "
    "interaction details below."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found at `{MODEL_PATH.name}`. "
        "Run the GitHub Actions pipeline so it trains and commits the model, "
        "then reboot this app."
    )
    st.stop()

model = load_model()

# ----------------------------------------------------------------- inputs
st.subheader("Customer details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=36, step=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox(
        "Marital status", ["Single", "Married", "Divorced", "Unmarried"]
    )
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"],
    )
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )

with col2:
    monthly_income = st.number_input(
        "Monthly income", min_value=1000, max_value=200000, value=22500, step=500
    )
    city_tier = st.selectbox(
        "City tier", [1, 2, 3],
        help="Tier 1 is the most developed, Tier 3 the least.",
    )
    passport = st.selectbox(
        "Holds a valid passport", ["No", "Yes"],
        help="The strongest single predictor in the model.",
    )
    own_car = st.selectbox("Owns a car", ["No", "Yes"])
    preferred_property_star = st.selectbox("Preferred hotel rating", [3, 4, 5])

st.subheader("Trip details")

col3, col4 = st.columns(2)

with col3:
    number_of_trips = st.number_input(
        "Average trips per year", min_value=1, max_value=25, value=3, step=1
    )
    number_of_person_visiting = st.number_input(
        "People travelling", min_value=1, max_value=10, value=3, step=1
    )

with col4:
    number_of_children_visiting = st.number_input(
        "Children under 5 travelling", min_value=0, max_value=5, value=0, step=1
    )

st.subheader("Interaction details")

col5, col6 = st.columns(2)

with col5:
    type_of_contact = st.selectbox(
        "Type of contact", ["Self Enquiry", "Company Invited"]
    )
    product_pitched = st.selectbox(
        "Product pitched",
        ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"],
    )
    duration_of_pitch = st.number_input(
        "Duration of pitch (minutes)", min_value=1, max_value=180, value=14, step=1
    )

with col6:
    number_of_followups = st.number_input(
        "Number of follow-ups", min_value=0, max_value=10, value=4, step=1
    )
    pitch_satisfaction_score = st.selectbox(
        "Pitch satisfaction score", [1, 2, 3, 4, 5], index=2
    )

threshold = st.slider(
    "Follow-up threshold",
    min_value=0.05, max_value=0.95, value=0.50, step=0.05,
    help=(
        "Customers scoring above this probability are flagged for follow-up. "
        "Lower it to catch more buyers at the cost of more wasted calls."
    ),
)

# ------------------------------------------------- assemble the input row
# Column names must match exactly what the pipeline was trained on
input_data = pd.DataFrame([{
    "Age": float(age),
    "TypeofContact": type_of_contact,
    "CityTier": int(city_tier),
    "DurationOfPitch": float(duration_of_pitch),
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": int(number_of_person_visiting),
    "NumberOfFollowups": float(number_of_followups),
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": float(preferred_property_star),
    "MaritalStatus": marital_status,
    "NumberOfTrips": float(number_of_trips),
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": int(pitch_satisfaction_score),
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": float(number_of_children_visiting),
    "Designation": designation,
    "MonthlyIncome": float(monthly_income),
}])

with st.expander("Review the input sent to the model"):
    st.dataframe(input_data.T.rename(columns={0: "value"}), use_container_width=True)

# ----------------------------------------------------------------- predict
if st.button("Predict", type="primary", use_container_width=True):

    probability = float(model.predict_proba(input_data)[0, 1])
    will_purchase = probability >= threshold

    st.subheader("Prediction")

    metric_col, verdict_col = st.columns([1, 2])
    metric_col.metric("Purchase probability", f"{probability:.1%}")

    if will_purchase:
        verdict_col.success(
            f"**Likely to purchase** — above the {threshold:.0%} threshold. "
            "Prioritise this customer for follow-up."
        )
    else:
        verdict_col.warning(
            f"**Unlikely to purchase** — below the {threshold:.0%} threshold. "
            "Deprioritise, or revisit if capacity allows."
        )

    st.progress(probability)
    st.caption(
        "Probabilities come from an XGBoost pipeline retrained automatically by GitHub "
        "Actions on every push to main. Use the score to rank leads rather than as a "
        "guarantee of outcome."
    )
