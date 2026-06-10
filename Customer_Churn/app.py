import pandas as pd
import joblib
import streamlit as st

# ---------------- LOAD DATA ----------------


@st.cache_data
def load_data():
    return pd.read_csv("customer_data.csv")


@st.cache_resource
def load_model():
    model = joblib.load("churn_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    return model, preprocessor


df = load_data()
model, preprocessor = load_model()
results_df = pd.read_csv("final_result_df.csv")

# ---------------- SEGMENT & RISK HELPERS ----------------

SEGMENT_DESC = {
    " Loyal Advocates": {
        "color": "success",
        "desc": "Long-term customers with low charges and very low churn. Your most stable base.",
        "stats": "Avg Tenure: 44 months | Avg Charges: $29 | Churn Rate: 5%",
    },
    "Leaving Risk": {
        "color": "error",
        "desc": "New customers on high charges with no long-term commitment. Highest churn risk.",
        "stats": "Avg Tenure: 11 months | Avg Charges: $62 | Churn Rate: 42%",
    },
    "High Value": {
        "color": "info",
        "desc": "Long-tenured, high-spending customers. Revenue backbone — worth actively protecting.",
        "stats": "Avg Tenure: 57 months | Avg Charges: $91 | Churn Rate: 17%",
    },
}


def get_risk_level(prob):
    if prob >= 0.70:
        return "🔴 Critical", "error"
    elif prob >= 0.50:
        return "🟠 High", "warning"
    elif prob >= 0.30:
        return "🟡 Medium", "warning"
    else:
        return "🟢 Low", "success"


def get_recommendations(row):
    prob = row.get("Churn_Probability", 0)
    segment = str(row.get("Segment", ""))
    contract = str(row.get("Contract", "")).lower()
    internet = str(row.get("Internet Service", "")).lower()
    tech = row.get("Tech Support", 0)
    tenure = row.get("Tenure Months", 0)

    actions = []

    # ── Discount & Offer Actions ──────────────────────────────────────────
    if prob >= 0.70:
        actions.append(
            "🎁 **Immediate Retention Offer** — Send a personalised 15–20% discount on the next 3 months to prevent imminent churn.")
    elif prob >= 0.50:
        actions.append(
            "🎟️ **Loyalty Reward** — Offer a complimentary service upgrade (e.g., free streaming add-on) as a goodwill gesture.")
    elif prob >= 0.30:
        actions.append(
            "💸 **Early Renewal Discount** — Offer 10% off if they renew or upgrade their plan within the next 30 days.")

    # ── Contract Upgrade Actions ──────────────────────────────────────────
    if "month-to-month" in contract and prob >= 0.40:
        actions.append("📄 **Contract Upgrade Pitch** — Actively offer a 1-year or 2-year plan with a locked-in rate. Month-to-month customers churn at 4× the rate of annual contract holders.")
    elif "one year" in contract and prob >= 0.30:
        actions.append(
            "📄 **Upgrade to 2-Year Contract** — Offer one free month to lock in a 2-year plan and significantly reduce churn risk.")

    # ── Support & Service Actions ─────────────────────────────────────────
    if tech == 0 and prob >= 0.35:
        actions.append(
            "🛠️ **Offer Tech Support Add-On** — Customers without tech support churn at 2× the rate. Offer a 30-day free trial to demonstrate value.")
    if "fiber optic" in internet and prob >= 0.40:
        actions.append("📡 **Fiber Optic Service Review** — Fiber optic customers have the highest churn rate (42%). Proactively check service quality and offer a speed upgrade or bill credit if issues exist.")
    if tenure < 12 and prob >= 0.40:
        actions.append(
            "👋 **New Customer Check-In Call** — Schedule a proactive support call. Early-tenure customers who feel supported are significantly less likely to churn.")

    # ── Segment-Specific Actions ──────────────────────────────────────────
    if "Flight Risk" in segment:
        actions.append(
            "⚡ **Priority Escalation** — Flag for a personal outreach call from the retention team within 48 hours.")
    elif "High Value" in segment:
        actions.append(
            "👑 **VIP Programme Enrolment** — Enrol in a premium loyalty programme. High-value customers respond strongly to exclusivity and recognition.")
    elif "Loyal Advocates" in segment:
        actions.append(
            "💌 **Referral Programme Invite** — Low churn risk but high satisfaction. Invite them to refer friends for bill credits — excellent ROI.")

    # ── Fallback ──────────────────────────────────────────────────────────
    if not actions:
        actions.append(
            "✅ **No Immediate Action Required** — Customer appears stable. Continue standard engagement and monitor quarterly.")

    return actions


# ---------------- CONFIG ----------------
st.set_page_config(page_title="Customer Intelligence System", layout="wide")
st.title("📊 Customer Intelligence & Retention System")

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    ["📊 Dashboard", "👥 Segmentation", "⚠️ Churn Prediction",
        "🧠 Model Comparison", "🎯 Recommendations"],
)

# ---------------- DASHBOARD ----------------
if page == "📊 Dashboard":
    st.header("📊 Business Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['Churn'].mean()*100:.1f}%")
    col3.metric("Avg Monthly Charges", f"${df['Monthly Charges'].mean():.2f}")
    col4.metric("Critical Risk Customers", int(
        (df["Churn_Probability"] >= 0.70).sum()))

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("👥 Segment Distribution")
        st.bar_chart(df["Segment"].value_counts())
    with col6:
        st.subheader("⚠️ Risk Level Distribution")
        df["_risk_label"] = df["Churn_Probability"].apply(
            lambda p: get_risk_level(p)[0])
        st.bar_chart(df["_risk_label"].value_counts())

    st.markdown("---")
    st.subheader("💡 Key Insights")
    c1, c2, c3 = st.columns(3)
    c1.info("📄 Month-to-month customers churn at **42%** vs 3% for 2-year contracts.")
    c2.warning("📡 Fiber optic customers have the **highest churn rate** at 42%.")
    c3.success(
        "🛠️ Adding Tech Support **halves** the churn rate for at-risk customers.")


# ---------------- SEGMENTATION ----------------
elif page == "👥 Segmentation":
    st.header("👥 Customer Segmentation")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📍 Cluster Visualization")
        st.scatter_chart(df, x="PCA1", y="PCA2", color="Segment")
    with col2:
        st.subheader("📊 Segment Summary")
        seg_summary = df.groupby("Segment").agg(
            Customers=("Churn", "count"),
            Avg_Churn_Prob=("Churn_Probability", "mean"),
            Actual_Churn_Rate=("Churn", "mean"),
        ).round(3)
        st.dataframe(seg_summary)

    st.markdown("---")
    st.subheader("📖 Segment Profiles")
    for name, info in SEGMENT_DESC.items():
        with st.expander(name, expanded=True):
            getattr(st, info["color"])(info["desc"])
            st.caption(info["stats"])


# ---------------- CHURN PREDICTION ----------------
elif page == "⚠️ Churn Prediction":
    st.header("⚠️ Churn Prediction")
    st.subheader("📥 Enter Customer Details")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (Months)", 0, 72, 12)
        monthly = st.slider("Monthly Charges ($)", 0, 150, 50)
        contract = st.selectbox(
            "Contract", ["Month-to-month", "One year", "Two year"])
    with col2:
        internet = st.selectbox("Internet Service", [
                                "DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        tech_input = st.selectbox("Tech Support", ["Yes", "No"])
        tech = 1 if tech_input == "Yes" else 0

    total_charges = tenure * monthly
    st.caption(
        f"📝 Total Charges estimated as tenure × monthly = **${total_charges}**")

    if tenure == 0 or monthly == 0:
        st.warning(
            "⚠️ Tenure or Monthly Charges is 0 — prediction may not be reliable.")

    if st.button("Predict Churn"):
        input_df = df.drop(
            columns=["Churn", "Segment", "Actions", "Risk_Level", "PCA1", "PCA2",
                     "Churn_Probability", "_risk_label"],
            errors="ignore"
        ).iloc[0:1].copy()

        input_df["Tenure Months"] = tenure
        input_df["Monthly Charges"] = monthly
        input_df["Total Charges"] = total_charges
        input_df["Contract"] = contract
        input_df["Internet Service"] = internet
        input_df["Payment Method"] = payment
        input_df["Tech Support"] = tech

        input_processed = preprocessor.transform(input_df)
        prob = model.predict_proba(input_processed)[0][1]
        risk_label, risk_color = get_risk_level(prob)

        st.markdown("---")
        st.subheader("📊 Prediction Result")
        c1, c2 = st.columns(2)
        c1.metric("Churn Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Level", risk_label)
        getattr(st, risk_color)(
            f"This customer is classified as **{risk_label}** risk.")
        st.progress(float(prob))

        st.markdown("---")
        st.subheader("🎯 Suggested Actions")
        mock_row = {
            "Churn_Probability": prob,
            "Segment": "🔴 Flight Risk" if prob > 0.5 else "💚 Loyal Advocates",
            "Contract": contract,
            "Internet Service": internet,
            "Tech Support": tech,
            "Tenure Months": tenure,
            "Monthly Charges": monthly,
        }
        for action in get_recommendations(mock_row):
            st.markdown(f"- {action}")


# ---------------- MODEL ----------------
elif page == "🧠 Model Comparison":
    st.header("🧠 Model Performance")
    st.dataframe(results_df.style.highlight_max(axis=0))

    st.markdown("---")
    st.subheader("✅ Why logisticRegression was selected")
    c1, c2, c3 = st.columns(3)
    c1.info("📈 Highest ROC-AUC across all tested models")
    c2.info("🔗 Captures non-linear feature interactions")
    c3.info("📊 Best generalisation on tabular business data")

    st.markdown("""
    > **Threshold tuned to 0.4** (vs default 0.5) — increases recall on churners from **50% → 67%**,
    catching more at-risk customers at the cost of a small precision drop.  
    In churn problems, a missed churner costs far more than a false alarm.
    """)


# ---------------- RECOMMENDATIONS ----------------
elif page == "🎯 Recommendations":
    st.header("🎯 Customer Action System")

    col_filter, _ = st.columns([1, 2])
    with col_filter:
        filter_segment = st.selectbox(
            "Filter by Segment", ["All"] + list(df["Segment"].unique()))

    filtered_df = df if filter_segment == "All" else df[df["Segment"]
                                                        == filter_segment]
    idx = st.selectbox("Select Customer", filtered_df.index)

    row = df.loc[idx]
    risk_label, risk_color = get_risk_level(row["Churn_Probability"])

    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("📋 Customer Profile")
        st.write(f"**Segment:** {row['Segment']}")
        st.write(f"**Risk Level:** {risk_label}")
        st.write(f"**Churn Probability:** {row['Churn_Probability']*100:.1f}%")
        st.write(f"**Contract:** {row.get('Contract', 'N/A')}")
        st.write(f"**Tenure:** {row.get('Tenure Months', 'N/A')} months")
        st.write(f"**Monthly Charges:** ${row.get('Monthly Charges', 'N/A')}")

    with col2:
        st.subheader("📊 Risk Gauge")
        getattr(st, risk_color)(
            f"**{risk_label}** — {row['Churn_Probability']*100:.1f}% churn probability")
        st.progress(float(row["Churn_Probability"]))
        st.caption("0% = No Risk  →  100% = Certain Churn")

    with col3:
        st.subheader("🎯 Recommended Actions")
        for action in get_recommendations(row):
            st.markdown(f"- {action}")

    st.markdown("---")
    st.info("💡 Actions are generated based on churn probability, segment, contract type, internet service, tech support status, and tenure.")
