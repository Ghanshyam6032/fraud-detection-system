# app.py - Advanced Streamlit Frontend

import streamlit as st
import requests
import os
from datetime import datetime

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Advanced Fraud Detector",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# API Configuration
# --------------------------------------------------

# API Configuration
API_URL = "https://fraud-detection-system-1-zn3l.onrender.com/predict"
HEALTH_CHECK_URL = "https://fraud-detection-system-1-zn3l.onrender.com/health"
# --------------------------------------------------
# Custom Styling
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 18px;
    font-weight: bold;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    background-color: #1E1E1E;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# Main Header
# --------------------------------------------------

st.title("💳 Advanced Credit Card Fraud Detection Dashboard")

st.markdown("""
AI-powered fraud detection system using:

- FastAPI Backend
- Streamlit Frontend
- Isolation Forest ML Model
""")

st.markdown("---")

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("⚙️ Application Information")

    st.info(
        "This application predicts whether a "
        "transaction is fraudulent or normal "
        "using Machine Learning."
    )

    st.write("### 🔗 API Endpoints")

    st.code(API_URL)
    st.code(HEALTH_CHECK_URL)

    st.markdown("---")

    st.subheader("🩺 API Health Check")

    if st.button("Check API Health"):

        try:

            health_response = requests.get(
                HEALTH_CHECK_URL,
                timeout=5
            )

            health_response.raise_for_status()

            health_data = health_response.json()

            st.success(
                f"✅ API Status: "
                f"{health_data.get('status')}"
            )

            st.info(
                health_data.get(
                    'message',
                    'API running'
                )
            )

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to API"
            )

        except requests.exceptions.Timeout:

            st.error(
                "⏰ API timeout"
            )

        except Exception as e:

            st.error(f"Error: {e}")

    st.markdown("---")

    st.write("### 👨‍💻 Developer")
    st.success("Ghanshyam Prajapati")

# --------------------------------------------------
# Input Section
# --------------------------------------------------

st.subheader("📝 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:

    amount = st.number_input(
        "💰 Transaction Amount ($)",
        min_value=0.0,
        value=100.0,
        format="%.2f",
        help="Enter transaction amount"
    )

with col2:

    merchant_id = st.number_input(
        "🏪 Merchant ID",
        min_value=1,
        value=12345,
        step=1,
        help="Enter Merchant ID"
    )

# --------------------------------------------------
# Predict Button
# --------------------------------------------------

predict_button = st.button(
    "🚀 Predict Transaction",
    type="primary"
)

# --------------------------------------------------
# Prediction Logic
# --------------------------------------------------

if predict_button:

    if amount <= 0:

        st.warning(
            "⚠️ Amount must be greater than 0"
        )

    else:

        payload = {
            "amount": amount,
            "merchant_id": merchant_id
        }

        st.markdown("---")

        st.subheader("📊 Prediction Results")

        with st.spinner(
            "Analyzing transaction..."
        ):

            try:

                response = requests.post(
                API_URL,
                json=payload,
                timeout=20)
                          

                response.raise_for_status()

                prediction_data = response.json()

                prediction_code = prediction_data.get(
                    'prediction_code'
                )

                status = prediction_data.get(
                    'status',
                    'Unknown'
                )

                anomaly_score = prediction_data.get(
                    'anomaly_score',
                    0
                )

                confidence = prediction_data.get(
                    'confidence',
                    0
                )

                request_id = prediction_data.get(
                    'request_id',
                    'N/A'
                )

                timestamp = prediction_data.get(
                    'timestamp',
                    'N/A'
                )

                user_ip = prediction_data.get(
                    'user_ip',
                    'N/A'
                )

                api_message = prediction_data.get(
                    'message',
                    'No message'
                )

                # --------------------------------------------------
                # Result Status
                # --------------------------------------------------

                if prediction_code == 1:

                    st.success(
                        f"✅ {status}"
                    )

                    st.balloons()

                else:

                    st.error(
                        f"🚨 {status}"
                    )

                # --------------------------------------------------
                # Metrics
                # --------------------------------------------------

                metric1, metric2, metric3 = st.columns(3)

                with metric1:

                    st.metric(
                        "Anomaly Score",
                        f"{anomaly_score:.4f}"
                    )

                with metric2:

                    st.metric(
                        "Confidence",
                        f"{confidence}%"
                    )

                with metric3:

                    st.metric(
                        "Merchant ID",
                        merchant_id
                    )

                st.markdown("---")

                # --------------------------------------------------
                # Additional Details
                # --------------------------------------------------

                st.info(
                    f"🆔 Request ID: {request_id}"
                )

                st.info(
                    f"🕒 Timestamp: {timestamp}"
                )

                st.info(
                    f"🌐 User IP: {user_ip}"
                )

                st.info(
                    f"📩 API Message: {api_message}"
                )

                # --------------------------------------------------
                # Notes
                # --------------------------------------------------

                st.markdown("""
### 📌 Important Note

This prediction is generated using an
Isolation Forest Machine Learning model.

- Lower anomaly score → Higher fraud possibility
- Higher anomaly score → Normal transaction

This system should be used as a support
tool and not as the final decision system.
""")

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI backend"
                )

                st.code(
                    "python -m uvicorn main:app --reload"
                )

            except requests.exceptions.Timeout:

                st.error(
                    "⏰ API request timed out"
                )

            except requests.exceptions.HTTPError as e:

                st.error(
                    f"🚨 HTTP Error: "
                    f"{e.response.status_code}"
                )

                st.code(e.response.text)

            except requests.exceptions.RequestException as e:

                st.error(
                    f"Request Error: {e}"
                )

            except Exception as e:

                st.error(
                    f"Unexpected Error: {e}"
                )

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Advanced Fraud Detection System using "
    "FastAPI + Streamlit + Machine Learning"
)
