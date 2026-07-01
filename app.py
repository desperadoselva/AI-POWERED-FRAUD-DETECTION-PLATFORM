import streamlit as st
import pandas as pd

#from predict import predict_transaction

# AI Explanation
# (Will connect in Part 2)
# from ai_agent.fraud_agent import explain


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(

    page_title="AI Fraud Detection Platform",

    page_icon="🛡️",

    layout="wide"

)


# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown("""

<style>

.main{

background-color:#F8F9FA;

}

.metric{

background:#FFFFFF;

padding:15px;

border-radius:12px;

box-shadow:2px 2px 10px lightgray;

}

</style>

""",unsafe_allow_html=True)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.image(
"https://img.icons8.com/color/96/shield.png",
width=80
)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

"Select",

[

"Dashboard",

"Single Prediction",

"About"

]

)


# --------------------------------------------------
# Dashboard
# --------------------------------------------------

if page=="Dashboard":

    st.title("🛡️ AI Powered Fraud Detection Platform")

    st.write("---")

    col1,col2,col3,col4=st.columns(4)

    col1.metric(

        "Transactions",

        "12,540"

    )

    col2.metric(

        "Fraud Cases",

        "265"

    )

    col3.metric(

        "Fraud Rate",

        "2.11%"

    )

    col4.metric(

        "Average Risk",

        "81%"

    )

    st.write("---")

    st.subheader("Platform Overview")

    st.info("""

This platform detects fraudulent transactions using

Machine Learning and explains suspicious activities

using Large Language Models.

Modules

✔ Random Forest

✔ Fraud Prediction

✔ AI Investigation

✔ Analytics Dashboard

✔ Report Generation

""")



# --------------------------------------------------
# Prediction
# --------------------------------------------------

elif page=="Single Prediction":

    st.title("Single Transaction Prediction")

    st.write("---")

    col1,col2=st.columns(2)

    with col1:

        amount=st.number_input(

            "Transaction Amount",

            min_value=0.0,

            value=5000.0

        )

        hour=st.slider(

            "Transaction Hour",

            0,

            23,

            12

        )

        transaction_type=st.selectbox(

            "Transaction Type",

            [

                0,

                1,

                2,

                3

            ],

            format_func=lambda x:{

                0:"Debit Card",

                1:"Credit Card",

                2:"UPI",

                3:"Wallet"

            }[x]

        )

        merchant=st.selectbox(

            "Merchant Category",

            [

                0,

                1,

                2,

                3,

                4,

                5

            ],

            format_func=lambda x:{

                0:"Grocery",

                1:"Retail",

                2:"Restaurant",

                3:"Electronics",

                4:"Travel",

                5:"International"

            }[x]

        )

    with col2:

        device=st.selectbox(

            "Device Type",

            [

                0,

                1,

                2,

                3

            ],

            format_func=lambda x:{

                0:"Unknown",

                1:"Mobile",

                2:"Desktop",

                3:"Tablet"

            }[x]

        )

        age=st.slider(

            "Customer Age",

            18,

            70,

            30

        )

        previous_fraud=st.selectbox(

            "Previous Fraud",

            [

                0,

                1

            ],

            format_func=lambda x:{

                0:"No",

                1:"Yes"

            }[x]

        )

    st.write("")

    if st.button(

        "Predict Fraud",

        use_container_width=True

    ):

        transaction={

            "Amount":amount,

            "Transaction_Hour":hour,

            "Transaction_Type":transaction_type,

            "Merchant_Category":merchant,

            "Device_Type":device,

            "Customer_Age":age,

            "Previous_Fraud":previous_fraud

        }

        result=predict_transaction(transaction)

        st.write("---")

        col1,col2,col3=st.columns(3)

        col1.metric(

            "Prediction",

            result["Prediction"]

        )

        col2.metric(

            "Risk Score",

            f'{result["Risk Score"]}%'

        )

        col3.metric(

            "Risk Level",

            result["Risk Level"]

        )

        st.write("---")

        if result["Prediction"]=="Fraudulent":

            st.error("⚠ Fraudulent Transaction")

        else:

            st.success("Transaction Looks Safe")

        st.subheader("Transaction Summary")

        st.dataframe(

            pd.DataFrame(

                [transaction]

            ),

            use_container_width=True

        )

        st.write("---")

        st.subheader("AI Investigation Report")

        # Part 2

        # report=explain(transaction,result)

        report=f"""

Prediction : {result['Prediction']}

Risk Score : {result['Risk Score']}%

Risk Level : {result['Risk Level']}

The Machine Learning model detected abnormal
transaction behaviour.

The AI Investigation module will generate a
complete explanation in Part 2.

"""

        st.info(report)


# --------------------------------------------------
# About
# --------------------------------------------------

else:

    st.title("About")

    st.write("---")

    st.markdown("""

### AI Powered Fraud Detection Platform

Version : 1.0

Technology Stack

- Python

- Streamlit

- Scikit-Learn

- LangChain

- Groq

- Plotly

- MongoDB

Features

✔ Fraud Prediction

✔ Risk Analysis

✔ AI Investigation

✔ Dashboard

✔ Reports

✔ Batch Prediction

""")