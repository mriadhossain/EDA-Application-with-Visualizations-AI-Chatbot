import streamlit as st
import pandas as pd
import json

def update_data_summary():
    """Update the data summary in the session state."""
    st.session_state.data_summary = json.dumps({
        "numeric": st.session_state.df.describe(include=['number']).to_string(),
        "categorical": st.session_state.df.describe(include=['object', 'category']).to_string()
    })

def upload_data():
    """Handle data file upload and store in session state."""
    uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV or XLSX)", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        # Read the dataset and store it in session state if not already loaded
        if st.session_state.df is None:
            if uploaded_file.type == "text/csv":
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            update_data_summary()  # Update the data summary after loading the dataset
            print("Dataset loaded")
