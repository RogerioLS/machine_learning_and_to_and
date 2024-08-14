import streamlit as st

def render_sidebar():
    st.sidebar.title("About")
    st.sidebar.info(
        """
        This app predicts real estate prices in California based on several features.
        """
    )
    st.sidebar.markdown("---")
    st.sidebar.title("Model Settings")
    # Add more settings if needed