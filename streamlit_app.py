import streamlit as st

# Configure the page
st.set_page_config(
    page_title="Dictionary Classification App",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Dictionary Classification App")

st.markdown("### Welcome to the Dictionary Classification Tool")

st.write("""
This application helps you perform dictionary-based text classification and evaluate model performance with comprehensive metrics.
""")

# App Features
st.markdown("### 🚀 Key Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **📊 Dictionary Classification**
    - Upload your text data and classification dictionaries
    - Automatically classify text based on keyword matching
    - Support for multiple classification categories
    """)
    
    st.markdown("""
    **📈 Performance Metrics**
    - Calculate F1 scores for classification accuracy
    - Precision and recall metrics
    - Confusion matrices and detailed performance reports
    """)

with col2:
    st.markdown("""
    **🔗 CSV Join Feature**
    - Merge multiple CSV files seamlessly
    - Combine classification results with original data
    - Export processed results
    """)
    
    st.markdown("""
    **📋 Easy to Use**
    - Intuitive web interface
    - Step-by-step workflow
    - Real-time results and visualizations
    """)

# Instructions link
st.markdown("---")
st.markdown("### 📖 Need Help?")
st.info("For detailed instructions and examples, check out our comprehensive documentation.")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.link_button(
        "📋 View Complete Instructions", 
        "https://clean-divan-40d.notion.site/APP-List-2458c7e764f1808db092eef2affb4cdd",
        use_container_width=True
    )

# Getting started
st.markdown("---")
st.markdown("### 🎯 Getting Started")
st.write("""
1. **Upload your data**: Start by uploading your text data in CSV format
2. **Define your dictionary**: Create or upload classification dictionaries
3. **Run classification**: Let the app classify your text automatically
4. **Analyze results**: Review F1 scores, precision, recall, and other metrics
5. **Export results**: Download your classified data and performance reports
""")

st.success("Ready to get started? Use the sidebar to navigate to different features!")
