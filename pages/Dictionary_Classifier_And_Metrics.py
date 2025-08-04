import streamlit as st
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import time

# Page configuration
st.set_page_config(
    page_title="Dictionary Validation Tool",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state with simple defaults
if 'keywords' not in st.session_state:
    st.session_state.keywords = [
        'limited', 'exclusive', 'premium', 'luxury', 'heritage',
        'timeless', 'craftsmanship', 'elegant', 'quality'
    ]

if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Helper functions
def classify_text(text, keywords):
    """Simple exact match classification"""
    if pd.isna(text):
        return 0, []
    
    text_lower = str(text).lower()
    matched_keywords = []
    
    for keyword in keywords:
        if re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower):
            matched_keywords.append(keyword)
    
    return len(matched_keywords), matched_keywords

def calculate_metrics(df):
    """Calculate basic metrics"""
    tp = len(df[(df['ground_truth'] == 1) & (df['prediction'] == 1)])
    fp = len(df[(df['ground_truth'] == 0) & (df['prediction'] == 1)])
    fn = len(df[(df['ground_truth'] == 1) & (df['prediction'] == 0)])
    tn = len(df[(df['ground_truth'] == 0) & (df['prediction'] == 0)])
    
    accuracy = (tp + tn) / len(df) if len(df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy * 100,
        'precision': precision * 100,
        'recall': recall * 100,
        'f1': f1 * 100,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
    }

# Main app
st.title("🎯 Dictionary Validation Tool")
st.markdown("**Test your keyword dictionaries against ground truth labels**")
st.markdown("---")

# Quick start with sample data
st.header("🚀 Quick Start")
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("📊 Load Sample Data", type="primary"):
        # Create sample data
        sample_data = {
            'text': [
                'Limited time offer - premium quality products at exclusive prices!',
                'Regular everyday products for your home and office needs',
                'Luxury heritage brand with timeless elegance and craftsmanship',
                'Standard item available in our regular catalog',
                'Exclusive members-only access to our premium collection',
                'Basic functionality with standard features included',
                'Elegant design meets superior quality in this luxury piece',
                'Simple and affordable option for budget-conscious buyers'
            ],
            'ground_truth': [1, 0, 1, 0, 1, 0, 1, 0]
        }
        
        st.session_state.sample_df = pd.DataFrame(sample_data)
        st.success("✅ Sample data loaded!")

with col2:
    st.info("👈 Click to load sample data and see the tool in action!")

# Data upload section
st.header("📁 Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose CSV file with 'text' and 'ground_truth' columns",
    type=['csv'],
    help="CSV should have a text column and a ground_truth column with 0/1 values"
)

# Use either uploaded file or sample data
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ File uploaded! {len(df)} rows loaded.")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

elif 'sample_df' in st.session_state:
    df = st.session_state.sample_df
    st.info("📊 Using sample data")

# Main interface when data is available
if df is not None:
    # Column selection
    st.subheader("🔧 Configure Columns")
    col_sel1, col_sel2 = st.columns(2)
    
    with col_sel1:
        text_columns = df.select_dtypes(include=['object']).columns.tolist()
        text_column = st.selectbox("Text column:", text_columns, 
                                  index=0 if 'text' in text_columns else 0)
    
    with col_sel2:
        numeric_columns = df.columns.tolist()
        gt_column = st.selectbox("Ground truth column:", numeric_columns,
                               index=numeric_columns.index('ground_truth') if 'ground_truth' in numeric_columns else 0)
    
    # Show data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df[[text_column, gt_column]].head(), use_container_width=True)
    
    # Show data stats
    total_rows = len(df)
    positive_cases = df[gt_column].sum()
    negative_cases = total_rows - positive_cases
    
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        st.metric("Total Rows", total_rows)
    with stat_col2:
        st.metric("Positive Cases", positive_cases)
    with stat_col3:
        st.metric("Negative Cases", negative_cases)
    
    st.markdown("---")
    
    # Dictionary management
    st.header("📚 Manage Keywords")
    
    col_dict1, col_dict2 = st.columns([2, 1])
    
    with col_dict1:
        # Show current keywords
        st.write("**Current Keywords:**")
        keyword_text = '\n'.join(st.session_state.keywords)
        
        updated_keywords = st.text_area(
            "Edit keywords (one per line):",
            value=keyword_text,
            height=150,
            help="Add or remove keywords. Each keyword should be on a separate line."
        )
        
        if st.button("🔄 Update Keywords"):
            new_keywords = [k.strip().lower() for k in updated_keywords.split('\n') if k.strip()]
            st.session_state.keywords = new_keywords
            st.success(f"✅ Updated! Now have {len(new_keywords)} keywords.")
            st.rerun()
    
    with col_dict2:
        st.metric("Total Keywords", len(st.session_state.keywords))
        
        # Quick add keywords
        new_keyword = st.text_input("Quick add keyword:")
        if st.button("➕ Add") and new_keyword:
            if new_keyword.lower() not in st.session_state.keywords:
                st.session_state.keywords.append(new_keyword.lower())
                st.success(f"Added '{new_keyword}'!")
                st.rerun()
            else:
                st.warning("Keyword already exists!")
    
    st.markdown("---")
    
    # Classification
    st.header("🎯 Run Classification")
    
    threshold = st.slider("Classification threshold (min keywords to classify as positive):", 
                         0, 5, 1, help="How many keywords need to match for a positive classification")
    
    if st.button("🚀 Classify All Texts", type="primary"):
        with st.spinner("Classifying texts..."):
            results = []
            
            for idx, row in df.iterrows():
                text = row[text_column]
                ground_truth = int(row[gt_column])
                
                match_count, matched_keywords = classify_text(text, st.session_state.keywords)
                prediction = 1 if match_count >= threshold else 0
                
                results.append({
                    'text': text,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'match_count': match_count,
                    'matched_keywords': ', '.join(matched_keywords),
                    'correct': ground_truth == prediction
                })
            
            st.session_state.predictions = pd.DataFrame(results)
        
        st.success("✅ Classification complete!")
    
    # Show results
    if st.session_state.predictions is not None:
        predictions_df = st.session_state.predictions
        
        st.markdown("---")
        st.header("📊 Results")
        
        # Calculate and show metrics
        metrics = calculate_metrics(predictions_df)
        
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        with metric_col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")
        with metric_col2:
            st.metric("Precision", f"{metrics['precision']:.1f}%")
        with metric_col3:
            st.metric("Recall", f"{metrics['recall']:.1f}%")
        with metric_col4:
            st.metric("F1 Score", f"{metrics['f1']:.1f}%")
        
        # Results tabs
        tab1, tab2, tab3 = st.tabs(["📋 All Results", "❌ Errors", "📊 Analysis"])
        
        with tab1:
            # Show all predictions
            display_df = predictions_df.copy()
            display_df['Status'] = display_df['correct'].map({True: '✅ Correct', False: '❌ Wrong'})
            
            st.dataframe(
                display_df[['text', 'ground_truth', 'prediction', 'match_count', 'matched_keywords', 'Status']],
                use_container_width=True
            )
            
            # Download button
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "classification_results.csv",
                "text/csv"
            )
        
        with tab2:
            # Show errors
            errors_df = predictions_df[predictions_df['correct'] == False]
            
            if len(errors_df) > 0:
                st.write(f"**Found {len(errors_df)} classification errors:**")
                
                for idx, row in errors_df.iterrows():
                    error_type = "False Positive" if row['prediction'] == 1 else "False Negative"
                    
                    with st.expander(f"{error_type}: {row['text'][:50]}..."):
                        st.write(f"**Text:** {row['text']}")
                        st.write(f"**Ground Truth:** {row['ground_truth']}")
                        st.write(f"**Prediction:** {row['prediction']}")
                        st.write(f"**Matches:** {row['matched_keywords'] if row['matched_keywords'] else 'None'}")
                        st.write(f"**Match Count:** {row['match_count']}")
            else:
                st.success("🎉 No classification errors! Perfect score!")
        
        with tab3:
            # Simple visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm_data = [
                    [metrics['tn'], metrics['fp']],
                    [metrics['fn'], metrics['tp']]
                ]
                
                fig, ax = plt.subplots(figsize=(6, 4))
                im = ax.imshow(cm_data, cmap='Blues')
                
                # Add text annotations
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, cm_data[i][j], ha="center", va="center", fontsize=20)
                
                ax.set_xticks([0, 1])
                ax.set_yticks([0, 1])
                ax.set_xticklabels(['Predicted 0', 'Predicted 1'])
                ax.set_yticklabels(['Actual 0', 'Actual 1'])
                ax.set_title('Confusion Matrix')
                
                st.pyplot(fig)
            
            with col_viz2:
                # Metrics bar chart
                st.subheader("Performance Metrics")
                metrics_data = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'Score': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
                })
                
                fig = px.bar(metrics_data, x='Metric', y='Score', 
                           title='Classification Performance',
                           color='Score', color_continuous_scale='viridis')
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

else:
    # Help section when no data is loaded
    st.header("📖 How to Use")
    st.markdown("""
    1. **Quick Start**: Click "Load Sample Data" to try the tool immediately
    2. **Upload Data**: Or upload your own CSV file with:
       - A text column (containing the text to classify)
       - A ground_truth column (with 0 for negative, 1 for positive)
    3. **Manage Keywords**: Add or edit keywords for your classification
    4. **Set Threshold**: Choose how many keyword matches = positive classification
    5. **Classify**: Run the classification and see results
    6. **Analyze**: Review metrics and errors to improve your keywords
    """)
    
    st.subheader("📋 Example CSV Format")
    example_df = pd.DataFrame({
        'text': [
            'Limited time premium offer!',
            'Regular everyday product',
            'Exclusive luxury item'
        ],
        'ground_truth': [1, 0, 1]
    })
    st.dataframe(example_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Dictionary Validation Tool")
