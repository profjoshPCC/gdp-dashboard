import streamlit as st
import pandas as pd
import re
import json
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import time

# Page configuration
st.set_page_config(
    page_title="Dictionary-Driven Text Classifier",
    page_icon="🧠",
    layout="wide"
)

# Initialize session state
def init_session_state():
    if 'dictionaries' not in st.session_state:
        st.session_state.dictionaries = {
            'urgency_marketing': [
                'limited', 'limited time', 'limited run', 'limited edition', 'order now',
                'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
                'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
                'expires soon', 'final hours', 'almost gone', 'flash sale', 'ending soon'
            ],
            'exclusive_marketing': [
                'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
                'members only', 'vip', 'special access', 'invitation only',
                'premium', 'privileged', 'limited access', 'select customers',
                'insider', 'private sale', 'early access', 'elite', 'platinum'
            ],
            'classic_timeless_luxury': [
                'heritage', 'timeless', 'craftsmanship', 'elegant', 'quality',
                'premium', 'luxury', 'sophisticated', 'refined', 'artisan',
                'tradition', 'excellence', 'masterpiece', 'prestige', 'distinguished',
                'classic', 'enduring', 'authentic', 'exceptional', 'superior'
            ]
        }
    
    if 'current_tactic' not in st.session_state:
        st.session_state.current_tactic = 'classic_timeless_luxury'
    
    if 'csv_data' not in st.session_state:
        st.session_state.csv_data = None
    
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None

# Helper functions
def classify_text(text, keywords, match_mode='exact'):
    """Classify text based on keyword matches"""
    if pd.isna(text):
        return {'count': 0, 'matches': []}
    
    text_lower = str(text).lower()
    count = 0
    matched_keywords = []
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        
        if match_mode == 'exact':
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            matches = re.findall(pattern, text_lower)
            count += len(matches)
            if matches:
                matched_keywords.extend([keyword] * len(matches))
        elif match_mode == 'partial':
            if keyword_lower in text_lower:
                count += text_lower.count(keyword_lower)
                matched_keywords.extend([keyword] * text_lower.count(keyword_lower))
        elif match_mode == 'fuzzy':
            words_in_text = text_lower.split()
            for word in words_in_text:
                if keyword_lower in word or word in keyword_lower:
                    count += 1
                    matched_keywords.append(keyword)
                    break
    
    return {'count': count, 'matches': matched_keywords}

def process_classification(df, text_column, ground_truth_column, keywords, match_mode='exact', threshold=0):
    """Process the dataframe with classification"""
    results = []
    
    for idx, row in df.iterrows():
        text = row[text_column]
        ground_truth = int(row[ground_truth_column]) if pd.notna(row[ground_truth_column]) else 0
        
        classification_result = classify_text(text, keywords, match_mode)
        prediction = 1 if classification_result['count'] > threshold else 0
        
        result = {
            'id': idx,
            'text': text,
            'ground_truth': ground_truth,
            'prediction': prediction,
            'match_count': classification_result['count'],
            'matched_keywords': classification_result['matches'],
            'is_tp': ground_truth == 1 and prediction == 1,
            'is_fp': ground_truth == 0 and prediction == 1,
            'is_fn': ground_truth == 1 and prediction == 0,
            'is_tn': ground_truth == 0 and prediction == 0
        }
        
        results.append(result)
    
    return pd.DataFrame(results)

def calculate_metrics(predictions_df):
    """Calculate classification metrics"""
    tp = len(predictions_df[predictions_df['is_tp']])
    fp = len(predictions_df[predictions_df['is_fp']])
    fn = len(predictions_df[predictions_df['is_fn']])
    tn = len(predictions_df[predictions_df['is_tn']])
    
    accuracy = (tp + tn) / len(predictions_df) if len(predictions_df) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }

def generate_mock_keywords(tactic, sample_texts):
    """Generate mock keywords based on tactic (simulating AI generation)"""
    time.sleep(2)  # Simulate processing time
    
    keyword_pools = {
        'urgency_marketing': [
            'urgent', 'immediate', 'now', 'quickly', 'fast', 'rush', 'deadline',
            'limited time', 'expires', 'ends soon', 'last chance', 'hurry',
            'don\'t wait', 'act now', 'while supplies last', 'running out'
        ],
        'exclusive_marketing': [
            'exclusive', 'premium', 'vip', 'elite', 'special', 'private',
            'members only', 'invitation only', 'select', 'privileged',
            'luxury', 'high-end', 'boutique', 'bespoke', 'custom'
        ],
        'classic_timeless_luxury': [
            'heritage', 'timeless', 'classic', 'traditional', 'elegant',
            'sophisticated', 'refined', 'quality', 'craftsmanship', 'artisan',
            'premium', 'luxury', 'prestige', 'distinguished', 'excellence'
        ],
        'discount_marketing': [
            'sale', 'discount', 'off', 'save', 'deal', 'bargain', 'cheap',
            'reduced', 'clearance', 'promotion', 'special price', 'markdown'
        ]
    }
    
    # Return relevant keywords or generate generic ones
    if tactic.lower().replace(' ', '_') in keyword_pools:
        return keyword_pools[tactic.lower().replace(' ', '_')]
    else:
        # Generate based on sample texts (mock analysis)
        return ['quality', 'premium', 'excellent', 'professional', 'reliable']

def create_confusion_matrix_viz(predictions_df):
    """Create confusion matrix visualization"""
    y_true = predictions_df['ground_truth'].values
    y_pred = predictions_df['prediction'].values
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=['Negative', 'Positive'],
        y=['Negative', 'Positive'],
        title="Confusion Matrix"
    )
    
    return fig

def main():
    init_session_state()
    
    # Title and description
    st.title("🧠 Dictionary-Driven Text Classifier")
    st.markdown("**Build and validate keyword dictionaries against ground truth labels**")
    st.markdown("---")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Classification Settings")
        
        match_mode = st.selectbox(
            "Matching Mode:",
            ['exact', 'partial', 'fuzzy'],
            help="Exact: whole word matches only\nPartial: substring matches\nFuzzy: flexible matching"
        )
        
        confidence_threshold = st.slider(
            "Classification Threshold:",
            0, 5, 0,
            help="Minimum number of matches required to classify as positive"
        )
        
        st.markdown("---")
        st.header("📚 Dictionary Management")
        
        # Dictionary selection
        selected_dict = st.selectbox(
            "Active Dictionary:",
            list(st.session_state.dictionaries.keys())
        )
        
        st.session_state.current_tactic = selected_dict
        
        # Dictionary editing
        if st.checkbox("Edit Dictionary"):
            current_keywords = st.session_state.dictionaries[selected_dict]
            keywords_text = '\n'.join(current_keywords)
            
            updated_keywords = st.text_area(
                "Keywords (one per line):",
                value=keywords_text,
                height=200
            )
            
            if st.button("Update Dictionary"):
                new_keywords = [line.strip() for line in updated_keywords.split('\n') if line.strip()]
                st.session_state.dictionaries[selected_dict] = new_keywords
                st.success(f"Updated {selected_dict}!")
                st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Step 1: Define Tactic
        st.header("1️⃣ Define Tactic")
        tactic_description = st.text_area(
            "Describe your classification tactic:",
            value=f"{st.session_state.current_tactic}: language emphasizing the characteristics of this category",
            height=100,
            help="Describe what kind of language patterns you want to detect"
        )
        
        st.markdown("---")
        
        # Step 2: Upload Data
        st.header("2️⃣ Upload & Inspect Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing text data and ground truth labels"
        )
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.csv_data = df
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Column selection
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                
                col_sel1, col_sel2 = st.columns(2)
                
                with col_sel1:
                    text_column = st.selectbox(
                        "Select text column:",
                        text_columns,
                        help="Column containing the text to classify"
                    )
                
                with col_sel2:
                    ground_truth_column = st.selectbox(
                        "Select ground truth column:",
                        numeric_columns,
                        help="Column containing the true labels (0/1)"
                    )
                
                # Preview data
                if text_column and ground_truth_column:
                    st.subheader("📋 Data Preview")
                    preview_df = df[[text_column, ground_truth_column]].head()
                    st.dataframe(preview_df, use_container_width=True)
                    
                    # Data statistics
                    total_rows = len(df)
                    positive_cases = df[ground_truth_column].sum()
                    negative_cases = total_rows - positive_cases
                    
                    stat_col1, stat_col2, stat_col3 = st.columns(3)
                    with stat_col1:
                        st.metric("Total Rows", total_rows)
                    with stat_col2:
                        st.metric("Positive Cases", positive_cases)
                    with stat_col3:
                        st.metric("Negative Cases", negative_cases)
                
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        else:
            # Show sample data format
            st.info("👆 Upload a CSV file to get started!")
            st.subheader("Expected Data Format")
            sample_df = pd.DataFrame({
                'ID': [1, 2, 3, 4],
                'Statement': [
                    'Limited time offer - act now! Get 50% off!',
                    'Exclusive deal for VIP members only - premium access',
                    'Regular product description with quality features',
                    'Flash sale ending soon - don\'t miss out on this discount!'
                ],
                'ground_truth': [1, 1, 0, 1]
            })
            st.dataframe(sample_df, use_container_width=True)
        
        st.markdown("---")
        
        # Step 3: AI Generation
        st.header("3️⃣ AI-Powered Keyword Generation")
        
        if st.button("🚀 Generate Keywords with AI", type="primary"):
            if tactic_description and st.session_state.csv_data is not None:
                with st.spinner("🤖 AI is analyzing your tactic and generating keywords..."):
                    sample_texts = st.session_state.csv_data[text_column].dropna().head(10).tolist()
                    new_keywords = generate_mock_keywords(tactic_description, sample_texts)
                    
                    # Update dictionary
                    st.session_state.dictionaries[st.session_state.current_tactic] = new_keywords
                    
                st.success("✅ Keywords generated successfully!")
                st.rerun()
            else:
                st.error("Please define a tactic and upload data first")
    
    with col2:
        # Current Dictionary Display
        st.header("📚 Current Dictionary")
        st.subheader(f"{st.session_state.current_tactic}")
        
        current_keywords = st.session_state.dictionaries[st.session_state.current_tactic]
        
        # Show keywords in an expandable format
        with st.expander(f"Keywords ({len(current_keywords)})", expanded=True):
            for i, keyword in enumerate(current_keywords):
                st.write(f"• {keyword}")
        
        # Add custom keywords
        st.subheader("➕ Add Custom Keywords")
        new_keywords = st.text_area(
            "Enter keywords (one per line):",
            height=100,
            placeholder="Enter new keywords here..."
        )
        
        if st.button("Add Keywords"):
            if new_keywords.strip():
                additional_keywords = [kw.strip() for kw in new_keywords.split('\n') if kw.strip()]
                st.session_state.dictionaries[st.session_state.current_tactic].extend(additional_keywords)
                st.success(f"Added {len(additional_keywords)} keywords!")
                st.rerun()
        
        # Live coverage calculation
        if st.session_state.csv_data is not None and 'text_column' in locals():
            coverage_count = 0
            total_count = len(st.session_state.csv_data)
            
            for text in st.session_state.csv_data[text_column].dropna():
                result = classify_text(text, current_keywords, match_mode)
                if result['count'] > 0:
                    coverage_count += 1
            
            coverage_percentage = (coverage_count / total_count) * 100 if total_count > 0 else 0
            
            st.metric("Coverage", f"{coverage_percentage:.1f}%", 
                     help=f"{coverage_count} out of {total_count} texts match at least one keyword")
    
    # Classification Section
    if st.session_state.csv_data is not None and 'text_column' in locals() and 'ground_truth_column' in locals():
        st.markdown("---")
        st.header("4️⃣ Run Classification & Validation")
        
        col_class1, col_class2 = st.columns([1, 3])
        
        with col_class1:
            if st.button("🎯 Classify Data", type="primary"):
                with st.spinner("Processing classification..."):
                    predictions_df = process_classification(
                        st.session_state.csv_data,
                        text_column,
                        ground_truth_column,
                        current_keywords,
                        match_mode,
                        confidence_threshold
                    )
                    st.session_state.predictions = predictions_df
                
                st.success("✅ Classification completed!")
        
        with col_class2:
            if st.session_state.predictions is not None:
                metrics = calculate_metrics(st.session_state.predictions)
                
                # Display metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
                with metric_col2:
                    st.metric("Precision", f"{metrics['precision']:.2%}")
                with metric_col3:
                    st.metric("Recall", f"{metrics['recall']:.2%}")
                with metric_col4:
                    st.metric("F1 Score", f"{metrics['f1']:.2%}")
    
    # Results Analysis
    if st.session_state.predictions is not None:
        st.markdown("---")
        st.header("5️⃣ Results Analysis")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔍 Error Analysis", "📈 Visualizations", "💾 Export"])
        
        with tab1:
            predictions_df = st.session_state.predictions
            metrics = calculate_metrics(predictions_df)
            
            # Confusion matrix values
            col_cm1, col_cm2 = st.columns([1, 1])
            
            with col_cm1:
                st.subheader("Confusion Matrix")
                cm_data = pd.DataFrame([
                    ['True Negative', 'False Positive'],
                    ['False Negative', 'True Positive']
                ], columns=['Predicted Negative', 'Predicted Positive'], 
                   index=['Actual Negative', 'Actual Positive'])
                
                cm_values = pd.DataFrame([
                    [metrics['tn'], metrics['fp']],
                    [metrics['fn'], metrics['tp']]
                ], columns=['Predicted Negative', 'Predicted Positive'], 
                   index=['Actual Negative', 'Actual Positive'])
                
                st.dataframe(cm_values)
            
            with col_cm2:
                st.subheader("Classification Report")
                st.write(f"**Total Predictions:** {len(predictions_df)}")
                st.write(f"**Correct Predictions:** {metrics['tp'] + metrics['tn']}")
                st.write(f"**Incorrect Predictions:** {metrics['fp'] + metrics['fn']}")
                st.write(f"**Baseline Accuracy:** {max(predictions_df['ground_truth'].sum(), len(predictions_df) - predictions_df['ground_truth'].sum()) / len(predictions_df):.2%}")
        
        with tab2:
            # Error Analysis
            false_positives = predictions_df[predictions_df['is_fp']]
            false_negatives = predictions_df[predictions_df['is_fn']]
            
            col_err1, col_err2 = st.columns(2)
            
            with col_err1:
                st.subheader(f"❌ False Positives ({len(false_positives)})")
                if len(false_positives) > 0:
                    for idx, row in false_positives.head(5).iterrows():
                        with st.expander(f"FP Example {idx + 1}"):
                            st.write(f"**Text:** {row['text']}")
                            st.write(f"**Matched Keywords:** {', '.join(row['matched_keywords'])}")
                            st.write(f"**Match Count:** {row['match_count']}")
            
            with col_err2:
                st.subheader(f"🔍 False Negatives ({len(false_negatives)})")
                if len(false_negatives) > 0:
                    for idx, row in false_negatives.head(5).iterrows():
                        with st.expander(f"FN Example {idx + 1}"):
                            st.write(f"**Text:** {row['text']}")
                            st.write("**No keywords matched** - Consider adding relevant keywords")
        
        with tab3:
            # Visualizations
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                # Confusion matrix heatmap
                fig_cm = create_confusion_matrix_viz(predictions_df)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col_viz2:
                # Metrics comparison
                metrics_data = pd.DataFrame({
                    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
                    'Score': [metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1']]
                })
                
                fig_metrics = px.bar(
                    metrics_data, 
                    x='Metric', 
                    y='Score',
                    title="Classification Metrics",
                    color='Score',
                    color_continuous_scale='viridis'
                )
                fig_metrics.update_layout(showlegend=False)
                st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Keyword impact analysis
            st.subheader("🎯 Keyword Impact Analysis")
            
            keyword_stats = []
            for keyword in current_keywords:
                # Find texts that match this keyword
                keyword_matches = predictions_df[
                    predictions_df['matched_keywords'].apply(
                        lambda x: keyword in x if isinstance(x, list) else False
                    )
                ]
                
                if len(keyword_matches) > 0:
                    tp_count = len(keyword_matches[keyword_matches['is_tp']])
                    fp_count = len(keyword_matches[keyword_matches['is_fp']])
                    
                    precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
                    
                    keyword_stats.append({
                        'Keyword': keyword,
                        'Total Matches': len(keyword_matches),
                        'True Positives': tp_count,
                        'False Positives': fp_count,
                        'Precision': precision
                    })
            
            if keyword_stats:
                keyword_df = pd.DataFrame(keyword_stats)
                keyword_df = keyword_df.sort_values('Precision', ascending=False)
                st.dataframe(keyword_df, use_container_width=True)
        
        with tab4:
            # Export options
            st.subheader("📥 Download Results")
            
            col_exp1, col_exp2 = st.columns(2)
            
            with col_exp1:
                # Download predictions
                predictions_csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📊 Download Predictions CSV",
                    data=predictions_csv,
                    file_name=f"predictions_{st.session_state.current_tactic}.csv",
                    mime="text/csv"
                )
            
            with col_exp2:
                # Download dictionary
                dictionary_json = json.dumps({
                    'tactic': st.session_state.current_tactic,
                    'keywords': current_keywords,
                    'metrics': {
                        'accuracy': float(metrics['accuracy']),
                        'precision': float(metrics['precision']),
                        'recall': float(metrics['recall']),
                        'f1': float(metrics['f1'])
                    },
                    'created': pd.Timestamp.now().isoformat()
                }, indent=2)
                
                st.download_button(
                    label="📚 Download Dictionary JSON",
                    data=dictionary_json,
                    file_name=f"dictionary_{st.session_state.current_tactic}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()
