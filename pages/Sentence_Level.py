import streamlit as st
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Marketing Statement Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

def count_words(text):
    """Count words in text"""
    if not text or pd.isna(text) or not isinstance(text, str):
        return 0
    return len([word for word in text.strip().split() if word])

def split_into_sentences(text):
    """Split text into sentences"""
    if not text or pd.isna(text) or not isinstance(text, str):
        return []
    # Split on sentence endings, then clean up
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def process_marketing_data(df):
    """Process the marketing data with word count calculations"""
    processed_rows = []
    
    for idx, row in df.iterrows():
        # Get the statement and matched keywords
        statement = str(row.get('Statement', '')) if pd.notna(row.get('Statement', '')) else ''
        matched_keywords = str(row.get('MatchedKeywords', '')) if pd.notna(row.get('MatchedKeywords', '')) else ''
        
        # Calculate statement_word_count
        statement_word_count = count_words(statement)
        
        # Calculate tactic_word_count
        tactic_word_count = 0
        
        if statement and matched_keywords:
            # Split statement into sentences
            sentences = split_into_sentences(statement)
            
            # Parse keywords (handle comma and space separation)
            keywords = []
            if matched_keywords:
                # Split by comma or semicolon, then by spaces, and clean up
                keyword_parts = re.split(r'[,;]', matched_keywords.lower())
                for part in keyword_parts:
                    keywords.extend([k.strip() for k in part.split() if k.strip()])
            
            # Count words from sentences that contain keywords
            for sentence in sentences:
                sentence_lower = sentence.lower()
                has_keyword = any(keyword in sentence_lower for keyword in keywords if keyword)
                
                if has_keyword:
                    tactic_word_count += count_words(sentence)
        
        # Calculate tactic_word_count_percent
        tactic_word_count_percent = (
            round((tactic_word_count / statement_word_count) * 100, 2) 
            if statement_word_count > 0 else 0.0
        )
        
        # Create new row with all original data plus new calculations
        new_row = dict(row)
        new_row['statement_word_count'] = statement_word_count
        new_row['tactic_word_count'] = tactic_word_count
        new_row['tactic_word_count_percent'] = tactic_word_count_percent
        
        processed_rows.append(new_row)
    
    return pd.DataFrame(processed_rows)

def create_sample_data():
    """Create sample data for demonstration"""
    return pd.DataFrame({
        'ID': [1, 2, 3, 4, 5],
        'Statement': [
            'Limited time offer on premium products! Get exclusive deals while supplies last.',
            'Our heritage brand represents timeless quality and craftsmanship in every piece.',
            'Regular everyday products for your basic needs and requirements.',
            'Luxury items with sophisticated design. Premium materials and elegant finishing.',
            'Standard features included. Basic functionality at an affordable price point.'
        ],
        'GroundTruth': [1, 1, 0, 1, 0],
        'Prediction': [1, 1, 0, 1, 0],
        'MatchedKeywords': [
            'limited, exclusive, premium',
            'heritage, quality, craftsmanship',
            '',
            'luxury, premium, sophisticated, elegant',
            ''
        ],
        'IsCorrect': ['Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
        'Type': ['TP', 'TP', 'TN', 'TP', 'TN']
    })

# Main app
st.title("📊 Marketing Statement Analyzer")
st.markdown("**Analyze word counts and keyword distribution in marketing statements**")
st.markdown("---")

# Quick start section
st.header("🚀 Quick Start")
if st.button("📋 Load Sample Data", type="primary"):
    st.session_state.uploaded_data = create_sample_data()
    st.success("✅ Sample marketing statements loaded!")

# File upload section
st.header("📁 Upload Your Data")

uploaded_file = st.file_uploader(
    "Choose CSV file",
    type=['csv'],
    help="Expected columns: ID, Statement, GroundTruth, Prediction, MatchedKeywords, IsCorrect, Type"
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_data = df
        st.success(f"✅ File uploaded successfully! {len(df)} rows loaded.")
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")

# Main processing section
if st.session_state.uploaded_data is not None:
    df = st.session_state.uploaded_data
    
    # Show data preview
    st.subheader("📋 Data Preview")
    st.dataframe(df.head(), use_container_width=True)
    
    # Data summary
    col_summary1, col_summary2, col_summary3 = st.columns(3)
    with col_summary1:
        st.metric("Total Rows", len(df))
    with col_summary2:
        st.metric("Columns", len(df.columns))
    with col_summary3:
        if 'Statement' in df.columns:
            avg_words = df['Statement'].apply(count_words).mean()
            st.metric("Avg Words/Statement", f"{avg_words:.1f}")
    
    # Column validation
    required_columns = ['Statement', 'MatchedKeywords']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.warning(f"⚠️ Missing required columns: {', '.join(missing_columns)}")
        st.info("The tool needs 'Statement' and 'MatchedKeywords' columns to work properly.")
    
    st.markdown("---")
    
    # Processing section
    st.header("⚙️ Process Data")
    
    if st.button("🔄 Calculate Word Counts", type="primary"):
        if not missing_columns:
            with st.spinner("Processing marketing statements..."):
                processed_df = process_marketing_data(df)
                st.session_state.processed_data = processed_df
            
            st.success("✅ Processing complete!")
        else:
            st.error("Cannot process: missing required columns")
    
    # Results section
    if st.session_state.processed_data is not None:
        processed_df = st.session_state.processed_data
        
        st.markdown("---")
        st.header("📊 Analysis Results")
        
        # Summary metrics
        col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
        
        with col_metric1:
            avg_statement_words = processed_df['statement_word_count'].mean()
            st.metric("Avg Statement Words", f"{avg_statement_words:.1f}")
        
        with col_metric2:
            avg_tactic_words = processed_df['tactic_word_count'].mean()
            st.metric("Avg Tactic Words", f"{avg_tactic_words:.1f}")
        
        with col_metric3:
            avg_tactic_percent = processed_df['tactic_word_count_percent'].mean()
            st.metric("Avg Tactic %", f"{avg_tactic_percent:.1f}%")
        
        with col_metric4:
            zero_tactic_count = (processed_df['tactic_word_count'] == 0).sum()
            st.metric("No Keywords", zero_tactic_count)
        
        # Results tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Processed Data", "📈 Analysis", "📊 Visualizations", "💾 Export"])
        
        with tab1:
            st.subheader("Processed Results")
            
            # Display options
            col_display1, col_display2 = st.columns(2)
            with col_display1:
                show_rows = st.selectbox("Rows to display:", [10, 25, 50, "All"])
            with col_display2:
                show_new_cols_only = st.checkbox("Show only new columns", value=False)
            
            # Select columns to display
            if show_new_cols_only:
                display_columns = ['ID', 'Statement', 'MatchedKeywords', 
                                 'statement_word_count', 'tactic_word_count', 'tactic_word_count_percent']
                display_columns = [col for col in display_columns if col in processed_df.columns]
            else:
                display_columns = processed_df.columns.tolist()
            
            # Display data
            if show_rows == "All":
                st.dataframe(processed_df[display_columns], use_container_width=True)
            else:
                st.dataframe(processed_df[display_columns].head(show_rows), use_container_width=True)
            
            if len(processed_df) > 10:
                st.info(f"Showing preview. Full dataset has {len(processed_df)} rows.")
        
        with tab2:
            st.subheader("📈 Detailed Analysis")
            
            # Tactic effectiveness analysis
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                st.write("**Tactic Word Count Distribution:**")
                tactic_dist = processed_df['tactic_word_count'].value_counts().sort_index()
                for count, freq in tactic_dist.head(10).items():
                    st.write(f"• {count} tactic words: {freq} statements")
            
            with col_analysis2:
                st.write("**Tactic Percentage Ranges:**")
                # Create percentage ranges
                processed_df['tactic_range'] = pd.cut(
                    processed_df['tactic_word_count_percent'], 
                    bins=[0, 25, 50, 75, 100], 
                    labels=['0-25%', '26-50%', '51-75%', '76-100%'],
                    include_lowest=True
                )
                
                range_counts = processed_df['tactic_range'].value_counts()
                for range_label, count in range_counts.items():
                    percentage = (count / len(processed_df)) * 100
                    st.write(f"• {range_label}: {count} statements ({percentage:.1f}%)")
            
            # Top and bottom performers
            st.subheader("🎯 Performance Insights")
            
            col_insights1, col_insights2 = st.columns(2)
            
            with col_insights1:
                st.write("**Highest Tactic Percentage:**")
                top_tactic = processed_df.nlargest(3, 'tactic_word_count_percent')
                for idx, row in top_tactic.iterrows():
                    st.write(f"• {row['tactic_word_count_percent']:.1f}%: {row['Statement'][:50]}...")
            
            with col_insights2:
                st.write("**Most Tactic Words:**")
                top_words = processed_df.nlargest(3, 'tactic_word_count')
                for idx, row in top_words.iterrows():
                    st.write(f"• {row['tactic_word_count']} words: {row['Statement'][:50]}...")
            
            # Keyword analysis
            if 'MatchedKeywords' in processed_df.columns:
                st.subheader("🔍 Keyword Analysis")
                
                # Extract all keywords
                all_keywords = []
                for keywords_str in processed_df['MatchedKeywords'].dropna():
                    if keywords_str:
                        keywords = re.split(r'[,;]', str(keywords_str).lower())
                        all_keywords.extend([k.strip() for k in keywords if k.strip()])
                
                if all_keywords:
                    from collections import Counter
                    keyword_counts = Counter(all_keywords)
                    
                    st.write("**Most Common Keywords:**")
                    for keyword, count in keyword_counts.most_common(10):
                        st.write(f"• {keyword}: {count} occurrences")
        
        with tab3:
            st.subheader("📊 Data Visualizations")
            
            # Visualization 1: Tactic word count distribution
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                fig_hist = px.histogram(
                    processed_df, 
                    x='tactic_word_count',
                    title='Distribution of Tactic Word Counts',
                    nbins=20,
                    color_discrete_sequence=['#636EFA']
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col_viz2:
                fig_scatter = px.scatter(
                    processed_df,
                    x='statement_word_count',
                    y='tactic_word_count',
                    title='Statement Words vs Tactic Words',
                    hover_data=['tactic_word_count_percent'],
                    color='tactic_word_count_percent',
                    color_continuous_scale='viridis'
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Visualization 2: Tactic percentage distribution
            fig_box = px.box(
                processed_df,
                y='tactic_word_count_percent',
                title='Distribution of Tactic Word Percentages'
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Visualization 3: Performance by type (if available)
            if 'Type' in processed_df.columns:
                type_analysis = processed_df.groupby('Type').agg({
                    'tactic_word_count_percent': 'mean',
                    'tactic_word_count': 'mean',
                    'statement_word_count': 'mean'
                }).round(2)
                
                fig_bar = px.bar(
                    type_analysis.reset_index(),
                    x='Type',
                    y='tactic_word_count_percent',
                    title='Average Tactic Percentage by Classification Type',
                    color='Type'
                )
                fig_bar.update_layout(height=400)
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab4:
            st.subheader("💾 Export Processed Data")
            
            # Export options
            col_export1, col_export2 = st.columns(2)
            
            with col_export1:
                export_format = st.selectbox("Export format:", ["CSV", "Excel (CSV)"])
                include_original = st.checkbox("Include original columns", value=True)
                
                if not include_original:
                    export_df = processed_df[['ID', 'Statement', 'MatchedKeywords', 
                                            'statement_word_count', 'tactic_word_count', 
                                            'tactic_word_count_percent']]
                else:
                    export_df = processed_df
                
                # Generate CSV
                csv_data = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Processed Data",
                    data=csv_data,
                    file_name="processed_marketing_statements.csv",
                    mime="text/csv"
                )
            
            with col_export2:
                st.write("**Export Summary:**")
                st.write(f"• **Rows:** {len(export_df):,}")
                st.write(f"• **Columns:** {len(export_df.columns)}")
                st.write(f"• **New columns added:** 3")
                st.write(f"• **File size:** ~{len(csv_data) / 1024:.1f} KB")
                
                # Show new columns info
                st.write("**New Columns:**")
                st.write("• `statement_word_count`: Total words in statement")
                st.write("• `tactic_word_count`: Words from keyword sentences")
                st.write("• `tactic_word_count_percent`: Percentage of tactic words")

else:
    # Help section when no data is loaded
    st.header("📖 How to Use")
    st.markdown("""
    1. **Quick Start**: Click "Load Sample Data" to try the tool immediately
    2. **Upload Data**: Or upload your CSV file with marketing statements
    3. **Process**: Click "Calculate Word Counts" to analyze your data
    4. **Analyze**: Review the results and visualizations
    5. **Export**: Download the processed data with new calculations
    """)
    
    st.subheader("🔍 What This Tool Does")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        **Calculations Performed:**
        - **statement_word_count**: Counts total words in each statement
        - **tactic_word_count**: Counts words from sentences containing matched keywords
        - **tactic_word_count_percent**: Percentage of statement words from keyword sentences
        """)
    
    with col_info2:
        st.markdown("""
        **Example:**
        Statement: "Limited time offer! Get premium quality products."
        Keywords: "limited, premium"
        
        - statement_word_count: 8
        - tactic_word_count: 8 (both sentences have keywords)
        - tactic_word_count_percent: 100%
        """)
    
    st.subheader("📋 Expected CSV Format")
    
    example_df = pd.DataFrame({
        'ID': [1, 2],
        'Statement': [
            'Limited time premium offer available now!',
            'Regular everyday products for basic needs.'
        ],
        'MatchedKeywords': ['limited, premium', ''],
        'GroundTruth': [1, 0],
        'Prediction': [1, 0]
    })
    
    st.dataframe(example_df, use_container_width=True)
    st.caption("Required columns: Statement, MatchedKeywords. Other columns are optional.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Marketing Statement Analyzer")
