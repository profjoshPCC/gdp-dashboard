import streamlit as st
import pandas as pd
import re
import io

# Page configuration
st.set_page_config(
    page_title="Text Preprocessing for Classification",
    page_icon="📝",
    layout="wide"
)

# Title and description
st.title("📝 Text Preprocessing for Classification")
st.markdown("**Dr. Lin's Method** - Convert text data into Statement/Context format for classification")

# Helper functions
def advanced_sentence_split(text):
    """Advanced sentence splitter that handles hashtags and special cases."""
    hashtag_pattern = r'(#\w+(?:\s+#\w+)*)'
    parts = re.split(hashtag_pattern, text)
    sentences = []
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if part.startswith('#'):
            sentences.append(part)
        else:
            # Handle abbreviations
            part = re.sub(r'\b(Mr|Mrs|Dr|Ms|Prof|Sr|Jr)\.\s*', r'\1<DOT> ', part)
            part = re.sub(r'\b(Inc|Ltd|Corp|Co)\.\s*', r'\1<DOT> ', part)
            part = re.sub(r'\b(i\.e|e\.g|etc|vs)\.\s*', r'\1<DOT> ', part)
            
            sub_sentences = re.split(r'[.!?]+\s*', part)
            
            for s in sub_sentences:
                s = s.replace('<DOT>', '.').strip()
                if s and not re.match(r'^[^\w]+$', s):
                    sentences.append(s)
    
    return sentences

def process_instagram_posts(df):
    """Process Instagram posts - converts shortcode to ID and caption to Context."""
    df = df.rename(columns={'shortcode': 'ID', 'caption': 'Context'})
    results = []
    
    for idx, row in df.iterrows():
        post_id = row['ID']
        context = str(row['Context']) if pd.notna(row['Context']) else ''
        sentences = advanced_sentence_split(context)
        
        for sentence_id, sentence in enumerate(sentences, 1):
            results.append({
                'ID': post_id,
                'Statement': sentence,
                'Sentence ID': sentence_id,
                'Context': context
            })
    
    return pd.DataFrame(results)

def process_conversation_data(df, window_mode='rolling', statement_level='sentence'):
    """Process conversation data with turns and speakers."""
    results = []
    
    for conv_id in df['ID'].unique():
        conv_df = df[df['ID'] == conv_id].sort_values('Turn')
        
        # Build full conversation context
        full_context = []
        for _, row in conv_df.iterrows():
            full_context.append(f"Turn {row['Turn']} ({row['Speaker']}): {row['Text']}")
        full_context_str = ' '.join(full_context)
        
        for idx, row in conv_df.iterrows():
            turn_num = row['Turn']
            speaker = row['Speaker']
            text = row['Text']
            
            if statement_level == 'sentence':
                sentences = advanced_sentence_split(text)
                
                for sentence_id, sentence in enumerate(sentences, 1):
                    if window_mode == 'rolling':
                        context_turns = conv_df[conv_df['Turn'] <= turn_num]
                        context = []
                        for _, ctx_row in context_turns.iterrows():
                            context.append(f"Turn {ctx_row['Turn']} ({ctx_row['Speaker']}): {ctx_row['Text']}")
                        context_str = ' '.join(context)
                    else:
                        context_str = full_context_str
                    
                    results.append({
                        'ID': conv_id,
                        'Turn': turn_num,
                        'Sentence': sentence_id,
                        'Speaker': speaker,
                        'Statement': sentence,
                        'Context': context_str
                    })
            else:  # turn level
                if window_mode == 'rolling':
                    context_turns = conv_df[conv_df['Turn'] <= turn_num]
                    context = []
                    for _, ctx_row in context_turns.iterrows():
                        context.append(f"Turn {ctx_row['Turn']} ({ctx_row['Speaker']}): {ctx_row['Text']}")
                    context_str = ' '.join(context)
                else:
                    context_str = full_context_str
                
                results.append({
                    'ID': conv_id,
                    'Turn': turn_num,
                    'Sentence': 1,
                    'Speaker': speaker,
                    'Statement': text,
                    'Context': context_str
                })
    
    return pd.DataFrame(results)

# Sidebar for processing options
with st.sidebar:
    st.header("⚙️ Processing Options")
    
    processing_type = st.radio(
        "Select data type:",
        ["Instagram/Social Media Posts", "Conversation/Live Chat"]
    )
    
    if processing_type == "Conversation/Live Chat":
        st.subheader("Conversation Options")
        
        window_mode = st.radio(
            "Context mode:",
            ["rolling", "whole"],
            help="Rolling: context up to current turn\nWhole: entire conversation"
        )
        
        statement_level = st.radio(
            "Statement level:",
            ["sentence", "turn"],
            help="Sentence: split each turn into sentences\nTurn: keep full turn as statement"
        )

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Data")
    
    if processing_type == "Instagram/Social Media Posts":
        st.info("Upload CSV with 'shortcode' and 'caption' columns")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        # Sample data button
        if st.button("Load Sample Instagram Data"):
            sample_data = {
                'shortcode': ['ABC123', 'DEF456', 'GHI789'],
                'caption': [
                    'Check out our amazing limited time offer! Get 30% off while supplies last. #sale #limitedtime #shopnow',
                    'Excellence. A cornerstone in building a successful career. Join us today! #success #career',
                    'New arrivals just dropped! Shop the latest trends now. Visit our store today. #fashion #newcollection'
                ]
            }
            st.session_state['sample_df'] = pd.DataFrame(sample_data)
    
    else:  # Conversation
        st.info("Upload CSV with 'ID', 'Turn', 'Speaker', and 'Text' columns")
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        # Sample data button
        if st.button("Load Sample Conversation Data"):
            sample_data = {
                'ID': ['CONV001', 'CONV001', 'CONV001', 'CONV001'],
                'Turn': [1, 2, 3, 4],
                'Speaker': ['customer', 'salesperson', 'customer', 'salesperson'],
                'Text': [
                    'Hi, I would like to purchase a vacuum. Do you have any coupons?',
                    'I will be glad to help you find the perfect vacuum. Which model are you interested in?',
                    'The Kenmore 21514, but it seems expensive.',
                    'I see that this model is already on clearance sale. You are saving $50 from the regular price!'
                ]
            }
            st.session_state['sample_df'] = pd.DataFrame(sample_data)

# Process data
if uploaded_file is not None or 'sample_df' in st.session_state:
    # Load data
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = st.session_state['sample_df']
    
    # Show input data
    with col1:
        st.subheader("Input Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        st.write(f"Total rows: {len(df)}")
    
    # Process button
    if st.button("🚀 Process Data", type="primary"):
        with st.spinner("Processing..."):
            if processing_type == "Instagram/Social Media Posts":
                processed_df = process_instagram_posts(df)
            else:
                processed_df = process_conversation_data(df, window_mode, statement_level)
        
        # Store in session state
        st.session_state['processed_df'] = processed_df
        st.success(f"✅ Processed {len(processed_df)} statements!")

# Show results
with col2:
    st.header("📊 Processed Data")
    
    if 'processed_df' in st.session_state:
        processed_df = st.session_state['processed_df']
        
        # Preview
        st.subheader("Preview")
        st.dataframe(processed_df.head(10), use_container_width=True)
        
        # Statistics
        st.subheader("Statistics")
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("Total Statements", len(processed_df))
        with col_stat2:
            if 'ID' in processed_df.columns:
                st.metric("Unique IDs", processed_df['ID'].nunique())
        
        # Download button
        csv = processed_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Processed CSV",
            data=csv,
            file_name="processed_data.csv",
            mime="text/csv"
        )
    else:
        st.info("👈 Upload a file and click 'Process Data' to see results")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • Text Preprocessing for Classification")
