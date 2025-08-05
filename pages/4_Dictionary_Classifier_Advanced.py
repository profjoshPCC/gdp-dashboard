import streamlit as st
import pandas as pd
import re
import json
from io import StringIO

# Set page configuration
st.set_page_config(
    page_title="Text Classification Tool",
    page_icon="📊",
    layout="wide"
)

def classify_text(text, dictionaries):
    """Classify text based on dictionary matches"""
    if pd.isna(text):
        return {dict_name: 0 for dict_name in dictionaries.keys()}
    
    text_lower = str(text).lower()
    results = {}
    
    for dict_name, keywords in dictionaries.items():
        count = 0
        for keyword in keywords:
            # Use word boundaries for exact matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            count += len(re.findall(pattern, text_lower))
        results[dict_name] = count
    
    return results

def process_data(df, dictionaries, text_column):
    """Process the dataframe with classification"""
    # Apply classification to specified column
    classifications = df[text_column].apply(lambda x: classify_text(x, dictionaries))
    
    # Convert results to separate columns
    for dict_name in dictionaries.keys():
        df[f'{dict_name}_count'] = classifications.apply(lambda x: x[dict_name])
        df[f'{dict_name}_present'] = df[f'{dict_name}_count'] > 0
    
    return df

def main():
    st.title("📊 Text Classification Tool")
    st.markdown("Upload your dataset and classify text using customizable keyword dictionaries.")
    
    # Initialize session state for dictionaries
    if 'dictionaries' not in st.session_state:
        st.session_state.dictionaries = {
            'urgency_marketing': {
                'limited', 'limited time', 'limited run', 'limited edition', 'order now',
                'last chance', 'hurry', 'while supplies last', 'before they\'re gone',
                'selling out', 'selling fast', 'act now', 'don\'t wait', 'today only',
                'expires soon', 'final hours', 'almost gone'
            },
            'exclusive_marketing': {
                'exclusive', 'exclusively', 'exclusive offer', 'exclusive deal',
                'members only', 'vip', 'special access', 'invitation only',
                'premium', 'privileged', 'limited access', 'select customers',
                'insider', 'private sale', 'early access'
            }
        }
    
    # Sidebar for dictionary management
    st.sidebar.header("🔧 Dictionary Management")
    
    # Dictionary editor
    st.sidebar.subheader("Edit Dictionaries")
    
    # Select dictionary to edit
    dict_names = list(st.session_state.dictionaries.keys())
    selected_dict = st.sidebar.selectbox("Select dictionary to edit:", dict_names)
    
    if selected_dict:
        st.sidebar.write(f"**{selected_dict}** keywords:")
        
        # Display current keywords
        current_keywords = list(st.session_state.dictionaries[selected_dict])
        keywords_text = '\n'.join(current_keywords)
        
        # Text area for editing keywords
        updated_keywords = st.sidebar.text_area(
            "Keywords (one per line):",
            value=keywords_text,
            height=200,
            key=f"keywords_{selected_dict}"
        )
        
        if st.sidebar.button(f"Update {selected_dict}"):
            # Update the dictionary
            new_keywords = set(line.strip() for line in updated_keywords.split('\n') if line.strip())
            st.session_state.dictionaries[selected_dict] = new_keywords
            st.sidebar.success(f"Updated {selected_dict}!")
    
    # Add new dictionary
    st.sidebar.subheader("Add New Dictionary")
    new_dict_name = st.sidebar.text_input("Dictionary name:")
    new_dict_keywords = st.sidebar.text_area("Keywords (one per line):", height=100)
    
    if st.sidebar.button("Add Dictionary"):
        if new_dict_name and new_dict_keywords:
            keywords = set(line.strip() for line in new_dict_keywords.split('\n') if line.strip())
            st.session_state.dictionaries[new_dict_name] = keywords
            st.sidebar.success(f"Added {new_dict_name}!")
        else:
            st.sidebar.error("Please provide both name and keywords.")
    
    # Remove dictionary
    if len(dict_names) > 1:
        dict_to_remove = st.sidebar.selectbox("Remove dictionary:", [""] + dict_names)
        if st.sidebar.button("Remove Dictionary") and dict_to_remove:
            del st.session_state.dictionaries[dict_to_remove]
            st.sidebar.success(f"Removed {dict_to_remove}!")
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file containing text data to classify"
        )
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
                
                st.success(f"✅ File uploaded successfully! Shape: {df.shape}")
                
                # Column selection
                text_columns = df.select_dtypes(include=['object']).columns.tolist()
                if text_columns:
                    selected_column = st.selectbox(
                        "Select the text column to classify:",
                        text_columns,
                        help="Choose the column containing the text you want to classify"
                    )
                    
                    # Display sample data
                    st.subheader("📋 Sample Data")
                    st.dataframe(df.head(), use_container_width=True)
                    
                    # Process button
                    if st.button("🚀 Run Classification", type="primary"):
                        with st.spinner("Processing..."):
                            # Process the data
                            processed_df = process_data(df.copy(), st.session_state.dictionaries, selected_column)
                            
                            # Display results
                            st.subheader("📊 Classification Results")
                            
                            # Show classification columns
                            classification_cols = [col for col in processed_df.columns 
                                                 if any(dict_name in col for dict_name in st.session_state.dictionaries.keys())]
                            
                            display_cols = [selected_column] + classification_cols
                            if 'ID' in processed_df.columns:
                                display_cols = ['ID'] + display_cols
                            
                            st.dataframe(processed_df[display_cols], use_container_width=True)
                            
                            # Summary statistics
                            st.subheader("📈 Summary Statistics")
                            summary_data = []
                            
                            for dict_name in st.session_state.dictionaries.keys():
                                total_matches = processed_df[f'{dict_name}_count'].sum()
                                texts_with_matches = processed_df[f'{dict_name}_present'].sum()
                                percentage = (texts_with_matches / len(processed_df)) * 100
                                
                                summary_data.append({
                                    'Dictionary': dict_name,
                                    'Total Matches': total_matches,
                                    'Texts with Matches': texts_with_matches,
                                    'Percentage': f"{percentage:.1f}%"
                                })
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True)
                            
                            # Download processed data
                            csv = processed_df.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Classification Results",
                                data=csv,
                                file_name="classified_data.csv",
                                mime="text/csv"
                            )
                            
                            # Store processed data in session state for further analysis
                            st.session_state.processed_df = processed_df
                            
                else:
                    st.error("No text columns found in the uploaded file.")
                    
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
        
        else:
            # Show sample data format
            st.info("👆 Upload a CSV file to get started!")
            st.subheader("Expected Data Format")
            sample_df = pd.DataFrame({
                'ID': [1, 2, 3],
                'Statement': [
                    'Limited time offer - act now!',
                    'Exclusive deal for VIP members only',
                    'Regular product description'
                ]
            })
            st.dataframe(sample_df, use_container_width=True)
    
    with col2:
        st.header("📚 Current Dictionaries")
        
        # Display current dictionaries
        for dict_name, keywords in st.session_state.dictionaries.items():
            with st.expander(f"**{dict_name}** ({len(keywords)} keywords)"):
                for keyword in sorted(keywords):
                    st.write(f"• {keyword}")
        
        # Export/Import dictionaries
        st.subheader("💾 Dictionary Export/Import")
        
        # Export
        dict_json = json.dumps({k: list(v) for k, v in st.session_state.dictionaries.items()}, indent=2)
        st.download_button(
            label="📤 Export Dictionaries",
            data=dict_json,
            file_name="dictionaries.json",
            mime="application/json"
        )
        
        # Import
        uploaded_dict = st.file_uploader("📥 Import Dictionaries", type=['json'])
        if uploaded_dict is not None:
            try:
                dict_data = json.load(uploaded_dict)
                imported_dicts = {k: set(v) for k, v in dict_data.items()}
                
                if st.button("Import Dictionaries"):
                    st.session_state.dictionaries.update(imported_dicts)
                    st.success("Dictionaries imported successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error importing dictionaries: {str(e)}")

if __name__ == "__main__":
    main()
