import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="CSV Joiner Tool",
    page_icon="🔗",
    layout="wide"
)

# Initialize session state
if 'df1' not in st.session_state:
    st.session_state.df1 = None
if 'df2' not in st.session_state:
    st.session_state.df2 = None
if 'joined_df' not in st.session_state:
    st.session_state.joined_df = None

def load_sample_data():
    """Create sample data for demonstration"""
    # Sample dataset 1 - Customer info
    df1 = pd.DataFrame({
        'customer_id': [1, 2, 3, 4, 5],
        'name': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eve Brown'],
        'email': ['alice@email.com', 'bob@email.com', 'carol@email.com', 'david@email.com', 'eve@email.com'],
        'signup_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-04-05', '2023-05-12']
    })
    
    # Sample dataset 2 - Order info
    df2 = pd.DataFrame({
        'customer_id': [1, 1, 2, 3, 3, 3, 4, 6],  # Note: customer 6 doesn't exist in df1
        'order_id': ['ORD001', 'ORD002', 'ORD003', 'ORD004', 'ORD005', 'ORD006', 'ORD007', 'ORD008'],
        'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Webcam', 'Tablet', 'Speaker'],
        'amount': [1200, 25, 75, 300, 150, 80, 500, 120],
        'order_date': ['2023-06-01', '2023-06-15', '2023-06-10', '2023-06-20', '2023-06-25', '2023-07-01', '2023-06-30', '2023-07-05']
    })
    
    return df1, df2

def perform_join(df1, df2, join_column1, join_column2, join_type):
    """Perform the specified join operation"""
    try:
        if join_type == "Inner Join":
            result = pd.merge(df1, df2, left_on=join_column1, right_on=join_column2, how='inner')
        elif join_type == "Left Join":
            result = pd.merge(df1, df2, left_on=join_column1, right_on=join_column2, how='left')
        elif join_type == "Right Join":
            result = pd.merge(df1, df2, left_on=join_column1, right_on=join_column2, how='right')
        elif join_type == "Outer Join":
            result = pd.merge(df1, df2, left_on=join_column1, right_on=join_column2, how='outer')
        
        return result, None
    except Exception as e:
        return None, str(e)

def get_join_preview(df1, df2, join_column1, join_column2, join_type):
    """Get a preview of how many records will result from the join"""
    try:
        # Get unique values in each join column
        unique_left = set(df1[join_column1].dropna())
        unique_right = set(df2[join_column2].dropna())
        
        # Calculate overlaps
        common_values = unique_left.intersection(unique_right)
        left_only = unique_left - unique_right
        right_only = unique_right - unique_left
        
        # Estimate result size based on join type
        if join_type == "Inner Join":
            # Only matching records
            matching_left = df1[df1[join_column1].isin(common_values)]
            matching_right = df2[df2[join_column2].isin(common_values)]
            estimated_rows = len(matching_left) * len(matching_right) // len(common_values) if len(common_values) > 0 else 0
        elif join_type == "Left Join":
            estimated_rows = len(df1)
        elif join_type == "Right Join":
            estimated_rows = len(df2)
        elif join_type == "Outer Join":
            estimated_rows = max(len(df1), len(df2))
        
        return {
            'common_values': len(common_values),
            'left_only': len(left_only),
            'right_only': len(right_only),
            'estimated_rows': estimated_rows
        }, None
    except Exception as e:
        return None, str(e)

# Main app
st.title("🔗 CSV Joiner Tool")
st.markdown("**Join two CSV files by matching columns**")
st.markdown("---")

# Quick start section
st.header("🚀 Quick Start")
if st.button("📊 Load Sample Data", type="primary"):
    df1, df2 = load_sample_data()
    st.session_state.df1 = df1
    st.session_state.df2 = df2
    st.success("✅ Sample data loaded! Customer data (5 records) and Order data (8 records)")

# File upload section
st.header("📁 Upload CSV Files")

col_upload1, col_upload2 = st.columns(2)

with col_upload1:
    st.subheader("📄 First CSV File")
    uploaded_file1 = st.file_uploader(
        "Choose first CSV file",
        type=['csv'],
        key="file1",
        help="This will be the 'left' table in the join"
    )
    
    if uploaded_file1 is not None:
        try:
            df1 = pd.read_csv(uploaded_file1)
            st.session_state.df1 = df1
            st.success(f"✅ Loaded: {len(df1)} rows, {len(df1.columns)} columns")
            
            # Show preview
            with st.expander("Preview First CSV"):
                st.dataframe(df1.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading first CSV: {str(e)}")

with col_upload2:
    st.subheader("📄 Second CSV File")
    uploaded_file2 = st.file_uploader(
        "Choose second CSV file",
        type=['csv'],
        key="file2",
        help="This will be the 'right' table in the join"
    )
    
    if uploaded_file2 is not None:
        try:
            df2 = pd.read_csv(uploaded_file2)
            st.session_state.df2 = df2
            st.success(f"✅ Loaded: {len(df2)} rows, {len(df2.columns)} columns")
            
            # Show preview
            with st.expander("Preview Second CSV"):
                st.dataframe(df2.head(), use_container_width=True)
                
        except Exception as e:
            st.error(f"Error reading second CSV: {str(e)}")

# Join configuration section
if st.session_state.df1 is not None and st.session_state.df2 is not None:
    st.markdown("---")
    st.header("⚙️ Configure Join")
    
    df1 = st.session_state.df1
    df2 = st.session_state.df2
    
    # Show data summaries
    col_summary1, col_summary2 = st.columns(2)
    
    with col_summary1:
        st.subheader("📊 First CSV Summary")
        st.write(f"**Rows:** {len(df1)}")
        st.write(f"**Columns:** {', '.join(df1.columns.tolist())}")
    
    with col_summary2:
        st.subheader("📊 Second CSV Summary")
        st.write(f"**Rows:** {len(df2)}")
        st.write(f"**Columns:** {', '.join(df2.columns.tolist())}")
    
    # Join configuration
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        join_column1 = st.selectbox(
            "Join column from first CSV:",
            df1.columns.tolist(),
            help="Select the column to match on from the first CSV"
        )
    
    with col_config2:
        join_column2 = st.selectbox(
            "Join column from second CSV:",
            df2.columns.tolist(),
            help="Select the column to match on from the second CSV"
        )
    
    with col_config3:
        join_type = st.selectbox(
            "Join type:",
            ["Inner Join", "Left Join", "Right Join", "Outer Join"],
            help="Choose how to combine the datasets"
        )
    
    # Join type explanation
    st.subheader("🔍 Join Type Explanation")
    join_explanations = {
        "Inner Join": "🎯 Only rows with matches in both CSVs",
        "Left Join": "⬅️ All rows from first CSV + matching rows from second CSV",
        "Right Join": "➡️ All rows from second CSV + matching rows from first CSV", 
        "Outer Join": "🔄 All rows from both CSVs (fills missing with NaN)"
    }
    
    st.info(join_explanations[join_type])
    
    # Join preview
    if join_column1 and join_column2:
        preview_info, preview_error = get_join_preview(df1, df2, join_column1, join_column2, join_type)
        
        if preview_info:
            st.subheader("📋 Join Preview")
            
            col_prev1, col_prev2, col_prev3, col_prev4 = st.columns(4)
            
            with col_prev1:
                st.metric("Common Values", preview_info['common_values'])
            with col_prev2:
                st.metric("Only in First CSV", preview_info['left_only'])
            with col_prev3:
                st.metric("Only in Second CSV", preview_info['right_only'])
            with col_prev4:
                st.metric("Estimated Result Rows", preview_info['estimated_rows'])
        
        elif preview_error:
            st.warning(f"Preview error: {preview_error}")
    
    # Perform join
    st.markdown("---")
    if st.button("🔗 Perform Join", type="primary"):
        with st.spinner("Joining datasets..."):
            joined_df, join_error = perform_join(df1, df2, join_column1, join_column2, join_type)
            
            if joined_df is not None:
                st.session_state.joined_df = joined_df
                st.success(f"✅ Join completed! Result has {len(joined_df)} rows and {len(joined_df.columns)} columns")
            else:
                st.error(f"Join failed: {join_error}")

# Results section
if st.session_state.joined_df is not None:
    st.markdown("---")
    st.header("📊 Join Results")
    
    joined_df = st.session_state.joined_df
    
    # Results summary
    col_result1, col_result2, col_result3 = st.columns(3)
    
    with col_result1:
        st.metric("Total Rows", len(joined_df))
    with col_result2:
        st.metric("Total Columns", len(joined_df.columns))
    with col_result3:
        # Count null values (indicating no match in outer joins)
        null_count = joined_df.isnull().sum().sum()
        st.metric("Null Values", null_count)
    
    # Results tabs
    tab1, tab2, tab3 = st.tabs(["📋 Preview", "📈 Analysis", "💾 Export"])
    
    with tab1:
        st.subheader("Joined Data Preview")
        
        # Show options
        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            show_rows = st.selectbox("Rows to display:", [10, 25, 50, 100, "All"])
        with col_opt2:
            if st.checkbox("Show data types"):
                st.write("**Column Data Types:**")
                for col, dtype in joined_df.dtypes.items():
                    st.write(f"- {col}: {dtype}")
        
        # Display data
        if show_rows == "All":
            st.dataframe(joined_df, use_container_width=True)
        else:
            st.dataframe(joined_df.head(show_rows), use_container_width=True)
        
        if len(joined_df) > 10:
            st.info(f"Showing preview. Full dataset has {len(joined_df)} rows.")
    
    with tab2:
        st.subheader("Join Analysis")
        
        # Column analysis
        st.write("**Columns from each CSV:**")
        
        if st.session_state.df1 is not None and st.session_state.df2 is not None:
            original_cols1 = set(st.session_state.df1.columns)
            original_cols2 = set(st.session_state.df2.columns)
            result_cols = set(joined_df.columns)
            
            col_analysis1, col_analysis2 = st.columns(2)
            
            with col_analysis1:
                st.write("**From First CSV:**")
                for col in original_cols1:
                    if col in result_cols:
                        st.write(f"✅ {col}")
                    else:
                        st.write(f"❌ {col} (not in result)")
            
            with col_analysis2:
                st.write("**From Second CSV:**")
                for col in original_cols2:
                    if col in result_cols:
                        suffix = "_y" if f"{col}_y" in result_cols else ""
                        display_name = f"{col}{suffix}" if suffix else col
                        st.write(f"✅ {display_name}")
                    else:
                        st.write(f"❌ {col} (not in result)")
        
        # Data quality check
        st.subheader("Data Quality Check")
        
        quality_issues = []
        
        # Check for duplicate rows
        duplicates = joined_df.duplicated().sum()
        if duplicates > 0:
            quality_issues.append(f"⚠️ Found {duplicates} duplicate rows")
        
        # Check for completely empty rows
        empty_rows = joined_df.isnull().all(axis=1).sum()
        if empty_rows > 0:
            quality_issues.append(f"⚠️ Found {empty_rows} completely empty rows")
        
        # Check for high null percentage columns
        for col in joined_df.columns:
            null_pct = (joined_df[col].isnull().sum() / len(joined_df)) * 100
            if null_pct > 50:
                quality_issues.append(f"⚠️ Column '{col}' is {null_pct:.1f}% null")
        
        if quality_issues:
            st.warning("Data Quality Issues Found:")
            for issue in quality_issues:
                st.write(issue)
        else:
            st.success("✅ No major data quality issues detected!")
    
    with tab3:
        st.subheader("Export Joined Data")
        
        # File format options
        col_export1, col_export2 = st.columns(2)
        
        with col_export1:
            export_format = st.selectbox("Export format:", ["CSV", "Excel"])
            
            if export_format == "CSV":
                # CSV options
                include_index = st.checkbox("Include row numbers", value=False)
                
                csv_data = joined_df.to_csv(index=include_index)
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="joined_data.csv",
                    mime="text/csv"
                )
            
            else:  # Excel
                # Excel options
                include_index = st.checkbox("Include row numbers", value=False)
                
                # Note: For Excel export, we'd need openpyxl
                st.info("💡 Excel export requires openpyxl package. Falling back to CSV.")
                csv_data = joined_df.to_csv(index=include_index)
                
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv_data,
                    file_name="joined_data.csv",
                    mime="text/csv"
                )
        
        with col_export2:
            st.write("**Export Summary:**")
            st.write(f"• **Rows to export:** {len(joined_df):,}")
            st.write(f"• **Columns to export:** {len(joined_df.columns)}")
            st.write(f"• **Estimated file size:** ~{len(str(joined_df)) / 1024:.1f} KB")

else:
    # Help section when no data is loaded
    st.header("📖 How to Use")
    st.markdown("""
    1. **Quick Start**: Click "Load Sample Data" to try the tool immediately
    2. **Upload Files**: Or upload your own two CSV files
    3. **Configure Join**: 
       - Select the columns to match on from each CSV
       - Choose the type of join you want
    4. **Preview**: See how many records will match before joining
    5. **Join**: Perform the join operation
    6. **Export**: Download the combined dataset
    """)
    
    st.subheader("🔗 Join Types Explained")
    
    join_details = {
        "Inner Join": {
            "description": "Only keeps rows where the join column has matches in both CSVs",
            "example": "Customer data + Order data → Only customers who have orders",
            "icon": "🎯"
        },
        "Left Join": {
            "description": "Keeps all rows from the first CSV, adds matching data from second CSV",
            "example": "All customers + their orders (if any) → All customers, some with orders",
            "icon": "⬅️"
        },
        "Right Join": {
            "description": "Keeps all rows from the second CSV, adds matching data from first CSV", 
            "example": "Customer data + All orders → All orders, some with customer details",
            "icon": "➡️"
        },
        "Outer Join": {
            "description": "Keeps all rows from both CSVs, fills missing matches with empty values",
            "example": "All customers + All orders → Everything, with gaps where no match",
            "icon": "🔄"
        }
    }
    
    for join_name, details in join_details.items():
        with st.expander(f"{details['icon']} {join_name}"):
            st.write(f"**Description:** {details['description']}")
            st.write(f"**Example:** {details['example']}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit • CSV Joiner Tool")
