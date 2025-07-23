import streamlit as st
import pandas as pd
import re
import json
from io import StringIO
from typing import Dict, List, Set

st.set_page_config(page_title="Dictionary‑Based Text Classifier", layout="wide")

st.title("🗂️ Dictionary‑Based Text Classifier")

st.markdown(
    """
Upload a CSV file containing a **Statement** (or any text) column,
customize the keyword dictionaries below, and classify your data on‑device.
"""
)

# ------------------------------------------------------------------
# 1️⃣  File upload & settings
# ------------------------------------------------------------------

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

column_name = st.text_input("Name of text column in CSV", value="Statement")

# ------------------------------------------------------------------
# 2️⃣  Dictionary editor
# ------------------------------------------------------------------

st.subheader("Keyword dictionaries (edit as JSON)")

default_dict: Dict[str, List[str]] = {
    "urgency_marketing": [
        "limited", "limited time", "limited run", "limited edition", "order now",
        "last chance", "hurry", "while supplies last", "before they're gone",
        "selling out", "selling fast", "act now", "don't wait", "today only",
        "expires soon", "final hours", "almost gone"
    ],
    "exclusive_marketing": [
        "exclusive", "exclusively", "exclusive offer", "exclusive deal",
        "members only", "vip", "special access", "invitation only",
        "premium", "privileged", "limited access", "select customers",
        "insider", "private sale", "early access"
    ]
}

# Show JSON in a textarea for user editing
initial_json = json.dumps(default_dict, indent=2)

dict_json = st.text_area(
    "Paste / edit JSON here", value=initial_json, height=300, key="dict_json"
)

# Attempt to parse user JSON input
try:
    user_dict_raw: Dict[str, List[str]] = json.loads(dict_json)
    dict_error = None
except json.JSONDecodeError as e:
    user_dict_raw = default_dict
    dict_error = str(e)

if dict_error:
    st.error(f"Invalid JSON – using last valid version.\n{dict_error}")

# Convert lists to sets for faster lookup during classification
user_dict: Dict[str, Set[str]] = {
    label: set(kw_list) for label, kw_list in user_dict_raw.items()
}

# ------------------------------------------------------------------
# 3️⃣  Helper functions (cached)
# ------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def compile_patterns(dictionaries: Dict[str, Set[str]]):
    """Pre‑compile regex patterns for each label."""
    return {
        label: re.compile(r"\b(?:" + "|".join(map(re.escape, kw_set)) + r")\b", re.IGNORECASE)
        for label, kw_set in dictionaries.items()
    }

@st.cache_data(show_spinner=False)
def classify_dataframe(csv_bytes, col: str, dictionaries: Dict[str, Set[str]]):
    """Read CSV from bytes, classify texts, return enriched DataFrame."""
    df = pd.read_csv(csv_bytes)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in CSV.")

    patterns = compile_patterns(dictionaries)

    def classify_text(text: str):
        if not isinstance(text, str):
            return []
        return [label for label, pat in patterns.items() if pat.search(text)]

    df["labels"] = df[col].apply(classify_text)
    return df

# ------------------------------------------------------------------
# 4️⃣  Run classification and display results
# ------------------------------------------------------------------

if uploaded_file is not None:
    if st.button("Run classification"):
        try:
            df_result = classify_dataframe(uploaded_file, column_name, user_dict)
            st.success("Classification complete!")

            # Show first few rows
            st.dataframe(df_result.head())

            # Label distribution summary
            label_counts = (
                df_result["labels"].explode().value_counts().rename_axis("label").to_frame("count")
            )
            st.subheader("Label distribution")
            st.bar_chart(label_counts)

            # Download link
            csv_buffer = StringIO()
            df_result.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download classified CSV",
                csv_buffer.getvalue(),
                file_name="classified_data.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error: {e}")
else:
    st.info("👆 Upload a CSV file to get started.")

