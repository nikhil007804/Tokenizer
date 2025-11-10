# app.py — Detailed Explanation Version

# Streamlit Token Counter: tiktoken + Hugging Face tokenizers
# Run with:  streamlit run app.py

import sys                      # Import system-specific parameters and functions (used for debugging or extension)
from typing import List, Tuple, Dict, Any  # Type hints for better code clarity

import streamlit as st           # Streamlit library for creating the web app UI

# --- Optional Dependencies ---
try:
    import tiktoken              # Try importing tiktoken for OpenAI-style tokenization
    _HAS_TIKTOKEN = True         # Flag to indicate availability
except Exception:
    _HAS_TIKTOKEN = False        # If not installed, set flag to False

try:
    from transformers import AutoTokenizer  # Import Hugging Face tokenizer
    _HAS_TRANSFORMERS = True                # Flag to indicate availability
except Exception:
    _HAS_TRANSFORMERS = False               # Set False if unavailable

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Token Counter", page_icon="🧮", layout="wide")  # Configure page title, emoji, layout
st.title("🧮 Token Counter App")             # Main title of the app
st.caption("Count tokens and inspect individual tokens using tiktoken or a Hugging Face tokenizer.")  # Subtitle

# Sidebar for tokenizer settings
with st.sidebar:
    st.header("Tokenizer Settings")        # Sidebar header

    # Choose tokenizer type: tiktoken or HF
    family = st.selectbox(
        "Tokenizer family",
        ["tiktoken", "huggingface"],
        help="Choose which library to use for tokenization.",
    )

    # If tiktoken selected
    if family == "tiktoken":
        if not _HAS_TIKTOKEN:
            st.error("tiktoken is not installed. Run: pip install tiktoken")  # Error if not installed
        # Dropdown to select encoding
        tiktoken_model = st.selectbox(
            "tiktoken encoding",
            ["cl100k_base", "o200k_base", "p50k_base", "r50k_base"],
            index=0,
        )
    else:
        # Hugging Face tokenizer selection
        if not _HAS_TRANSFORMERS:
            st.error("transformers is not installed. Run: pip install transformers")
        hf_model = st.selectbox(
            "Hugging Face tokenizer",
            ["gpt2", "bert-base-uncased", "roberta-base", "distilbert-base-uncased"],
            index=0,
        )

# Text area for user input
text = st.text_area(
    "Input text",
    placeholder="Paste or type your paragraph here…",
    height=200,
)

# Column layout for options
col_a, col_b, col_c = st.columns([1,1,2])
with col_a:
    add_special = st.checkbox(
        "Add special tokens (HF only)", value=False,
        help="Whether to include special tokens like [CLS]/[SEP] for HF tokenizers."
    )
with col_b:
    show_ids = st.checkbox("Show token IDs", value=True)  # Whether to show token IDs
with col_c:
    show_bytes = st.checkbox("Show byte values (tiktoken)", value=False)  # Whether to show byte-level info

st.divider()  # Add horizontal divider line

# --- Tokenization Functions ---

def tokenize_with_tiktoken(s: str, encoding_name: str) -> Dict[str, Any]:
    if not _HAS_TIKTOKEN:
        raise RuntimeError("tiktoken not available")
    enc = tiktoken.get_encoding(encoding_name)  # Load encoding
    ids: List[int] = enc.encode(s)              # Convert text to token IDs
    tokens: List[str] = [                      # Decode each token to string
        enc.decode_single_token_bytes(tok).decode('utf-8', errors='replace') for tok in ids
    ]
    bytes_list: List[List[int]] = [list(enc.decode_single_token_bytes(tok)) for tok in ids]  # Byte view
    return {
        "tokens": tokens,
        "ids": ids,
        "bytes": bytes_list,
        "count": len(ids),
        "encoding_name": encoding_name,
    }


def tokenize_with_hf(s: str, model_name: str, add_special_tokens: bool=False) -> Dict[str, Any]:
    if not _HAS_TRANSFORMERS:
        raise RuntimeError("transformers not available")
    tok = AutoTokenizer.from_pretrained(model_name)   # Load tokenizer model
    token_pieces: List[str] = tok.tokenize(s)         # Get token strings
    enc = tok(s, add_special_tokens=add_special_tokens, return_attention_mask=False, return_tensors=None)
    ids: List[int] = enc["input_ids"]                # Extract token IDs
    if isinstance(ids[0], list):                     # Unpack if batched
        ids = ids[0]
    if add_special_tokens:
        token_pieces = tok.convert_ids_to_tokens(ids) # Include special tokens
    return {
        "tokens": token_pieces,
        "ids": ids,
        "bytes": None,
        "count": len(ids),
        "encoding_name": model_name,
    }

# --- Run the tokenization when button is clicked ---
run = st.button("Tokenize")

if run:
    if not text.strip():
        st.warning("Please enter some text to tokenize.")  # Warn if input empty
    else:
        try:
            # Run respective tokenizer
            if family == "tiktoken":
                result = tokenize_with_tiktoken(text, tiktoken_model)
            else:
                result = tokenize_with_hf(text, hf_model, add_special_tokens=add_special)
        except Exception as e:
            st.error(f"Tokenization failed: {e}")
            st.stop()

        # --- Display Results ---
        st.subheader("Results")
        st.metric(label="Token count", value=result["count"])  # Show total tokens
        st.caption(f"Tokenizer: {family} · Encoding/Model: {result['encoding_name']}")

        # Create DataFrame to show token info
        import pandas as pd
        rows = []
        for i, piece in enumerate(result["tokens"]):
            row = {"#": i, "token": piece}
            if show_ids:
                row["id"] = result["ids"][i]
            if show_bytes and result["bytes"] is not None:
                row["bytes"] = result["bytes"][i]
            rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)  # Display tokens in table

        # Download options
        st.download_button(
            "Download tokens (.txt)",
            data="\n".join(result["tokens"]),
            file_name="tokens.txt",
            mime="text/plain",
        )
        st.download_button(
            "Download tokens+ids (.csv)",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="tokens.csv",
            mime="text/csv",
        )

        # Optional raw output
        with st.expander("Preview raw Python objects"):
            st.write(result)

st.divider()

# --- Help Section ---
st.markdown(
    """
    **Quickstart**

    ```bash
    python -m venv .venv && source .venv/bin/activate
    pip install streamlit tiktoken transformers
    streamlit run app.py
    ```

    **Notes:**
    - tiktoken: use encodings like `cl100k_base`, `o200k_base`, etc.
    - Hugging Face: supports any tokenizer from the Hub.
    - `Add special tokens` adds model-specific symbols like `[CLS]`.
    - Toggle display of IDs/byte values as needed.
    """
)
