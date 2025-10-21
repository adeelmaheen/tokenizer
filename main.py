import streamlit as st
import requests
from typing import List, Dict
import json
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer

# Page configuration
st.set_page_config(
    page_title="AI Tokenizer Pro",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional UI
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .token-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        color: white;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 4px solid #667eea;
    }
    h1 {
        color: #667eea;
        font-weight: 700;
    }
    .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .token-display {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
        max-height: 300px;
        overflow-y: auto;
    }
    .token-item {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 5px 10px;
        margin: 3px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'tokenization_history' not in st.session_state:
    st.session_state.tokenization_history = []
if 'tokenizer_loaded' not in st.session_state:
    st.session_state.tokenizer_loaded = False
if 'current_tokenizer' not in st.session_state:
    st.session_state.current_tokenizer = None

@st.cache_resource
def load_tokenizer(model_name: str):
    """Load and cache the Hugging Face tokenizer"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer: {str(e)}")
        return None

def tokenize_text(text: str, tokenizer, show_tokens: bool = False) -> Dict:
    """Tokenize text using Hugging Face tokenizer"""
    try:
        # Tokenize the text
        tokens = tokenizer.encode(text)
        token_strings = tokenizer.convert_ids_to_tokens(tokens)
        
        # Get detailed analysis
        words = text.split()
        characters = len(text)
        characters_no_spaces = len(text.replace(" ", ""))
        sentences = text.count('.') + text.count('!') + text.count('?')
        
        result = {
            'token_count': len(tokens),
            'tokens': token_strings if show_tokens else [],
            'token_ids': tokens if show_tokens else [],
            'word_count': len(words),
            'character_count': characters,
            'character_count_no_spaces': characters_no_spaces,
            'sentence_count': max(sentences, 1),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'text_preview': text[:100] + "..." if len(text) > 100 else text
        }
        
        return result
    except Exception as e:
        st.error(f"Tokenization Error: {str(e)}")
        return None

# Sidebar
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=100)
    st.title("⚙️ Configuration")
    
    st.subheader("🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose Tokenizer Model",
        [
            "gpt2",
            "bert-base-uncased",
            "meta-llama/Llama-2-7b-hf",
            "mistralai/Mistral-7B-v0.1",
            "google/flan-t5-base",
            "facebook/opt-350m",
            "EleutherAI/gpt-neo-2.7B",
            "t5-small"
        ],
        help="Select the Hugging Face model tokenizer"
    )
    
    if st.button("🔄 Load Tokenizer", width="stretch"):
        with st.spinner(f"Loading {model_choice} tokenizer..."):
            tokenizer = load_tokenizer(model_choice)
            if tokenizer:
                st.session_state.current_tokenizer = tokenizer
                st.session_state.tokenizer_loaded = True
                st.success(f"✅ {model_choice} loaded successfully!")
            else:
                st.error("❌ Failed to load tokenizer")
    
    st.divider()
    
    st.subheader("🎨 Display Options")
    show_tokens = st.checkbox("Show Token Details", value=False, help="Display individual tokens")
    show_token_ids = st.checkbox("Show Token IDs", value=False, help="Display token IDs")
    
    st.divider()
    
    st.subheader("🔍 Quick Stats")
    if st.session_state.tokenization_history:
        total_requests = len(st.session_state.tokenization_history)
        total_tokens = sum(h['token_count'] for h in st.session_state.tokenization_history)
        avg_tokens = total_tokens / total_requests
        st.metric("Total Requests", total_requests)
        st.metric("Total Tokens", f"{total_tokens:,}")
        st.metric("Avg Tokens/Request", f"{avg_tokens:.0f}")
    else:
        st.info("No tokenization history yet")
    
    if st.button("🗑️ Clear History", width="stretch"):
        st.session_state.tokenization_history = []
        st.rerun()

# Main content
st.title("🔤 AI Tokenizer Pro")
st.markdown("### Professional Text Tokenization with Hugging Face")

if not st.session_state.tokenizer_loaded:
    st.warning("⚠️ Please load a tokenizer from the sidebar to get started.")
    
    # Display model information
    st.info("""
    **Available Models:**
    - **GPT-2**: OpenAI's GPT-2 tokenizer (BPE)
    - **BERT**: Google's BERT tokenizer (WordPiece)
    - **LLaMA 2**: Meta's LLaMA 2 tokenizer
    - **Mistral**: Mistral AI's tokenizer
    - **FLAN-T5**: Google's T5 tokenizer
    - **OPT**: Meta's OPT tokenizer
    - **GPT-Neo**: EleutherAI's GPT-Neo tokenizer
    - **T5**: Google's T5 tokenizer
    
    Select a model and click "Load Tokenizer" to begin!
    """)
else:
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Input Text")
        # Use session state to control the text area value
        if 'input_text' not in st.session_state:
            st.session_state.input_text = ""
        
        input_text = st.text_area(
            "Enter your text here",
            value=st.session_state.input_text,
            height=200,
            placeholder="Type or paste your text here to tokenize...",
            label_visibility="collapsed",
            key="text_input"
        )
        
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            tokenize_btn = st.button("🚀 Tokenize", width="stretch")
        with col_btn2:
            if st.button("🔄 Clear", width="stretch", key="clear_text"):
                st.session_state.pop('input_text', None)
                st.rerun()
    
    with col2:
        st.subheader("💡 Quick Tips")
        st.info(f"""
        **Current Model:**
        {model_choice}
        
        **About Tokenization:**
        - Tokens ≠ Words
        - Different models = different tokens
        - Used for model input limits
        - Important for API pricing
        """)
    
    if tokenize_btn and input_text:
        # Update session state with current input
        st.session_state.input_text = input_text
        with st.spinner("🔄 Tokenizing..."):
            result = tokenize_text(input_text, st.session_state.current_tokenizer, show_tokens or show_token_ids)
            
            if result:
                # Add to history (without tokens to save memory)
                history_entry = {k: v for k, v in result.items() if k not in ['tokens', 'token_ids']}
                st.session_state.tokenization_history.append(history_entry)
                
                # Display results
                st.divider()
                st.subheader("📊 Tokenization Results")
                
                # Metrics row
                metric_cols = st.columns(5)
                
                with metric_cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #667eea; margin: 0;">🎯 Tokens</h3>
                        <h2 style="margin: 10px 0;">{result['token_count']:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #667eea; margin: 0;">📝 Words</h3>
                        <h2 style="margin: 10px 0;">{result['word_count']:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #667eea; margin: 0;">🔤 Characters</h3>
                        <h2 style="margin: 10px 0;">{result['character_count']:,}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #667eea; margin: 0;">📄 Sentences</h3>
                        <h2 style="margin: 10px 0;">{result['sentence_count']}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with metric_cols[4]:
                    ratio = result['token_count'] / result['word_count'] if result['word_count'] > 0 else 0
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3 style="color: #667eea; margin: 0;">📊 Token/Word</h3>
                        <h2 style="margin: 10px 0;">{ratio:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Token Display
                if show_tokens and result.get('tokens'):
                    st.divider()
                    st.subheader("🔍 Token Breakdown")
                    
                    tokens_html = '<div class="token-display">'
                    for i, token in enumerate(result['tokens']):
                        token_display = token.replace('Ġ', '▁').replace('Ċ', '↵')
                        if show_token_ids:
                            tokens_html += f'<span class="token-item">{token_display}<br><small>ID: {result["token_ids"][i]}</small></span>'
                        else:
                            tokens_html += f'<span class="token-item">{token_display}</span>'
                    tokens_html += '</div>'
                    
                    st.markdown(tokens_html, unsafe_allow_html=True)
                    st.caption(f"Total: {len(result['tokens'])} tokens • ▁ = space • ↵ = newline")
                
                # Detailed breakdown
                st.divider()
                st.subheader("🔍 Detailed Analysis")
                
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.markdown("**Text Statistics**")
                    st.write(f"- Characters (with spaces): {result['character_count']:,}")
                    st.write(f"- Characters (no spaces): {result['character_count_no_spaces']:,}")
                    st.write(f"- Average word length: {result['character_count_no_spaces'] / max(result['word_count'], 1):.2f} chars")
                    st.write(f"- Average sentence length: {result['word_count'] / max(result['sentence_count'], 1):.2f} words")
                
                with col_detail2:
                    st.markdown("**Tokenization Info**")
                    st.write(f"- Tokenizer: {model_choice}")
                    st.write(f"- Tokens per character: {result['token_count'] / max(result['character_count'], 1):.3f}")
                    st.write(f"- Characters per token: {result['character_count'] / max(result['token_count'], 1):.2f}")
                    st.write(f"- Timestamp: {result['timestamp']}")
                
                # Export options
                st.divider()
                st.subheader("💾 Export Results")
                
                col_export1, col_export2, col_export3 = st.columns(3)
                
                with col_export1:
                    json_data = json.dumps({k: v for k, v in result.items() if k != 'token_ids'}, indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        json_data,
                        file_name=f"tokenization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        width="stretch"
                    )
                
                with col_export2:
                    csv_data = pd.DataFrame([{k: v for k, v in result.items() if k not in ['tokens', 'token_ids']}]).to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        csv_data,
                        file_name=f"tokenization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        width="stretch"
                    )
                
                with col_export3:
                    if result.get('tokens'):
                        tokens_txt = "\n".join([f"{i+1}. {token} (ID: {result['token_ids'][i]})" 
                                               for i, token in enumerate(result['tokens'])])
                        st.download_button(
                            "📥 Download Tokens",
                            tokens_txt,
                            file_name=f"tokens_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            width="stretch"
                        )

# History section
if st.session_state.tokenizer_loaded and st.session_state.tokenization_history:
    st.divider()
    st.subheader("📜 Tokenization History")
    
    history_df = pd.DataFrame(st.session_state.tokenization_history)
    history_df = history_df[['timestamp', 'token_count', 'word_count', 'character_count', 'text_preview']]
    history_df.columns = ['Timestamp', 'Tokens', 'Words', 'Characters', 'Text Preview']
    
    st.dataframe(history_df, width="stretch", hide_index=True)
    
    # Summary statistics
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    with col_stat1:
        st.metric("Total Tokenizations", len(history_df))
    with col_stat2:
        st.metric("Total Tokens", f"{history_df['Tokens'].sum():,}")
    with col_stat3:
        st.metric("Avg Tokens", f"{history_df['Tokens'].mean():.0f}")
    with col_stat4:
        st.metric("Max Tokens", f"{history_df['Tokens'].max():,}")

# Footer
st.divider()
st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>Built with ❤️ using Streamlit & Hugging Face Transformers</p>
        <p style="font-size: 0.9em;">Professional tokenization for AI applications • No API key required</p>
    </div>
""", unsafe_allow_html=True)