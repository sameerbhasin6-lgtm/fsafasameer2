import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
# Error handling for libraries
try:
    from textstat import textstat
    from textblob import TextBlob
except ImportError:
    st.error("Please install missing libraries: pip install textstat textblob")
    st.stop()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity # --- NEW IMPORT FOR COMPARISON ---
from sklearn.linear_model import LinearRegression
import numpy as np
from pypdf import PdfReader
import re
from collections import Counter

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Forensic AI Analyst", page_icon="🕵️‍♂️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 15px; 
        border-radius: 8px; 
        border-left: 5px solid #4e8cff; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
        margin-bottom: 10px;
    }
    .high-risk { border-left: 5px solid #ff4b4b !important; background-color: #fff5f5 !important; }
    .good-metric { border-left: 5px solid #00cc96 !important; background-color: #f0fff4 !important; }
    .metric-label { font-size: 14px; color: #555; margin-bottom: 5px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #222; }
    .metric-sub { font-size: 12px; color: #888; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADER (RED FLAGS) ---
@st.cache_data
def load_red_flag_dictionary():
    """
    Loads the custom anomaly dictionary. 
    """
    try:
        df = pd.read_csv("Annual_Report_Red_Flags.csv") 
    except:
        try:
            # Fallback data
            data = {
                "Word": ["contingent", "estimate", "fluctuate", "litigation", "claim", "uncertainty", "pending", "unresolved", "material", "adverse", "risk", "doubt", "going concern", "restatement", "write-off", "impairment"],
                "Category": ["Uncertainty", "Uncertainty", "Volatility", "Legal", "Legal", "Uncertainty", "Legal", "Legal", "Materiality", "Negative", "Risk", "Viability", "Viability", "Accounting", "Loss", "Loss"]
            }
            df = pd.DataFrame(data)
        except Exception as e:
            st.error(f"Could not load dictionary: {e}")
            return pd.DataFrame()
            
    df['Word'] = df['Word'].str.lower().str.strip()
    return df

# --- 2. TEXT EXTRACTION ---
@st.cache_data
def extract_text_fast(file, start_p=1, end_p=None):
    text = ""
    try:
        reader = PdfReader(file)
        total_pages = len(reader.pages)
        if end_p is None or end_p > total_pages: end_p = total_pages
        
        # Progress bar
        my_bar = st.progress(0, text="Scanning pages...")
        for i in range(start_p - 1, end_p):
            page_text = reader.pages[i].extract_text()
            if page_text:
                text += page_text + "\n"
            my_bar.progress(min(int(((i - start_p + 1) / (end_p - start_p + 1)) * 100), 100))
        my_bar.empty()
        return text, total_pages
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return "", 0

# --- 3. METRICS ENGINE (UPDATED) ---
def analyze_metrics(text, red_flag_df):
    if not text: return None
    
    # A. Basic Counts
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    # B. Readability
    try: fog_index = textstat.gunning_fog(text)
    except: fog_index = 0
    
    # C. Sentiment
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity # -1 to 1
    
    # D. Complex Words
    unique_words = set(words)
    complex_words_list = [w for w in unique_words if textstat.syllable_count(w) >= 3]
    complex_count = sum(1 for w in words if w in complex_words_list)
    complex_pct = (complex_count / total_words) * 100
    
    # E. Custom Red Flag Analysis
    red_flag_set = set(red_flag_df['Word'].unique())
    matched_words = [w for w in words if w in red_flag_set]
    word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict()
    matched_categories = [word_to_cat.get(w, "Unknown") for w in matched_words]

    # --- NEW: ADVANCED LINGUISTIC METRICS (As per Report) ---
    
    # 1. Passive Voice (Heuristic: 'to be' + past participle/ed)
    # Using regex for speed. Matches "was/were/is/are/been" followed by word ending in "ed"
    passive_matches = re.findall(r'\b(am|is|are|was|were|be|been|being)\b\s+\w+ed\b', text.lower())
    passive_count = len(passive_matches)
    passive_ratio = (passive_count / total_words) * 1000 # Per 1000 words
    
    # 2. Uncertainty/Hedging Score (Loughran-McDonald style)
    hedging_words = ["approximate", "estimate", "fluctuate", "indefinite", "maybe", "possible", "likely", "assume", "could", "might"]
    hedging_count = sum(1 for w in words if w in hedging_words)
    uncertainty_score = (hedging_count / total_words) * 100 # Percentage
    
    # 3. Justification/Defensiveness Density
    justification_words = ["because", "due to", "despite", "attributed", "thereby", "explanation", "consequently"]
    just_count = sum(1 for w in words if w in justification_words)
    justification_score = (just_count / total_words) * 100
    
    return {
        "fog_index": fog_index,
        "sentiment": sentiment,
        "complex_pct": complex_pct,
        "total_words": total_words,
        "matched_words": matched_words,
        "matched_categories": matched_categories,
        # New Metrics
        "passive_ratio": passive_ratio,
        "uncertainty_score": uncertainty_score,
        "justification_score": justification_score
    }

# --- NEW: COSINE SIMILARITY FUNCTION ---
def calculate_similarity(text1, text2):
    """Calculates Cosine Similarity between two texts for YoY comparison."""
    try:
        # Create vectors
        documents = [text1, text2]
        count_vectorizer = CountVectorizer(stop_words='english')
        sparse_matrix = count_vectorizer.fit_transform(documents)
        
        # Compute Cosine Similarity
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names_out(), index=['Current', 'Previous'])
        similarity = cosine_similarity(df, df)
        
        return similarity[0][1] # Return the similarity score (0 to 1)
    except Exception as e:
        return 0.0

# --- SIDEBAR (UPDATED) ---
with st.sidebar:
    st.title("🕵️‍♂️ Forensic Settings")
    
    st.subheader("1. Current Year Report")
    uploaded_file = st.file_uploader("Upload Current Report (PDF)", type="pdf", key="curr")
    
    st.subheader("2. Previous Year Report")
    st.caption("Optional: Upload last year's report to detect structural changes.")
    prev_file = st.file_uploader("Upload Previous Report (PDF)", type="pdf", key="prev")
    
    st.markdown("---")
    st.subheader("3. Configuration")
    # Allow user to upload their own dictionary optionally
    uploaded_dict = st.file_uploader("Upload Red Flag List (CSV/Excel)", type=["csv", "xlsx"])
    
    use_all = st.checkbox("Scan Entire Document", value=False)
    start_p, end_p = 1, 50
    if not use_all:
        c1, c2 = st.columns(2)
        start_p = c1.number_input("Start Page", 1, value=50)
        end_p = c2.number_input("End Page", 1, value=100)
    else: end_p = None

# --- MAIN APP ---
if uploaded_file:
    # Load Dictionary
    if uploaded_dict:
        if uploaded_dict.name.endswith('.csv'):
            red_flags_df = pd.read_csv(uploaded_dict)
        else:
            red_flags_df = pd.read_excel(uploaded_dict)
        if 'Word' not in red_flags_df.columns: 
            st.error("Uploaded dictionary must have a 'Word' column.")
            st.stop()
        red_flags_df['Word'] = red_flags_df['Word'].str.lower().str.strip()
    else:
        red_flags_df = load_red_flag_dictionary()

    # Process Current Text
    text, _ = extract_text_fast(uploaded_file, start_p, end_p)
    
    # Process Previous Text (if available)
    prev_text = ""
    similarity_score = None
    if prev_file:
        with st.spinner("Processing Previous Year Report..."):
            prev_text, _ = extract_text_fast(prev_file, start_p, end_p) # Use same page range for fair comparison
            if prev_text and text:
                similarity_score = calculate_similarity(text, prev_text)

    if text:
        metrics = analyze_metrics(text, red_flags_df)
        
        st.title("Forensic Analysis Dashboard")
        st.caption(f"Analyzing {metrics['total_words']:,} words against {len(red_flags_df)} Red Flags.")

        # --- ROW 1: PRIMARY METRICS (Matching Report Layout) ---
        # Need 4 columns: Fog Index, Passive Voice, Uncertainty, YoY Similarity
        col1, col2, col3, col4 = st.columns(4)
        
        # 1. Fog Index
        with col1:
            risk = "High" if metrics['fog_index'] > 18 else "Low"
            color = "high-risk" if metrics['fog_index'] > 18 else "good-metric"
            st.markdown(f"""
            <div class="metric-card {color}">
                <div class="metric-label">Fog Index</div>
                <div class="metric-value">{metrics['fog_index']:.1f}</div>
                <div class="metric-sub">{risk} Complexity</div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Passive Voice Ratio (NEW)
        with col2:
            pv_risk = "high-risk" if metrics['passive_ratio'] > 40 else "" # Threshold example
            st.markdown(f"""
            <div class="metric-card {pv_risk}">
                <div class="metric-label">Passive Voice</div>
                <div class="metric-value">{metrics['passive_ratio']:.1f}</div>
                <div class="metric-sub">Per 1k Words</div>
            </div>
            """, unsafe_allow_html=True)

        # 3. Uncertainty/Hedging (NEW)
        with col3:
            u_risk = "high-risk" if metrics['uncertainty_score'] > 2.0 else ""
            st.markdown(f"""
            <div class="metric-card {u_risk}">
                <div class="metric-label">Uncertainty</div>
                <div class="metric-value">{metrics['uncertainty_score']:.2f}%</div>
                <div class="metric-sub">Hedging Frequency</div>
            </div>
            """, unsafe_allow_html=True)

        # 4. YoY Similarity (NEW)
        with col4:
            if similarity_score is not None:
                # Low similarity (<0.8) is a risk (structural rewrite)
                sim_pct = similarity_score * 100
                sim_risk = "high-risk" if sim_pct < 80 else "good-metric" 
                sim_label = "Structural Rewrite" if sim_pct < 80 else "Consistent"
                
                st.markdown(f"""
                <div class="metric-card {sim_risk}">
                    <div class="metric-label">YoY Similarity</div>
                    <div class="metric-value">{sim_pct:.1f}%</div>
                    <div class="metric-sub">{sim_label}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">YoY Similarity</div>
                    <div class="metric-value">N/A</div>
                    <div class="metric-sub">Upload prev report</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # --- ROW 1.5: SENTIMENT & DEFENSIVENESS ---
        c_sent1, c_sent2, c_sent3 = st.columns(3)
        
        with c_sent1:
             # Sentiment Score
            sent = metrics['sentiment']
            if sent > 0.2: s_label = "Overly Positive"
            elif sent < -0.1: s_label = "Negative"
            else: s_label = "Neutral"
            
            st.metric("Tone / Sentiment", f"{sent:.2f}", s_label)

        with c_sent2:
            # Justification Score (Defensiveness)
            # High justification = Defensiveness
            def_label = "High Defensiveness" if metrics['justification_score'] > 0.5 else "Normal"
            st.metric("Justification Density", f"{metrics['justification_score']:.2f}%", def_label, delta_color="inverse")
            
        with c_sent3:
            st.metric("Complex Words %", f"{metrics['complex_pct']:.1f}%")

        st.markdown("---")

        # --- ROW 2: CUSTOM WORD CLOUD & STATS ---
        st.subheader("🚩 Red Flag Analysis")
        
        c_cloud, c_stats = st.columns([2, 1])
        
        with c_cloud:
            if metrics['matched_words']:
                text_for_cloud = " ".join(metrics['matched_words'])
                wc = WordCloud(
                    background_color="white", 
                    colormap="Reds", 
                    width=600, 
                    height=350,
                    collocations=False
                ).generate(text_for_cloud)
                
                fig_wc, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No Red Flag words found in this document.")

        with c_stats:
            if metrics['matched_words']:
                st.markdown("**Top Anomalous Words**")
                counts = Counter(metrics['matched_words'])
                df_counts = pd.DataFrame(counts.most_common(10), columns=["Word", "Count"])
                st.dataframe(df_counts, hide_index=True, use_container_width=True, height=150)
                
                st.markdown("**Risk Category Distribution**")
                cat_counts = Counter(metrics['matched_categories'])
                fig_cat = px.pie(names=list(cat_counts.keys()), values=list(cat_counts.values()), hole=0.4)
                fig_cat.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=150, showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")

        # --- ROW 3: REGRESSION BENCHMARK ---
        st.subheader("📈 Fraud Risk Regression Model")
        
        col_reg_desc, col_reg_chart = st.columns([1, 2])
        
        with col_reg_desc:
            st.markdown("""
            **Industry Benchmark:**
            Comparing your document's Fog Index against a simulated dataset of 50 companies.
            
            * **X-Axis:** Fog Index (Complexity)
            * **Y-Axis:** Fraud Risk Score
            * **Red X:** Your Document
            """)
            
        with col_reg_chart:
            # 1. Generate Dummy Data
            np.random.seed(42)
            bench_fog = np.random.normal(16, 3, 50) 
            bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
            
            df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
            
            # 2. Train Model
            model = LinearRegression()
            X = df_bench[["Fog Index"]]
            y = df_bench["Risk Score"]
            model.fit(X, y)
            
            # 3. Predict for Current File
            curr_fog = metrics['fog_index']
            pred_risk = model.predict([[curr_fog]])[0]
            
            # 4. Plot
            fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4)
            
            # Add Trend Line
            line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
            line_y = model.predict(line_x)
            fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=line_y, mode='lines', name='Industry Trend', line=dict(color='gray', dash='dash')))
            
            # Add Current File Marker
            fig_reg.add_traces(go.Scatter(
                x=[curr_fog], 
                y=[pred_risk], 
                mode='markers+text', 
                marker=dict(color='red', size=15, symbol='x'),
                name='Your File',
                text=["YOU"],
                textposition="top center"
            ))
            fig_reg.update_layout(title="Complexity vs. Risk Benchmark")
            
            st.plotly_chart(fig_reg, use_container_width=True)

else:
    st.info("Upload a PDF to begin analysis.")
