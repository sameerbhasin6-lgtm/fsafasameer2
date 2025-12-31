import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import re
from collections import Counter
import numpy as np
from pypdf import PdfReader

# Error handling for libraries
try:
    from textstat import textstat
    from textblob import TextBlob
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("Please install missing libraries: pip install textstat textblob scikit-learn")
    st.stop()

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
    .medium-risk { border-left: 5px solid #ffa500 !important; background-color: #fffaf0 !important; }
    .good-metric { border-left: 5px solid #00cc96 !important; background-color: #f0fff4 !important; }
    .metric-label { font-size: 14px; color: #555; margin-bottom: 5px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #222; }
    .metric-sub { font-size: 12px; color: #888; }
    .section-header { font-size: 18px; font-weight: bold; margin-top: 20px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. DATA LOADER ---
@st.cache_data
def load_red_flag_dictionary():
    try:
        df = pd.read_csv("Annual_Report_Red_Flags.csv") 
    except:
        data = {
            "Word": ["contingent", "estimate", "fluctuate", "litigation", "claim", "uncertainty", "pending", "unresolved", "material", "adverse", "risk", "doubt", "going concern", "restatement", "write-off", "impairment"],
            "Category": ["Uncertainty", "Uncertainty", "Volatility", "Legal", "Legal", "Uncertainty", "Legal", "Legal", "Materiality", "Negative", "Risk", "Viability", "Viability", "Accounting", "Loss", "Loss"]
        }
        df = pd.DataFrame(data)
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

# --- 3. HELPER FUNCTIONS FOR MODULES ---

def get_justification_score(text):
    """Calculates density of defensive/explanatory words."""
    words = re.findall(r'\w+', text.lower())
    total = len(words) if words else 1
    
    justification_lexicon = [
        "because", "due to", "despite", "attributed", "thereby", "explanation", 
        "consequently", "factors", "primarily", "driven by", "offset by", 
        "impacted by", "result of", "account of", "reason"
    ]
    
    count = sum(1 for w in words if w in justification_lexicon)
    density = (count / total) * 100
    
    # Qualitative Label
    if density > 1.5: label = "High"
    elif density > 0.8: label = "Medium"
    else: label = "Low"
    
    return density, label

def analyze_sections(text):
    """
    Extracts context windows around sensitive keywords to create the Risk Heatmap.
    """
    keywords = {
        "Revenue Recognition": ["revenue", "sales", "turnover", "recognition"],
        "Provisions & Contingencies": ["provision", "contingent", "legal", "lawsuit"],
        "Related Party": ["related party", "associate", "subsidiary", "arm's length"],
        "Other Income/Expenses": ["exceptional", "extraordinary", "other income", "write-off"]
    }
    
    results = {}
    
    text_lower = text.lower()
    
    for section, keys in keywords.items():
        # Find first occurrence of any key to anchor the section
        found = False
        snippet = ""
        for k in keys:
            idx = text_lower.find(k)
            if idx != -1:
                # Extract a window of 500 words (approx 3000 chars)
                start = max(0, idx - 500)
                end = min(len(text), idx + 3000)
                snippet = text[start:end]
                found = True
                break
        
        if found:
            # Analyze snippet
            fog = textstat.gunning_fog(snippet)
            jd_score, jd_label = get_justification_score(snippet)
            
            # Risk logic for snippet
            risk = "Low"
            if fog > 20 or jd_label == "High": risk = "High"
            elif fog > 17 or jd_label == "Medium": risk = "Medium"
            
            results[section] = {
                "risk": risk,
                "fog": fog,
                "justification": jd_label,
                "snippet_found": True
            }
        else:
            results[section] = {"risk": "N/A", "snippet_found": False}
            
    return results

def calculate_manipulation_score(metrics, one_time_count):
    """
    Composite Score Logic:
    1. High Uncertainty (Hedging)
    2. High Justification (Defensiveness)
    3. Repeated 'One-time' items
    4. Passive Voice
    """
    score = 0
    reasons = []
    
    # Factor 1: Hedging
    if metrics['uncertainty_score'] > 1.5: 
        score += 3
        reasons.append("Excessive Hedging/Uncertainty")
    elif metrics['uncertainty_score'] > 1.0:
        score += 1
        
    # Factor 2: Defensiveness
    if metrics['justification_score'] > 1.2:
        score += 3
        reasons.append("Highly Defensive Tone")
    elif metrics['justification_score'] > 0.8:
        score += 1
        
    # Factor 3: 'One-Time' Items (Structural anomaly)
    if one_time_count > 5:
        score += 2
        reasons.append("Repeated 'One-Time' Adjustments")
        
    # Factor 4: Passive Voice
    if metrics['passive_ratio'] > 35:
        score += 2
        reasons.append("Excessive Passive Voice (Distancing)")
        
    # Categorize
    if score >= 6: risk_label = "High"
    elif score >= 3: risk_label = "Moderate"
    else: risk_label = "Low"
    
    return risk_label, reasons

# --- 4. CORE ANALYZER ---
def analyze_metrics(text, red_flag_df):
    if not text: return None
    
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    # 1. Standard Metrics
    try: fog_index = textstat.gunning_fog(text)
    except: fog_index = 0
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    unique_words = set(words)
    complex_words_list = [w for w in unique_words if textstat.syllable_count(w) >= 3]
    complex_pct = (sum(1 for w in words if w in complex_words_list) / total_words) * 100
    
    # 2. Red Flags
    red_flag_set = set(red_flag_df['Word'].unique())
    matched_words = [w for w in words if w in red_flag_set]
    word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict()
    matched_categories = [word_to_cat.get(w, "Unknown") for w in matched_words]

    # 3. New Linguistic Modules
    
    # Passive Voice
    passive_matches = re.findall(r'\b(am|is|are|was|were|be|been|being)\b\s+\w+ed\b', text.lower())
    passive_ratio = (len(passive_matches) / total_words) * 1000
    
    # Hedging (Uncertainty)
    hedging_words = ["approximate", "estimate", "fluctuate", "indefinite", "maybe", "possible", "likely", "assume", "could", "might", "subject to"]
    uncertainty_score = (sum(1 for w in words if w in hedging_words) / total_words) * 100
    
    # Justification Density (Global)
    justification_score, justification_label = get_justification_score(text)
    
    # 'One-Time' / Exceptional Item counter
    one_time_terms = ["exceptional", "one-time", "non-recurring", "special item", "restructuring cost"]
    one_time_count = sum(text.lower().count(t) for t in one_time_terms)
    
    return {
        "fog_index": fog_index,
        "sentiment": sentiment,
        "complex_pct": complex_pct,
        "total_words": total_words,
        "matched_words": matched_words,
        "matched_categories": matched_categories,
        "passive_ratio": passive_ratio,
        "uncertainty_score": uncertainty_score,
        "justification_score": justification_score,
        "justification_label": justification_label,
        "one_time_count": one_time_count
    }

def calculate_similarity(text1, text2):
    try:
        documents = [text1, text2]
        count_vectorizer = CountVectorizer(stop_words='english')
        sparse_matrix = count_vectorizer.fit_transform(documents)
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names_out())
        similarity = cosine_similarity(df, df)
        return similarity[0][1]
    except:
        return 0.0

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Forensic Settings")
    st.subheader("1. Current Year Report")
    uploaded_file = st.file_uploader("Upload Current Report (PDF)", type="pdf", key="curr")
    
    st.subheader("2. Previous Year Report")
    st.caption("Required for Tone Drift & Boilerplate checks.")
    prev_file = st.file_uploader("Upload Previous Report (PDF)", type="pdf", key="prev")
    
    st.markdown("---")
    uploaded_dict = st.file_uploader("Upload Red Flag List (CSV)", type=["csv"])
    
    use_all = st.checkbox("Scan Entire Document", value=False)
    start_p, end_p = 1, 50
    if not use_all:
        c1, c2 = st.columns(2)
        start_p = c1.number_input("Start Page", 1, value=50)
        end_p = c2.number_input("End Page", 1, value=100)
    else: end_p = None

# --- MAIN APP ---
if uploaded_file:
    # Load Dict
    if uploaded_dict:
        red_flags_df = pd.read_csv(uploaded_dict)
        red_flags_df['Word'] = red_flags_df['Word'].str.lower().str.strip()
    else:
        red_flags_df = load_red_flag_dictionary()

    # Process Texts
    text, _ = extract_text_fast(uploaded_file, start_p, end_p)
    
    prev_text = ""
    similarity_score = None
    if prev_file:
        with st.spinner("Comparing with Previous Year..."):
            prev_text, _ = extract_text_fast(prev_file, start_p, end_p)
            if prev_text and text:
                similarity_score = calculate_similarity(text, prev_text)

    if text:
        metrics = analyze_metrics(text, red_flags_df)
        
        # Calculate Composite Manipulation Risk
        manip_risk, manip_reasons = calculate_manipulation_score(metrics, metrics['one_time_count'])
        
        st.title("Forensic Analysis Dashboard")
        st.caption(f"Analyzing {metrics['total_words']:,} words. Focus: Linguistic Manipulation & Tone.")

        # --- MODULE 2: MANIPULATION RISK SCORE (CORE ADDITION) ---
        st.markdown("### 🚨 Linguistic Manipulation Risk Score")
        
        m_col1, m_col2 = st.columns([1, 3])
        
        with m_col1:
            color = "good-metric"
            if manip_risk == "High": color = "high-risk"
            elif manip_risk == "Moderate": color = "medium-risk"
            
            st.markdown(f"""
            <div class="metric-card {color}" style="text-align:center;">
                <div class="metric-label">Composite Risk Level</div>
                <div class="metric-value">{manip_risk}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.info("**Risk Drivers:**")
            if manip_reasons:
                for reason in manip_reasons:
                    st.write(f"• {reason}")
            else:
                st.write("• No significant linguistic manipulation signals detected.")
            st.caption("Based on aggregation of hedging, justification density, passive voice, and structural anomalies.")

        st.markdown("---")

        # --- EXISTING METRICS ROW ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            risk = "High" if metrics['fog_index'] > 18 else "Low"
            color = "high-risk" if metrics['fog_index'] > 18 else "good-metric"
            st.markdown(f"""<div class="metric-card {color}"><div class="metric-label">Fog Index</div><div class="metric-value">{metrics['fog_index']:.1f}</div><div class="metric-sub">{risk} Complexity</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Passive Voice</div><div class="metric-value">{metrics['passive_ratio']:.1f}</div><div class="metric-sub">Per 1k Words</div></div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Uncertainty</div><div class="metric-value">{metrics['uncertainty_score']:.2f}%</div><div class="metric-sub">Hedging Words</div></div>""", unsafe_allow_html=True)
        with col4:
            # MODULE 4: REPETITION/BOILERPLATE
            if similarity_score is not None:
                sim_pct = similarity_score * 100
                if sim_pct > 98: 
                    sim_label = "High Boilerplate"
                    sim_risk = "high-risk" # Too much repetition = lazy disclosure
                elif sim_pct < 80:
                    sim_label = "Structural Change"
                    sim_risk = "medium-risk"
                else:
                    sim_label = "Consistent"
                    sim_risk = "good-metric"
                
                st.markdown(f"""<div class="metric-card {sim_risk}"><div class="metric-label">YoY Similarity</div><div class="metric-value">{sim_pct:.1f}%</div><div class="metric-sub">{sim_label}</div></div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="metric-card"><div class="metric-label">YoY Similarity</div><div class="metric-value">N/A</div><div class="metric-sub">Upload prev report</div></div>""", unsafe_allow_html=True)

        # --- MODULE 3: TONE DRIFT TRACKER ---
        if prev_file and prev_text:
            st.markdown("### 📉 Disclosure Tone Drift (Year-on-Year)")
            prev_metrics = analyze_metrics(prev_text, red_flags_df)
            
            t1, t2, t3 = st.columns(3)
            
            # 1. Justification Drift
            curr_j = metrics['justification_score']
            prev_j = prev_metrics['justification_score']
            delta_j = curr_j - prev_j
            drift_label = "Stable"
            if delta_j > 0.2: drift_label = "More Defensive 🚩"
            elif delta_j < -0.2: drift_label = "Less Defensive"
            
            with t1:
                st.metric("Defensiveness (Justification)", f"{curr_j:.2f}%", f"{delta_j:.2f}% YoY", delta_color="inverse")
                st.caption(f"Trend: {drift_label}")

            # 2. Sentiment Drift
            curr_s = metrics['sentiment']
            prev_s = prev_metrics['sentiment']
            delta_s = curr_s - prev_s
            with t2:
                st.metric("Sentiment Polarity", f"{curr_s:.2f}", f"{delta_s:.2f} YoY")
                
            # 3. Complexity Drift
            curr_f = metrics['fog_index']
            prev_f = prev_metrics['fog_index']
            delta_f = curr_f - prev_f
            with t3:
                st.metric("Complexity (Fog)", f"{curr_f:.1f}", f"{delta_f:.1f} YoY", delta_color="inverse")

        st.markdown("---")
        
        # --- MODULE 5: LINGUISTIC RISK HEATMAP ---
        st.markdown("### 🔥 Linguistic Risk Heatmap (Narrative Sections)")
        st.write("Analysis of linguistic patterns in specific sensitive disclosure areas.")
        
        section_data = analyze_sections(text)
        
        # Create a grid layout for sections
        h_col1, h_col2 = st.columns(2)
        cols = [h_col1, h_col2]
        
        for idx, (section_name, data) in enumerate(section_data.items()):
            with cols[idx % 2]:
                if data['snippet_found']:
                    color = "green"
                    if data['risk'] == "High": color = "red"
                    elif data['risk'] == "Medium": color = "orange"
                    
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px;">
                        <strong>{section_name}</strong> <span style="color:{color}; font-weight:bold; float:right;">{data['risk']} Risk</span>
                        <br><span style="font-size:12px; color:#666;">Fog Index: {data['fog']:.1f} | Justification: {data['justification']}</span>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px; opacity: 0.6;">
                        <strong>{section_name}</strong>
                        <br><span style="font-size:12px; color:#666;">Section keyword not found in scanned text.</span>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        
        # --- ORIGINAL REGRESSION & WORD CLOUD (KEPT AS IS) ---
        st.subheader("Visual Forensics")
        
        # Cloud
        if metrics['matched_words']:
            text_for_cloud = " ".join(metrics['matched_words'])
            wc = WordCloud(background_color="white", colormap="Reds", width=600, height=300).generate(text_for_cloud)
            fig_wc, ax = plt.subplots(figsize=(6, 3))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig_wc)
        
        # Regression
        np.random.seed(42)
        bench_fog = np.random.normal(16, 3, 50) 
        bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
        df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
        model = LinearRegression()
        model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
        pred_risk = model.predict([[metrics['fog_index']]])[0]
        
        fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4, title="Industry Benchmark (Simulated)")
        line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
        fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=model.predict(line_x), mode='lines', name='Trend', line=dict(color='gray', dash='dash')))
        fig_reg.add_traces(go.Scatter(x=[metrics['fog_index']], y=[pred_risk], mode='markers', marker=dict(color='red', size=15, symbol='x'), name='You'))
        
        st.plotly_chart(fig_reg, use_container_width=True)

else:
    st.info("Upload a PDF to begin analysis.")
