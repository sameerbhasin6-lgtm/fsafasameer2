import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from collections import Counter
import numpy as np
from pypdf import PdfReader

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
    .metric-value { font-size: 24px; font-weight: bold; color: #222; }
    .metric-sub { font-size: 12px; color: #888; }
    .section-header { font-size: 20px; font-weight: bold; margin-top: 25px; margin-bottom: 15px; border-bottom: 1px solid #eee; padding-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

# --- ERROR HANDLING FOR LIBRARIES ---
try:
    from textstat import textstat
    from textblob import TextBlob
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.linear_model import LinearRegression
except ImportError:
    st.error("Please install missing libraries: pip install textstat textblob scikit-learn wordcloud plotly pypdf")
    st.stop()

# --- 1. DATA LOADER ---
@st.cache_data
def load_red_flag_dictionary():
    try:
        df = pd.read_csv("Annual_Report_Red_Flags.csv") 
    except:
        # Fallback dictionary
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

# --- 3. CORE ANALYTICS ENGINE ---

def get_justification_score(text):
    words = re.findall(r'\w+', text.lower())
    total = len(words) if words else 1
    justification_lexicon = ["because", "due to", "despite", "attributed", "thereby", "explanation", "consequently", "factors", "primarily", "driven by"]
    count = sum(1 for w in words if w in justification_lexicon)
    density = (count / total) * 100
    if density > 1.2: label = "High"
    elif density > 0.6: label = "Medium"
    else: label = "Low"
    return density, label

def calculate_manipulation_score(metrics, one_time_count):
    score = 0
    reasons = []
    
    # 1. Hedging
    if metrics['uncertainty_score'] > 1.8: 
        score += 3; reasons.append("Excessive Hedging")
    elif metrics['uncertainty_score'] > 1.2: score += 1
        
    # 2. Defensiveness
    if metrics['justification_score'] > 1.0:
        score += 3; reasons.append("High Defensiveness")
    elif metrics['justification_score'] > 0.7: score += 1
        
    # 3. 'One-Time' Items
    if one_time_count > 5:
        score += 2; reasons.append("Repeated 'One-Time' Items")
        
    if score >= 5: risk_label = "High"
    elif score >= 3: risk_label = "Medium"
    else: risk_label = "Low"
    
    return risk_label, reasons

def calculate_similarity(text1, text2):
    try:
        documents = [text1, text2]
        cv = CountVectorizer(stop_words='english')
        matrix = cv.fit_transform(documents)
        return cosine_similarity(matrix)[0][1]
    except: return 0.0

def analyze_metrics(text, red_flag_df):
    if not text: return None
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if words else 1
    
    # Basic
    try: fog_index = textstat.gunning_fog(text)
    except: fog_index = 0
    sentiment = TextBlob(text).sentiment.polarity
    
    # Red Flags
    red_flag_set = set(red_flag_df['Word'].unique())
    matched_words = [w for w in words if w in red_flag_set]
    word_to_cat = pd.Series(red_flag_df.Category.values, index=red_flag_df.Word).to_dict()
    matched_categories = [word_to_cat.get(w, "Unknown") for w in matched_words]
    
    # Specifics
    passive_matches = re.findall(r'\b(am|is|are|was|were|be|been|being)\b\s+\w+ed\b', text.lower())
    passive_ratio = (len(passive_matches) / total_words) * 1000
    
    hedging_words = ["approximate", "estimate", "fluctuate", "indefinite", "maybe", "possible", "likely", "assume", "could", "might"]
    uncertainty_score = (sum(1 for w in words if w in hedging_words) / total_words) * 100
    
    just_score, just_label = get_justification_score(text)
    
    one_time_count = sum(text.lower().count(t) for t in ["exceptional", "one-time", "non-recurring", "restructuring"])
    
    return {
        "fog_index": fog_index,
        "sentiment": sentiment,
        "total_words": total_words,
        "matched_words": matched_words,
        "matched_categories": matched_categories,
        "passive_ratio": passive_ratio,
        "uncertainty_score": uncertainty_score,
        "justification_score": just_score,
        "justification_label": just_label,
        "one_time_count": one_time_count
    }

# --- SIDEBAR ---
with st.sidebar:
    st.title("🕵️‍♂️ Forensic Settings")
    
    # Financial Inputs for Credit Scoring (New)
    st.subheader("📊 Financial Inputs (For Credit Score)")
    st.caption("Enter key figures from the Balance Sheet (in Millions)")
    net_debt = st.number_input("Net Debt", value=0.0)
    ebitda = st.number_input("EBITDA", value=0.0)
    interest_expense = st.number_input("Interest Expense", value=0.0)
    
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Current Report (PDF)", type="pdf", key="curr")
    prev_file = st.file_uploader("Upload Previous Report (PDF)", type="pdf", key="prev")
    
    st.markdown("---")
    uploaded_dict = st.file_uploader("Upload Red Flag List (CSV)", type=["csv"])
    
    use_all = st.checkbox("Scan Entire Document", value=False)
    if not use_all:
        c1, c2 = st.columns(2)
        start_p = c1.number_input("Start Page", 1, value=50)
        end_p = c2.number_input("End Page", 1, value=100)
    else: start_p, end_p = 1, None

# --- MAIN APP ---
if uploaded_file:
    # 1. Load Data & Text
    if uploaded_dict:
        red_flags_df = pd.read_csv(uploaded_dict)
        red_flags_df['Word'] = red_flags_df['Word'].str.lower().str.strip()
    else:
        red_flags_df = load_red_flag_dictionary()

    text, _ = extract_text_fast(uploaded_file, start_p, end_p)
    
    prev_text = ""
    similarity_score = None
    if prev_file:
        prev_text, _ = extract_text_fast(prev_file, start_p, end_p)
        if prev_text and text:
            similarity_score = calculate_similarity(text, prev_text)

    if text:
        metrics = analyze_metrics(text, red_flags_df)
        manip_risk, manip_reasons = calculate_manipulation_score(metrics, metrics['one_time_count'])
        
        st.title("Forensic Analysis Dashboard")
        st.caption(f"Analyzing {metrics['total_words']:,} words against {len(red_flags_df)} Red Flags.")

        # --- ROW 1: FLASHCARDS ---
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            risk = "High" if metrics['fog_index'] > 18 else "Low"
            color = "high-risk" if metrics['fog_index'] > 18 else "good-metric"
            st.markdown(f"""<div class="metric-card {color}"><div class="metric-label">Fog Index</div><div class="metric-value">{metrics['fog_index']:.1f}</div><div class="metric-sub">{risk} Complexity</div></div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class="metric-card"><div class="metric-label">Tone (Sentiment)</div><div class="metric-value">{metrics['sentiment']:.2f}</div><div class="metric-sub">-1 (Neg) to +1 (Pos)</div></div>""", unsafe_allow_html=True)
        with col3:
             st.markdown(f"""<div class="metric-card"><div class="metric-label">Uncertainty</div><div class="metric-value">{metrics['uncertainty_score']:.2f}%</div><div class="metric-sub">Hedging Frequency</div></div>""", unsafe_allow_html=True)
        with col4:
            sim_val = f"{similarity_score*100:.1f}%" if similarity_score else "N/A"
            st.markdown(f"""<div class="metric-card"><div class="metric-label">YoY Similarity</div><div class="metric-value">{sim_val}</div><div class="metric-sub">Boilerplate Check</div></div>""", unsafe_allow_html=True)

        st.markdown("---")

        # --- ROW 2: RED FLAG ANALYSIS (Restored Layout) ---
        st.markdown('<div class="section-header">🚩 Red Flag Analysis</div>', unsafe_allow_html=True)
        
        c_cloud, c_stats = st.columns([2, 1]) # 2:1 Ratio as per screenshot
        
        with c_cloud:
            if metrics['matched_words']:
                text_for_cloud = " ".join(metrics['matched_words'])
                wc = WordCloud(background_color="white", colormap="Reds", width=800, height=400).generate(text_for_cloud)
                fig_wc, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                st.pyplot(fig_wc)
            else:
                st.info("No Red Flags Found.")

        with c_stats:
            if metrics['matched_words']:
                st.markdown("**Top Anomalous Words Found**")
                counts = Counter(metrics['matched_words'])
                df_counts = pd.DataFrame(counts.most_common(5), columns=["Word", "Count"])
                st.dataframe(df_counts, hide_index=True, use_container_width=True)
                
                # Primary Category
                cat_counts = Counter(metrics['matched_categories'])
                top_cat = cat_counts.most_common(1)[0]
                st.markdown("**Primary Anomaly Category**")
                st.info(f"🚨 **{top_cat[0]}** ({top_cat[1]} hits)")
                
                # Pie Chart
                fig_cat = px.pie(names=list(cat_counts.keys()), values=list(cat_counts.values()), hole=0.5)
                fig_cat.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200, showlegend=False)
                st.plotly_chart(fig_cat, use_container_width=True)

        st.markdown("---")

        # --- ROW 3: REGRESSION MODEL (Restored Layout) ---
        st.markdown('<div class="section-header">📈 Fraud Risk Regression Model</div>', unsafe_allow_html=True)
        
        r_desc, r_chart = st.columns([1, 2]) # 1:2 Ratio as per screenshot
        
        with r_desc:
            st.markdown("""
            **Comparing your document's Fog Index against an Industry Benchmark dataset.**
            
            **How this works:**
            We simulated a dataset of 50 companies.
            
            * **X-Axis:** Fog Index (Complexity)
            * **Y-Axis:** Fraud Risk Score
            
            The **Red X** represents the uploaded document. If it falls high on the line, the complexity suggests higher risk.
            """)
            st.caption("High Fog Index correlates with attempts to obfuscate poor performance.")
        
        with r_chart:
            # Simulation
            np.random.seed(42)
            bench_fog = np.random.normal(16, 3, 50) 
            bench_risk = (bench_fog * 2.5) + np.random.normal(0, 8, 50)
            df_bench = pd.DataFrame({"Fog Index": bench_fog, "Risk Score": bench_risk})
            
            model = LinearRegression()
            model.fit(df_bench[["Fog Index"]], df_bench["Risk Score"])
            pred_risk = model.predict([[metrics['fog_index']]])[0]
            
            fig_reg = px.scatter(df_bench, x="Fog Index", y="Risk Score", opacity=0.4, title="Industry Benchmark Analysis")
            line_x = np.linspace(df_bench["Fog Index"].min(), df_bench["Fog Index"].max(), 100).reshape(-1, 1)
            fig_reg.add_traces(go.Scatter(x=line_x.flatten(), y=model.predict(line_x), mode='lines', name='Industry Trend', line=dict(color='gray', dash='dash')))
            fig_reg.add_traces(go.Scatter(x=[metrics['fog_index']], y=[pred_risk], mode='markers+text', marker=dict(color='red', size=20, symbol='x'), name='Your File', text=["YOU"], textposition="top center"))
            fig_reg.update_layout(height=400)
            st.plotly_chart(fig_reg, use_container_width=True)

        st.markdown("---")

        # --- ROW 4: CREDIT RISK & INTENT ANALYSIS (NEW SECTION) ---
        st.markdown('<div class="section-header">💳 Credit Risk & Intent Analysis</div>', unsafe_allow_html=True)
        
        # We need 3 columns for the 3 sub-modules requested
        cr_col1, cr_col2, cr_col3 = st.columns(3)
        
        # 1. Credit Scoring Engine (Financials)
        with cr_col1:
            st.subheader("1. Credit Scoring Engine")
            if ebitda > 0:
                net_leverage = net_debt / ebitda
                icr = ebitda / interest_expense if interest_expense > 0 else 0
                
                # Scoring Logic
                credit_risk = "Low"
                if net_leverage > 4 or icr < 1.5: credit_risk = "High"
                elif net_leverage > 3 or icr < 3: credit_risk = "Medium"
                
                color_cr = "high-risk" if credit_risk == "High" else "good-metric"
                st.markdown(f"""
                <div class="metric-card {color_cr}">
                    <div class="metric-label">Credit Risk Level</div>
                    <div class="metric-value">{credit_risk}</div>
                    <div class="metric-sub">Lev: {net_leverage:.1f}x | ICR: {icr:.1f}x</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Enter financial data in the sidebar to calculate Credit Risk.")

        # 2. Manipulation Risk Score (NLP)
        with cr_col2:
            st.subheader("2. Manipulation Risk")
            m_color = "high-risk" if manip_risk == "High" else ("medium-risk" if manip_risk == "Medium" else "good-metric")
            
            st.markdown(f"""
            <div class="metric-card {m_color}">
                <div class="metric-label">Linguistic Manipulation</div>
                <div class="metric-value">{manip_risk}</div>
            </div>
            """, unsafe_allow_html=True)
            
            if manip_reasons:
                for r in manip_reasons:
                    st.caption(f"🚩 {r}")
            else:
                st.caption("✅ No manipulation signals.")

        # 3. Boilerplate Detector (Comparison)
        with cr_col3:
            st.subheader("3. Boilerplate Detector")
            if similarity_score:
                sim_pct = similarity_score * 100
                bp_status = "High Repetition" if sim_pct > 95 else "Genuine Updates"
                bp_color = "medium-risk" if sim_pct > 95 else "good-metric"
                
                st.markdown(f"""
                <div class="metric-card {bp_color}">
                    <div class="metric-label">Disclosure Copying</div>
                    <div class="metric-value">{sim_pct:.1f}%</div>
                    <div class="metric-sub">{bp_status}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Upload Previous Year Report to enable.")

        st.markdown("---")

        # --- CONCLUSION ---
        st.markdown("### 📝 Executive Conclusion")
        
        # Generate dynamic conclusion text
        conclusion_text = f"The analysis of the uploaded Annual Report indicates a **{manip_risk}** risk of linguistic manipulation. "
        
        if metrics['fog_index'] > 18:
            conclusion_text += "The document is **highly complex (Fog Index > 18)**, suggesting potential obfuscation of information. "
        else:
            conclusion_text += "The readability is within normal ranges. "
            
        if metrics['matched_words']:
            top_cat = Counter(metrics['matched_categories']).most_common(1)[0][0]
            conclusion_text += f"A significant number of red flags related to **{top_cat}** were detected. "
            
        if similarity_score and similarity_score > 0.95:
            conclusion_text += "Furthermore, the **high similarity (>95%)** to the previous year suggests 'lazy disclosure' without meaningful updates. "
            
        st.success(conclusion_text)

else:
    st.info("Upload a PDF to begin analysis.")
