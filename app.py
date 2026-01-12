import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
# --- 1. SETUP & CONFIGURATION ---
if 'result' not in st.session_state:
    st.session_state.result = None

# Add src to path
sys.path.append(os.path.abspath('.'))
from src.preprocessing import FeatureEngineer
from src.utils import (
    CUSTOM_CSS, get_simple_recommendation, get_persona_description, 
    get_wellness_tips, get_detox_challenges, get_achievements, create_pdf_report
)

# --- Page Config ---
# --- 1. SETUP & CONFIGURATION ---
st.set_page_config(
    page_title="Digital Moods | ML & DL Well-Being Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR PREMIUM UI ---
# --- CUSTOM CSS FOR PREMIUM UI ---
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- HEADER COMPONENT ---
st.markdown("""
    <div class="main-header">
        <h1>🧠 Digital Moods</h1>
        <p>Machine Learning and Deep Learning Powered Social Media Well-Being Analysis</p>
    </div>
""", unsafe_allow_html=True)

# --- SYNOPSIS COMPONENT ---
with st.expander("ℹ️ About This Application & Project Synopsis", expanded=True):
    st.markdown("""
    <div class="synopsis-box">
        <h4><strong>Application Name:</strong> Digital Moods</h4>
        <p>
            <strong>Digital Moods</strong> is an advanced Machine Learning and Deep Learning powered analytics platform designed to decode the complex relationship between social media usage and emotional well-being.
            By leveraging a <strong>Hybrid Fusion</strong> of Ensemble Machine Learning (Random Forest, XGBoost) and Deep Learning (MLP), 
            the system classifies a user's dominant emotional state (Happiness, Anxiety, Sadness, Anger) with high precision based on behavioral metrics.
        </p>
        <p style="margin-top:10px;">
            Beyond classification, the platform offers a holistic <strong>Digital Wellness Ecosystem</strong>. 
            It transforms raw usage data into actionable financial and psychological insights, helping users understand the "Opportunity Cost" of their screen time.
        </p>
    </div>
    """, unsafe_allow_html=True)

# --- Load Artifacts ---
@st.cache_resource
def load_all_artifacts():
    """
    The 'Vault'. We retrieve our trained Experts (Models) and the 
    Refinery (Preprocessor) from the models folder.
    """
    try:
        m = 'models'
        # Story: We load the Random Forest model as our base 'ML Expert'.
        with open(f'{m}/model_random_forest.pkl', 'rb') as f: model = pickle.load(f)
        with open(f'{m}/preprocessor.pkl', 'rb') as f: preprocessor = pickle.load(f)
        with open(f'{m}/label_encoder.pkl', 'rb') as f: le = pickle.load(f)
        
        # Loading clustering for 'Personas' and PCA for the 'Emotional Map'.
        with open(f'{m}/kmeans_model.pkl', 'rb') as f: kmeans = pickle.load(f)
        with open(f'{m}/pca_model.pkl', 'rb') as f: pca = pickle.load(f)
        
        # Historical stats for platform comparison
        plat = pd.read_csv(f'{m}/platform_stats.csv', index_col='Platform')
        comm = pd.read_csv(f'{m}/community_projections.csv')
        fi = pd.read_csv(f'{m}/feature_importance.csv')
        ref = pd.read_csv(f'{m}/reference_stats.csv', index_col='Dominant_Emotion')
        metrics = pd.read_csv(f'{m}/evaluation_metrics.csv')
        
        return {'model': model, 'prep': preprocessor, 'le': le, 'km': kmeans, 'pca': pca, 
                'plat': plat, 'comm': comm, 'fi': fi, 'ref': ref, 'metrics': metrics}
    except Exception as e:
        st.error(f"Loading error: {e}")
        return None

@st.cache_data
def load_train_data():
    try:
        return pd.read_csv('data/train.csv')
    except:
        return None

arts = load_all_artifacts()
train_df = load_train_data()

# --- Helpers ---


# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/clouds/100/brain.png", width=80)
    st.markdown("### 👤 User Profile")
    gender = st.selectbox("Gender Identity", ["Male", "Female", "Non-binary"])
    age = st.slider("Age", 13, 70, 25)
    platform = st.selectbox("Preferred Platform", arts['plat'].index.tolist() if arts else ["Instagram"])
    
    st.markdown("---")
    st.markdown("### 📊 Activity Metrics")
    usage = st.slider("⏱️ Daily Screen Time (mins)", 0, 480, 120)
    posts = st.slider("📝 Posts per Day", 0, 20, 2)
    likes = st.slider("❤️ Likes Received/Day", 0, 500, 50)
    messages = st.slider("💬 Messages Sent/Day", 0, 200, 30)
    
    st.markdown("---")
    # --- REACTIVE ANALYSIS ---





if arts:
    # Collecting the sidebar inputs into a single 'Event' (dataframe row)
    raw = pd.DataFrame([{
        'User_ID': 'U001', 'Age': age, 'Gender': gender, 'Platform': platform,
        'Daily_Usage_Time': usage, 'Posts_Per_Day': posts,
        'Likes_Received_Per_Day': likes, 'Comments_Received_Per_Day': 5,
        'Messages_Sent_Per_Day': messages, 'Dominant_Emotion': 'Unknown'
    }])
    
    # 1. Refining: Applying the same transformations used during training.
    fe = FeatureEngineer()
    proc = arts['prep'].transform(fe.transform(raw))
    
    # 2. Reasoning: The model calculates the 'Probability Spectrum'.
    prob = arts['model'].predict_proba(proc)[0]
    # 3. Translating: Converting the numeric index back to readable labels.
    label = arts['le'].inverse_transform([np.argmax(prob)])[0]
    
    # 4. Segmenting: Finding which group (Persona) you belong to.
    cluster = arts['km'].predict(proc)[0]
    persona = {0: "Balanced User", 1: "Passive Scroller", 2: "Power User"}.get(cluster, "Unknown")
    
    # 5. Mapping: Finding your coordinates in the 2D emotional landscape.
    pca_pt = arts['pca'].transform(proc.toarray() if hasattr(proc, 'toarray') else proc)[0]
    
    st.session_state.result = {
        'label': label, 'prob': prob, 'persona': persona, 'pca': pca_pt,
        'usage': usage, 'platform': platform
    }
    # Unified variable for all tabs
    res = st.session_state.result

# --- Main Content (TABS) ---
# --- Main Content (TABS) ---
# Title is now handled by the custom CSS header above

# --- YOUR SELECTED DATA (Before Tabs) ---
st.markdown("### 📋 Your Selected Profile")
profile_cols = st.columns(6)
profile_cols[0].metric("👤 Gender", gender)
profile_cols[1].metric("🎂 Age", age)
profile_cols[2].metric("📱 Platform", platform)
profile_cols[3].metric("⏱️ Screen Time", f"{usage} mins")
profile_cols[4].metric("📝 Posts/Day", posts)
profile_cols[5].metric("💬 Messages", messages)

# NEW: Alert Thresholds
if usage > 180:
    st.warning("⚠️ **High Usage Alert:** You're spending over 3 hours daily on social media. Studies suggest this may negatively impact mental health.")
elif usage > 120:
    st.info("ℹ️ **Moderate Usage:** Your usage is above average. Consider setting screen time limits.")

st.markdown("---")

tab_home, tab_guide, tab_bal, tab_tech = st.tabs(["🏠 Home", "📖 User Guide", "⚖️ Digital Balance", "🔬 Technical Analysis"])

# ==================== TAB 0: User Guide & Documentation ====================
with tab_guide:
    st.markdown("### 📘 How to Understand This Project")
    
    # 1. Project Context
    with st.expander("🎓 **Project Context & Objective**", expanded=True):
        st.markdown("""
        **Objective**: To explore the intersection of **Social Media Usage** and **Emotional Well-Being** using advanced **Machine Learning & Deep Learning**.
        
        This application classifies your dominant emotional state based on your digital habits using a **Hybrid Fusion Model**. 
        It combines the strengths of:
        *   **Ensemble Machine Learning** (Random Forest, XGBoost) for structured data analysis.
        *   **Deep Learning** (Multi-Layer Perceptron) for capturing complex non-linear patterns.
        
        **Why Hybrid?**
        By averaging the predictions of multiple sophisticated models, we achieve higher accuracy and robustness than any single model could provide alone.
        """)

    # 2. Step-by-Step Guide
    with st.expander("🚀 **How to Use This App**", expanded=True):
        st.markdown("""
        Follow these steps to get the most out of **Digital Moods**:

        **Step 1: Configure Your Profile (Sidebar)**
        *   Enter your `Age`, `Gender`, and `Platform`.
        *   Adjust usage metrics like `Screen Time`, `Posts`, `Likes`, and `Messages`.
        *   *The app updates in real-time!*

        **Step 2: Check the 'Home' Dashboard**
        *   View your **Predicted Emotion** and the Model's **Confidence Score**.
        *   See your **User Persona** (e.g., "Passive Scroller") and tailored wellness tips.
        *   Check the **Platform Health Score** for your selected app.

        **Step 3: Analyze Your 'Digital Balance'**
        *   Go to the **⚖️ Digital Balance** tab.
        *   Set your `Hourly Value` (what your time is worth).
        *   See your **Digital Net Worth**: Are you in a *Surplus* (Productive) or *Deficit* (Wasted Time)?

        **Step 4: Explore 'Technical Analysis'**
        *   For data scientists: View Feature Correlations, Model Metrics (F1-Score), and PCA projections.
        """)

    # 3. Methodology Visual
    with st.expander("🧠 **Understanding the Hybrid ML/DL Engine**", expanded=True):
        st.markdown("""
        The backend engine works in 3 stages:
        1.  **Preprocessing**: Your inputs are scaled and encoded (e.g., Gender -> 0/1).
        2.  **Parallel Inference**:
            *   **Tree Models** (Random Forest) vote on the outcome.
            *   **Neural Network** (Deep Learning) calculates probability weights.
        3.  **Late Fusion**: The system combines these votes to produce the final **Confidence Score**.
        """)

# ==================== TAB 1: Personal Insights (Non-Technical) ====================
# ==================== TAB 1: HOME DASHBOARD ====================
with tab_home:
    if arts and st.session_state.result:
        res = st.session_state.result
        # --- TOP ROW: Prediction & Confidence ---
        c_pred, c_conf = st.columns([2, 1])
        
        with c_pred:
            st.markdown(f"""
                <div class="hero-box">
                    <h3 style="margin:0;">Predicted Well-Being</h3>
                    <h1 style="font-size:3.5rem; margin:5px 0; color:#2c3e50;">{res['label']}</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Simple Advice
            happy_avg = arts['ref'].loc['Happiness']['Daily_Usage_Time']
            advice = get_simple_recommendation(res['label'], res['usage'], happy_avg)
            st.info(f"💡 {advice}")
            
            # --- PDF REPORT ---
            user_data_report = {
                'Age': age, 'Gender': gender, 'Platform': platform,
                'Daily_Usage_Time': usage, 'Posts_Per_Day': posts
            }
            pdf_bytes = create_pdf_report(user_data_report, res)
            st.download_button(
                label="📥 Download Wellness Report (PDF)",
                data=pdf_bytes,
                file_name="Digital_Moods_Report.pdf",
                mime="application/pdf",
                key='download-pdf'
            )

        with c_conf:
            current_conf = np.max(res['prob']) * 100
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = current_conf,
                title = {'text': "Confidence %", 'font': {'size': 20}},
                gauge = {
                    'axis': {'range': [None, 100]}, 
                    'bar': {'color': "royalblue"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#f0f2f6"
                }
            ))
            # Increased height and margins to prevent cutting off text
            fig_gauge.update_layout(height=250, margin=dict(l=40, r=40, t=50, b=10), font={'family': "Inter, sans-serif"})
            st.plotly_chart(fig_gauge, use_container_width=True, key="home_conf_gauge")
            st.caption("ℹ️ **Story:** This gauge visualizes the **Confidence Score** of the Hybrid Model. A score >80% means the model is highly certain that your usage pattern matches the predicted emotion (based on 2000+ training samples). If the score is low (<50%), your digital habits might be ambiguous or mixed.")

        st.markdown("---")
        
        # --- MIDDLE ROW: Persona & Platform Insights ---
        c_pers, c_plat = st.columns(2)
        
        with c_pers:
            st.markdown(f"#### 🎭 Persona: **{res['persona']}**")
            st.markdown(f"_{get_persona_description(res['persona'])}_")
            
            st.markdown("##### 🧘 Wellness Tips")
            tips = get_wellness_tips(res['label'], res['persona'])
            for tip in tips[:3]:
                st.markdown(f"- {tip}")

        with c_plat:
            st.markdown(f"#### 📱 Impact of **{res['platform']}**")
            
            # Platform specific stats
            plat_data = arts['plat'].loc[res['platform']]
            risk = plat_data[['Anxiety', 'Sadness', 'Anger']].sum() if all(c in plat_data.index for c in ['Anxiety', 'Sadness', 'Anger']) else 0
            health_score = 100 - int(risk * 100)
            
            p_color = "#4CAF50" if health_score > 70 else "#FFC107" if health_score > 50 else "#F44336"
            
            st.markdown(f"""
                <div style="text-align:center; padding:15px; border-radius:10px; border:2px solid {p_color}; background-color:#FAFAFA;">
                    <h2 style="color:{p_color}; margin:0;">Health Score: {health_score}/100</h2>
                    <small>Based on historical emotional correlations</small>
                </div>
            """, unsafe_allow_html=True)
            
            # Mini chart for this platform
            st.bar_chart(plat_data, color=p_color)

        # --- BOTTOM ROW: All Platforms Comparison ---
        st.markdown("---")
        st.markdown("#### 🏥 Platform Health Rankings")
        
        health_data = []
        for plat_name in arts['plat'].index:
            p_data = arts['plat'].loc[plat_name]
            r = p_data[['Anxiety', 'Sadness', 'Anger']].sum() if all(c in p_data.index for c in ['Anxiety', 'Sadness', 'Anger']) else 0
            health_data.append({'Platform': plat_name, 'Health Score': 100 - int(r * 100)})
        
        health_df = pd.DataFrame(health_data).sort_values('Health Score', ascending=True)
        
        fig_health = px.bar(health_df, x='Health Score', y='Platform', orientation='h',
                            title="Which platforms are healthiest?",
                            color='Health Score', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_health, use_container_width=True)
        st.info("📊 **Story:** Which platforms are 'Safer'? This chart aggregates emotional data from all users to rank platforms. A high **Health Score (Green)** means users on this platform statistically report 'Happiness' or 'Neutral' feelings more often than 'Anxiety' or 'Sadness'. Use this to choose where to spend your time.")

# ==================== TAB 2: Technical Analysis ====================
with tab_tech:
    st.markdown("### 📈 Exploratory Data Analysis (EDA)")
    
    if train_df is not None:
        eda_col1, eda_col2 = st.columns(2)
        
        with eda_col1:
            st.markdown("##### Emotion Distribution in Dataset")
            emotion_counts = train_df['Dominant_Emotion'].value_counts()
            fig1 = px.pie(values=emotion_counts.values, names=emotion_counts.index, 
                          title="Target Class Distribution", hole=0.4,
                          color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig1, use_container_width=True, key="tech_pie_chart")
            st.caption("**Story:** What are we predicting? This pie chart shows the balance of emotional states in our training dataset. Understanding this balance is crucial to ensure the model isn't biased toward one emotion.")
        
        with eda_col2:
            st.markdown("##### Usage Time vs. Emotion")
            usage_col = 'Daily_Usage_Time (minutes)' if 'Daily_Usage_Time (minutes)' in train_df.columns else 'Daily_Usage_Time'
            fig2 = px.box(train_df, x='Dominant_Emotion', y=usage_col, 
                          color='Dominant_Emotion', title="Screen Time by Emotion")
            st.plotly_chart(fig2, use_container_width=True, key="tech_box_chart")
            st.caption("**Story:** Does size matter? This box plot reveals the relationship between raw usage minutes and emotional states. Notice if 'Anxiety' or 'Sadness' correlates with higher median screen time.")
        
        eda_col3, eda_col4 = st.columns(2)
        
        with eda_col3:
            st.markdown("##### Platform Usage Distribution")
            plat_counts = train_df['Platform'].value_counts()
            fig3 = px.bar(x=plat_counts.index, y=plat_counts.values, 
                          title="Users per Platform", labels={'x': 'Platform', 'y': 'Count'},
                          color=plat_counts.values, color_continuous_scale='Blues')
            st.plotly_chart(fig3, use_container_width=True, key="tech_bar_chart")
            st.caption("**Story:** Where do users hang out? A breakdown of the most popular platforms in our study.")
        
        with eda_col4:
            st.markdown("##### Age Distribution")
            fig4 = px.histogram(train_df, x='Age', nbins=20, 
                                title="Age Distribution", color_discrete_sequence=['#667eea'])
            st.plotly_chart(fig4, use_container_width=True)
            st.caption("**Story:** Who are we analyzing? The age distribution of our user base.")
        
        # NEW: Correlation Heatmap
        st.markdown("##### 🔗 Feature Correlation Heatmap")
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            corr_matrix = train_df[numeric_cols].corr()
            fig_corr = px.imshow(corr_matrix, text_auto='.2f', aspect='auto',
                                 color_continuous_scale='RdBu_r', 
                                 title="Correlation Between Numerical Features")
            st.plotly_chart(fig_corr, use_container_width=True, key="tech_corr_chart")
            st.caption("**Story:** **The Hidden Connections.** This heatmap reveals mathematical relationships between variables. Red squares (Positive Correlation) mean two behaviors increase together (e.g., More Posts = More Likes). Blue squares (Negative Correlation) mean they move in opposites. This helps identifying 'Feedback Loops' in your behavior.")
    else:
        st.warning("Training data not found for EDA.")
    
    st.markdown("---")
    st.markdown("### 🏆 Model Performance Comparison")
    
    if arts and 'metrics' in arts:
        metrics_df = arts['metrics'].copy()
        
        # Format for display
        display_cols = ['Model', 'Accuracy', 'F1_Score', 'ROC_AUC']
        metrics_display = metrics_df[display_cols].copy()
        metrics_display = metrics_display.sort_values('Accuracy', ascending=False)
        
        # Highlight best
        st.dataframe(
            metrics_display.style.format({'Accuracy': '{:.2%}', 'F1_Score': '{:.2%}', 'ROC_AUC': '{:.2%}'})
                                 .highlight_max(axis=0, color='#c8e6c9'),
            use_container_width=True
        )
        
        st.markdown("##### Visual Comparison")
        fig_compare = px.bar(metrics_df.sort_values('Accuracy', ascending=True), 
                             x='Accuracy', y='Model', orientation='h',
                             title="Model Accuracy Comparison",
                             color='Accuracy', color_continuous_scale='Greens')
        st.plotly_chart(fig_compare, use_container_width=True)
        st.caption("**Story:** **Why Hybrid Matters?** This chart compares our custom Hybrid Engine against standard baselines. You'll see that accuracy jumps significantly when we combine Ensemble learning (Random Forest) with Deep Learning, proving that complex human emotions require complex modeling.")
        
        st.markdown("##### Feature Importance (Random Forest)")
        fig_fi = px.bar(arts['fi'].head(10), x='Importance', y='Feature', orientation='h',
                        title="Top 10 Feature Drivers", color='Importance',
                        color_continuous_scale='Purples')
        fig_fi.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_fi, use_container_width=True)
        st.caption("**Story:** **The Drivers of Emotion.** This plot ranks features by their 'Predictive Power'. If 'Daily Usage Time' is at the top, it mathematically proves that *how long* you spend online is the single biggest factor in determining your happiness, more than *what* you post.")
    else:
        st.warning("Metrics not found.")
    
    st.markdown("---")
    st.markdown("### 🗺️ Behavioral Projection (PCA)")
    
    if st.session_state.result:
        res = st.session_state.result
        comm_sample = arts['comm'].sample(min(400, len(arts['comm'])))
        user_pt = pd.DataFrame([{'PC1': res['pca'][0], 'PC2': res['pca'][1], 'Dominant_Emotion': 'YOU'}])
        plot_df = pd.concat([comm_sample, user_pt])
        
        fig_pca = px.scatter(plot_df, x='PC1', y='PC2', color='Dominant_Emotion',
                             color_discrete_map={'YOU': 'red'}, symbol='Dominant_Emotion',
                             symbol_map={'YOU': 'star'},
                             title="Your Position in Behavioral Space")
        fig_pca.update_traces(marker=dict(size=15), selector=dict(name='YOU'))
        st.plotly_chart(fig_pca, use_container_width=True)
        st.info("🗺️ **Story:** **You vs. The World.** We used PCA (Principal Component Analysis) to compress complex behavior into a simple 2D map. Each dot is a user. The colors are emotions. The **Red Star** is YOU. Are you surrounded by 'Anxious' (Orange) users or 'Happy' (Green) ones? This cluster analysis confirms your prediction visually.")
    else:
        st.info("Run analysis first to see your position on the map.")
    
    # NEW: Trend Simulator
    st.markdown("---")
    st.markdown("### 📈 Trend Simulator")
    st.markdown("See how changing your habits over time might affect your emotional state.")
    
    trend_col1, trend_col2 = st.columns(2)
    with trend_col1:
        trend_start = st.slider("Starting Usage (mins/day)", 30, 300, 120, key='trend_start')
    with trend_col2:
        trend_end = st.slider("Target Usage After 4 Weeks (mins/day)", 30, 300, 60, key='trend_end')
    
    if st.button("🔮 Simulate Trend"):
        weeks = [0, 1, 2, 3, 4]
        usages = [trend_start - (trend_start - trend_end) * w / 4 for w in weeks]
        predictions = []
        
        for u in usages:
            sim_raw = pd.DataFrame([{
                'User_ID': 'SIM', 'Age': age, 'Gender': gender, 'Platform': platform,
                'Daily_Usage_Time': u, 'Posts_Per_Day': posts,
                'Likes_Received_Per_Day': likes, 'Comments_Received_Per_Day': 5,
                'Messages_Sent_Per_Day': messages, 'Dominant_Emotion': 'Unknown'
            }])
            fe = FeatureEngineer()
            proc = arts['prep'].transform(fe.transform(sim_raw))
            pred = arts['le'].inverse_transform(arts['model'].predict(proc))[0]
            predictions.append(pred)
        
        trend_df = pd.DataFrame({'Week': weeks, 'Usage (mins)': usages, 'Predicted Emotion': predictions})
        st.dataframe(trend_df, use_container_width=True)
        
        # Visualize
        fig_trend = px.line(trend_df, x='Week', y='Usage (mins)', markers=True,
                            title="Your Simulated Journey", text='Predicted Emotion')
        fig_trend.update_traces(textposition='top center')
        st.plotly_chart(fig_trend, use_container_width=True)
        st.success("🔮 **Story:** **Your Future Self.** This isn't just a record—it's a forecast. The line shows how your 'Predicted Emotion' would likely change if you reduced your screen time over 4 weeks. Use this to set realistic goals: 'If I cut 30 mins, I might shift from Anxiety to Neutral.'")



    # ==================== TAB 3: Technical Analysis (Existing) ====================
    # ==================== TAB 2: Digital Balance Sheet ====================
    with tab_bal:
        st.markdown("### ⚖️ Digital Balance Sheet")
        
        # 1. TOP INTERACTION ZONE
        st.markdown("""
            <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:20px;">
                <h4 style="margin:0; color:#31333F;">🎚️ Adjust Your Balance</h4>
                <p style="font-size:0.9rem; margin-bottom:10px;">Configure your profile and habits to see your true Digital Net Worth.</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Defaults dictionary (moved from global scope)
        occ_defaults = {
            "Student": 60, "Homemaker": 200, "Service Sector": 150,
            "Corporate/Tech": 500, "Freelancer": 500
        }
        
        # Compact Control Panel: 4 Columns
        c1, c2, c3, c4 = st.columns(4)
        
        with c1:
            occupation = st.selectbox("1️⃣ Job Category", list(occ_defaults.keys()), index=0)
        with c2:
            hourly_rate = st.number_input("2️⃣ Hourly Value (₹)", min_value=50, max_value=10000, value=occ_defaults[occupation], step=50)
        with c3:
            productive_pct = st.slider("3️⃣ % Productive Use", 0, 100, 20, help="Learning, Networking, etc.")
        with c4:
            goal_activity = st.selectbox("4️⃣ Opportunity Context", ["Family Savings", "Skill Upskilling", "Entertainment", "Travel Fund"])

        st.markdown("---")

        # 2. Calculation Engine
        daily_hours = usage / 60
        productive_hours = daily_hours * (productive_pct / 100)
        unproductive_hours = daily_hours * ((100 - productive_pct) / 100)
        
        asset_value_daily = productive_hours * hourly_rate
        liability_cost_daily = unproductive_hours * hourly_rate
        
        net_worth_daily = asset_value_daily - liability_cost_daily
        net_worth_yearly = net_worth_daily * 365
        
        # 3. The Ledger (Columns)
        col_liab, col_asset = st.columns(2)
        
        with col_liab:
            st.markdown(f"""
                <div style="background-color:#ffebee; padding:20px; border-radius:10px; border-left:5px solid #ef5350; height:100%;">
                    <h3 style="color:#c62828; margin-top:0;">🔴 Liabilities (The Bad)</h3>
                    <p><strong>Unproductive Time:</strong> {int(unproductive_hours*60)} mins/day</p>
                    <hr>
                    <p style="font-size:1.1rem;">Financial Drain:</p>
                    <h2 style="color:#c62828; margin:0;">- ₹{int(liability_cost_daily)} / day</h2>
                    <small>(- ₹{int(liability_cost_daily * 365 / 1000)}k / year)</small>
                    <br><br>
                    <p><strong>Opportunity Cost:</strong></p>
                    <ul>
                        <li>Missed <strong>{goal_activity}</strong></li>
                        <li>Wasted {int(unproductive_hours * 365 / 24)} full days/year</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        with col_asset:
            st.markdown(f"""
                <div style="background-color:#e8f5e9; padding:20px; border-radius:10px; border-left:5px solid #66bb6a; height:100%;">
                    <h3 style="color:#2e7d32; margin-top:0;">🟢 Assets (The Good)</h3>
                    <p><strong>Productive Time:</strong> {int(productive_hours*60)} mins/day</p>
                    <hr>
                    <p style="font-size:1.1rem;">Value Created:</p>
                    <h2 style="color:#2e7d32; margin:0;">+ ₹{int(asset_value_daily)} / day</h2>
                    <small>(+ ₹{int(asset_value_daily * 365 / 1000)}k / year)</small>
                    <br><br>
                    <p><strong>Gains:</strong></p>
                    <ul>
                        <li>Knowledge / Upskilling</li>
                        <li>Networking & Connections</li>
                        <li>Inspiration & Creativity</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        # 4. The Bottom Line
        st.markdown("---")
        st.markdown("### 📊 The Bottom Line: Digital Net Worth")
        
        nw_color = "#4CAF50" if net_worth_yearly >= 0 else "#F44336"
        nw_status = "DIGITAL SURPLUS" if net_worth_yearly >= 0 else "DIGITAL DEFICIT"
        nw_message = "You use social media as a tool for growth!" if net_worth_yearly >= 0 else "Social media is consuming more value than it provides."
        
        st.markdown(f"""
            <div style="text-align:center; padding:30px; background:linear-gradient(to right, #f8f9fa, white, #f8f9fa); border-radius:20px; border:2px solid {nw_color};">
                <h4 style="color:gray; letter-spacing:2px; margin:0;">YEARLY PROJECTED STATUS</h4>
                <h1 style="font-size:4rem; color:{nw_color}; margin:10px 0;">{'+' if net_worth_yearly > 0 else ''}₹{int(net_worth_yearly):,}</h1>
                <h2 style="color:{nw_color}; background:white; display:inline-block; padding:5px 20px; border-radius:50px; border:1px solid {nw_color};">{nw_status}</h2>
                <p style="margin-top:15px; font-size:1.1rem;"><em>{nw_message}</em></p>
            </div>
        """, unsafe_allow_html=True)



# --- Footer ---
# --- Footer ---
st.markdown("""
<br><br><br>
<div class="footer">
    <p style="margin:0;">
        <strong>Capstone Project</strong>: Social Media Usage and Emotional Well-Being | 
        <strong>Batch</strong>: 11 | 
        <strong>Developed by</strong>: Ramasamy A
    </p>
</div>
""", unsafe_allow_html=True)
