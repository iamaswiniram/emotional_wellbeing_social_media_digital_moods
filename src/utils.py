import streamlit as st
import datetime
from fpdf import FPDF
import io

# --- CUSTOM CSS ---
CUSTOM_CSS = """
<style>
    /* 1. GLOBAL FONTS & THEME */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp { background: linear-gradient(to bottom, #f0f4f8, #e8f0f7); }
    
    /* 2. HEADER STYLING */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        font-size: 2.5rem;
    }
    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* 3. SYNOPSIS BOX */
    .synopsis-box {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #764ba2;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 25px;
        font-size: 0.95rem;
        color: #2c3e50;
    }

    /* 4. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e9ecef;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 2rem;
    }

    /* 5. CARD & METRIC STYLING */
    .hero-box {
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        text-align: center;
        transition: transform 0.2s;
    }
    .hero-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
    }
    
    .action-box {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white; padding: 20px; border-radius: 15px; font-size: 0.95rem;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white; padding: 20px; border-radius: 15px; font-size: 0.95rem;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
    [data-testid="stMetricLabel"] { font-size: 0.75rem !important; }

    /* 6. TABS STYLING */
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 10px 10px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        border: 1px solid #e9ecef;
        border-bottom: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #fff;
        border-bottom: 2px solid #667eea;
        color: #667eea !important;
        font-weight: 600;
    }

    /* 7. FOOTER STYLING */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: white;
        color: #6c757d;
        text-align: center;
        padding: 15px;
        border-top: 1px solid #e9ecef;
        font-size: 0.85rem;
        z-index: 999;
    }
    
    /* Hide default streamlit footer/menu for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
</style>
"""

# --- HELPER FUNCTIONS ---

def get_simple_recommendation(label, usage, happy_avg=120):
    if label == 'Happiness':
        return "🎉 **Great News!** Your habits are associated with positive emotions. Keep it up!"
    elif label in ['Anxiety', 'Sadness']:
        diff = usage - happy_avg
        if diff > 30:
            return f"💡 **Suggestion:** You use social media **{int(diff)} mins more** than happy users. Try reducing by 20-30 mins."
        return "💡 **Suggestion:** Focus on meaningful interactions over passive scrolling."
    return "💡 **Tip:** Balance your online and offline activities."

def get_persona_description(persona):
    return {
        "Balanced User": "Moderate usage, similar to users with positive emotions.",
        "Passive Scroller": "You consume more than create. Try engaging more actively.",
        "Power User": "Highly active. Remember to take breaks."
    }.get(persona, "Unique usage pattern.")

def get_wellness_tips(emotion, persona):
    tips = []
    
    # Emotion-based tips
    if emotion in ['Anxiety', 'Sadness']:
        tips.extend([
            "🧘 **Mindfulness Break**: Take 5 mins to breathe deeply before checking your phone.",
            "🌿 **Nature Walk**: Try replacing 15 mins of scrolling with a short walk outside.",
            "📵 **No-Phone Zone**: Keep devices out of the bedroom for better sleep."
        ])
    elif emotion == 'Anger':
        tips.extend([
            "⏸️ **Pause Before Posting**: Wait 10 mins before responding to triggering content.",
            "🔇 **Mute Triggers**: Use mute/block features on accounts that upset you.",
            "🎵 **Mood Music**: Listen to calming music instead of doom-scrolling."
        ])
    else:  # Happiness/Neutral
        tips.extend([
            "🌟 **Share Positivity**: Your good vibes can help others! Share helpful content.",
            "📅 **Schedule Socials**: Keep a consistent but limited social media schedule.",
            "🤝 **Real Connections**: Convert online friendships to real-life meetups."
        ])
    
    # Persona-based tips
    if persona == "Passive Scroller":
        tips.append("✍️ **Create More**: Try posting 1 thing per day instead of just scrolling.")
    elif persona == "Power User":
        tips.append("⏰ **Set Timers**: Use app timers to limit sessions to 20-minute blocks.")
    
    return tips[:4]  # Return top 4 tips

def get_detox_challenges():
    return [
        {"name": "📵 Phone-Free Mornings", "desc": "No social media for the first hour after waking up.", "days": 7, "difficulty": "Easy"},
        {"name": "🌅 Sunset Scrolling", "desc": "No scrolling after sunset. Read a book instead!", "days": 5, "difficulty": "Medium"},
        {"name": "🗓️ Social Media Sabbath", "desc": "One full day per week without any social media.", "days": 4, "difficulty": "Hard"},
        {"name": "🎯 Purpose Only", "desc": "Only open apps with a specific purpose in mind.", "days": 7, "difficulty": "Medium"}
    ]

def get_achievements(usage, posts, messages, emotion):
    badges = []
    if usage <= 60:
        badges.append({"icon": "⏱️", "name": "Digital Minimalist", "desc": "Under 1 hour daily usage!"})
    if usage <= 120:
        badges.append({"icon": "⚖️", "name": "Balanced Soul", "desc": "Healthy 2-hour limit maintained."})
    if messages > posts * 3:
        badges.append({"icon": "💬", "name": "Connector", "desc": "Prioritizing real conversations."})
    if emotion == 'Happiness':
        badges.append({"icon": "😊", "name": "Joy Keeper", "desc": "Your habits align with happiness!"})
    if posts >= 1 and posts <= 3:
        badges.append({"icon": "🎨", "name": "Mindful Creator", "desc": "Quality over quantity in posting."})
    return badges

# --- PDF GENERATOR ---

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Digital Moods | Wellness Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def clean_text(text):
    """Sanitize text for standard FPDF (Latin-1 only). Removes emojis/special chars."""
    if isinstance(text, str):
        return text.encode('latin-1', 'ignore').decode('latin-1')
    return str(text)

def create_pdf_report(user_data, results):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title Section
    pdf.set_font("Arial", "B", 16)
    title_text = clean_text(f"Wellness Report for {user_data.get('Gender', 'User')}")
    pdf.cell(200, 10, txt=title_text, ln=True, align='L')
    
    pdf.set_font("Arial", size=10)
    date_text = clean_text(f"Generated on {datetime.date.today()}")
    pdf.cell(200, 10, txt=date_text, ln=True, align='L')
    pdf.ln(10)
    
    # 1. Prediction Summary
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=clean_text("1. Emotional Analysis"), ln=True)
    pdf.set_font("Arial", size=12)
    
    pdf.cell(200, 10, txt=clean_text(f"Predicted Emotion: {results['label']}"), ln=True)
    pdf.cell(200, 10, txt=clean_text(f"Confidence Score: {results['prob'].max()*100:.1f}%"), ln=True)
    pdf.cell(200, 10, txt=clean_text(f"Digital Persona: {results['persona']}"), ln=True)
    pdf.ln(5)
    
    # 2. Usage Stats
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=clean_text("2. Usage Statistics"), ln=True)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=clean_text(f"Platform: {user_data['Platform']}"), ln=True)
    pdf.cell(200, 10, txt=clean_text(f"Daily Screen Time: {user_data['Daily_Usage_Time']} mins"), ln=True)
    pdf.cell(200, 10, txt=clean_text(f"Posts per Day: {user_data['Posts_Per_Day']}"), ln=True)
    pdf.ln(5)
    
    # 3. Recommendations
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt=clean_text("3. Tailored Recommendations"), ln=True)
    pdf.set_font("Arial", size=12)
    
    tips = get_wellness_tips(results['label'], results['persona'])
    for i, tip in enumerate(tips, 1):
        # Strip markdown bolding and sanitize for PDF
        clean_tip_text = tip.replace("**", "").replace("*", "")
        # Remove emojis by encoding/decoding
        final_tip = clean_text(f"{i}. {clean_tip_text}")
        pdf.multi_cell(0, 10, txt=final_tip)
    
    return pdf.output(dest='S').encode('latin-1')
