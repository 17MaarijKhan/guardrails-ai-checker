import streamlit as st
import sqlparse
from guardrails import Guard
from guardrails.hub import ProfanityFree
from transformers import pipeline
from datetime import datetime

# Page settings
st.set_page_config(
    page_title="Guardrails AI - Safety Checker",
    page_icon="🛡️",
    layout="wide"
)

# ---------------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------------
st.markdown("""
<style>
    .stApp { background-color: #f0f2f5; }

    [data-testid="stSidebar"] {
        background-color: #1a1a2e !important;
        border-right: 3px solid #0078d4;
    }
    [data-testid="stSidebar"] * { color: white !important; }

    /* Sidebar nav buttons look like menu links, not big blue buttons */
    [data-testid="stSidebar"] .stButton > button {
        background: transparent !important;
        border: none !important;
        text-align: left !important;
        font-weight: 600 !important;
        padding: 8px 4px !important;
        box-shadow: none !important;
        width: 100% !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(0,120,212,0.25) !important;
        border-radius: 6px !important;
        transform: none !important;
        box-shadow: none !important;
    }

    .header-bar {
        background: linear-gradient(90deg, #0078d4, #005a9e);
        padding: 15px 25px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: white;
    }

    .stat-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-top: 4px solid #0078d4;
    }
    .stat-card h2 { color: #0078d4; font-size: 2.5rem; margin: 0; font-weight: 800; }
    .stat-card p { color: #666; margin: 5px 0 0 0; font-size: 0.9rem; font-weight: 600;
                   text-transform: uppercase; letter-spacing: 1px; }

    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border-left: 4px solid #0078d4;
    }
    .feature-card h3 { color: #1a1a2e; margin: 10px 0 5px 0; font-size: 1rem; font-weight: 700; }
    .feature-card p { color: #888; font-size: 0.85rem; margin: 0; }

    .stTextInput > div > div > input {
        background: white !important;
        border: 2px solid #0078d4 !important;
        border-radius: 8px !important;
        color: #1a1a2e !important;
        font-size: 1rem !important;
        padding: 12px !important;
    }

    /* Main-area buttons (not sidebar) keep the blue gradient style */
    .main .stButton > button {
        background: linear-gradient(90deg, #0078d4, #005a9e) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        padding: 12px !important;
        letter-spacing: 1px !important;
    }
    .main .stButton > button:hover {
        background: linear-gradient(90deg, #005a9e, #003f6e) !important;
        box-shadow: 0 5px 15px rgba(0,120,212,0.4) !important;
    }

    .result-safe {
        background: white; border-left: 6px solid #00b050; border-radius: 10px;
        padding: 20px 25px; box-shadow: 0 2px 10px rgba(0,176,80,0.15);
        color: #00b050; font-size: 1.2rem; font-weight: 700;
    }
    .result-blocked {
        background: white; border-left: 6px solid #e53935; border-radius: 10px;
        padding: 20px 25px; box-shadow: 0 2px 10px rgba(229,57,53,0.15);
        color: #e53935; font-size: 1.2rem; font-weight: 700;
    }

    .history-safe {
        background: white; border-left: 4px solid #00b050; border-radius: 8px;
        padding: 12px 15px; margin: 8px 0; box-shadow: 0 1px 5px rgba(0,0,0,0.05); color: #1a1a2e;
    }
    .history-blocked {
        background: white; border-left: 4px solid #e53935; border-radius: 8px;
        padding: 12px 15px; margin: 8px 0; box-shadow: 0 1px 5px rgba(0,0,0,0.05); color: #1a1a2e;
    }

    .section-title {
        color: #1a1a2e; font-size: 1.1rem; font-weight: 700; text-transform: uppercase;
        letter-spacing: 1px; margin: 20px 0 10px 0; padding-bottom: 8px; border-bottom: 2px solid #0078d4;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------
# ML MODELS
# ---------------------------------------------------------------
@st.cache_resource
def load_models():
    return pipeline("text-classification", model="unitary/toxic-bert", top_k=None)

profanity_guard = Guard().use(ProfanityFree(on_fail="exception"))

def is_sql_injection(text):
    try:
        parsed = sqlparse.parse(text)
        for statement in parsed:
            stmt_type = statement.get_type()
            if stmt_type in ("SELECT", "INSERT", "UPDATE", "DELETE",
                              "DROP", "CREATE", "ALTER", "TRUNCATE"):
                return True, stmt_type
        return False, None
    except Exception:
        return False, None

def check_input(user_input, toxic_classifier, threshold):
    try:
        profanity_guard.validate(user_input)
    except Exception:
        return False, "profanity", "🤬 Profanity detected!"

    is_sql, sql_type = is_sql_injection(user_input)
    if is_sql:
        return False, "sql", f"💉 SQL Injection — '{sql_type}' detected!"

    results = toxic_classifier(user_input)[0]
    for result in results:
        if result['score'] > threshold:
            labels = {
                'toxic': "☠️ Toxic language detected!",
                'threat': "⚠️ Threat detected!",
                'insult': "🤬 Insult detected!",
                'obscene': "🔞 Obscene content!",
                'identity_hate': "😡 Hate speech detected!",
                'severe_toxic': "🚫 Severely toxic!"
            }
            if result['label'] in labels:
                return False, result['label'], labels[result['label']]

    return True, "safe", "✅ Input is completely safe!"

# ---------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []
if "safe_count" not in st.session_state:
    st.session_state.safe_count = 0
if "blocked_count" not in st.session_state:
    st.session_state.blocked_count = 0
if "page" not in st.session_state:
    st.session_state.page = "dashboard"
if "threshold" not in st.session_state:
    st.session_state.threshold = 0.8

# ---------------------------------------------------------------
# SIDEBAR (NOW FUNCTIONAL NAVIGATION)
# ---------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ Guardrails AI")
    st.markdown("---")
    st.markdown("### MENU")

    if st.button("📊  Dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
    if st.button("🔍  Input Checker", use_container_width=True):
        st.session_state.page = "checker"
    if st.button("📋  History", use_container_width=True):
        st.session_state.page = "history"
    if st.button("⚙️  Settings", use_container_width=True):
        st.session_state.page = "settings"

    st.markdown("---")
    st.markdown("### PROTECTION STATUS")
    st.markdown("🟢 ML Profanity — **Active**")
    st.markdown("🟢 SQL Detector — **Active**")
    st.markdown("🟢 Toxic Detector — **Active**")
    st.markdown("---")
    st.markdown(f"🕐 {datetime.now().strftime('%m/%d/%Y  %H:%M:%S')}")

# ---------------------------------------------------------------
# TOP HEADER (shown on every page)
# ---------------------------------------------------------------
page_titles = {
    "dashboard": "Dashboard",
    "checker": "Input Checker",
    "history": "History",
    "settings": "Settings"
}
st.markdown(f"""
<div class="header-bar">
    <div><b style="font-size:1.4rem">🛡️ GUARDRAILS AI</b> — {page_titles[st.session_state.page]}</div>
    <div>admin &nbsp;|&nbsp; {datetime.now().strftime("%m/%d/%Y %H:%M")}</div>
</div>
""", unsafe_allow_html=True)

with st.spinner("🤖 Loading AI Models..."):
    toxic_classifier = load_models()

# ---------------------------------------------------------------
# PAGE: DASHBOARD
# ---------------------------------------------------------------
if st.session_state.page == "dashboard":
    st.markdown('<p class="section-title">📊 Dashboard Stats</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="stat-card"><h2>{st.session_state.safe_count}</h2><p>Safe Inputs</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="stat-card"><h2 style="color:#e53935">{st.session_state.blocked_count}</h2><p>Blocked Inputs</p></div>', unsafe_allow_html=True)
    with col3:
        total = st.session_state.safe_count + st.session_state.blocked_count
        st.markdown(f'<div class="stat-card"><h2 style="color:#7b2ff7">{total}</h2><p>Total Checked</p></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔰 Protection Layers</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="feature-card"><div style="font-size:2rem">🤖</div><h3>ML Profanity Detector</h3><p>Detects bad & offensive language automatically</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="feature-card"><div style="font-size:2rem">💉</div><h3>SQL Injection Detector</h3><p>Automatically detects SQL attacks</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="feature-card"><div style="font-size:2rem">☠️</div><h3>Toxic & Threat Detector</h3><p>ML-based toxic content detection</p></div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">📋 Recent Activity</p>', unsafe_allow_html=True)
        for item in reversed(st.session_state.history[-5:]):
            cls = "history-safe" if item["status"] == "safe" else "history-blocked"
            badge = "✅ SAFE" if item["status"] == "safe" else "❌ BLOCKED"
            st.markdown(f'<div class="{cls}">{badge} &nbsp;|&nbsp; <code>{item["input"]}</code> &nbsp;|&nbsp; {item["reason"]} &nbsp;|&nbsp; 🕐 {item["time"]}</div>', unsafe_allow_html=True)
    else:
        st.info("No checks yet — go to **Input Checker** in the sidebar to test something!")

# ---------------------------------------------------------------
# PAGE: INPUT CHECKER
# ---------------------------------------------------------------
elif st.session_state.page == "checker":
    st.markdown('<p class="section-title">🔍 Input Checker</p>', unsafe_allow_html=True)
    user_input = st.text_input("", placeholder="Type your input here to check safety...")

    if st.button("🔍 CHECK INPUT", use_container_width=True):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter something!")
        else:
            with st.spinner("🔍 Analyzing with AI..."):
                is_safe, reason, message = check_input(user_input, toxic_classifier, st.session_state.threshold)

            if is_safe:
                st.markdown(f'<div class="result-safe">✅ SAFE &nbsp;|&nbsp; {message}</div>', unsafe_allow_html=True)
                st.balloons()
                st.session_state.safe_count += 1
            else:
                st.markdown(f'<div class="result-blocked">❌ BLOCKED &nbsp;|&nbsp; {message}</div>', unsafe_allow_html=True)
                st.session_state.blocked_count += 1

            st.session_state.history.append({
                "input": user_input,
                "status": "safe" if is_safe else "blocked",
                "reason": message,
                "time": datetime.now().strftime("%H:%M:%S")
            })

# ---------------------------------------------------------------
# PAGE: HISTORY
# ---------------------------------------------------------------
elif st.session_state.page == "history":
    st.markdown('<p class="section-title">📋 Full Check History</p>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.info("No history yet — go to **Input Checker** to test something first!")
    else:
        for item in reversed(st.session_state.history):
            cls = "history-safe" if item["status"] == "safe" else "history-blocked"
            badge = "✅ SAFE" if item["status"] == "safe" else "❌ BLOCKED"
            st.markdown(f'<div class="{cls}">{badge} &nbsp;|&nbsp; <code>{item["input"]}</code> &nbsp;|&nbsp; {item["reason"]} &nbsp;|&nbsp; 🕐 {item["time"]}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------
# PAGE: SETTINGS
# ---------------------------------------------------------------
elif st.session_state.page == "settings":
    st.markdown('<p class="section-title">⚙️ Settings</p>', unsafe_allow_html=True)

    st.markdown("#### Toxic Detection Sensitivity")
    st.session_state.threshold = st.slider(
        "Lower value = stricter (blocks more), Higher value = more lenient",
        min_value=0.3, max_value=0.95, value=st.session_state.threshold, step=0.05
    )
    st.caption(f"Current threshold: {st.session_state.threshold}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Reset Data")
    if st.button("🗑️ Clear All History & Stats", use_container_width=True):
        st.session_state.history = []
        st.session_state.safe_count = 0
        st.session_state.blocked_count = 0
        st.success("History and stats cleared!")