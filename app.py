# app.py (updated with 5th app)
import streamlit as st
from auth import login_user, signup_user, logout_user, check_session
from database import init_database

# Initialize database
init_database()

# Page configuration
st.set_page_config(
    page_title="iRMC aura - AI Apps Hub",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        font-weight: 800;
        margin: 2rem 0 3rem 0;
        padding: 0;
    }
    
    .app-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 4px solid #175CFF;
        box-shadow: 0 4px 12px rgba(23, 92, 255, 0.1);
        margin-bottom: 1rem;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #175CFF, #00A3FF);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(23, 92, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background-color: #F8FAFF;
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #175CFF;
        color: white;
    }
    
    div[data-testid="stForm"] {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(23, 92, 255, 0.15);
        border: 1px solid #E6F0FF;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

def login_page():
    st.markdown('<div class="main-header">iRMC aura</div>', unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["**Login**", "**Sign Up**"])
    
    with tab1:
        st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Welcome Back</h3>', unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("**Email Address**", placeholder="Enter your email")
            password = st.text_input("**Password**", type="password", placeholder="Enter your password")
            login_btn = st.form_submit_button("**Login**")
            
            if login_btn:
                if email and password:
                    if login_user(email, password):
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid email or password")
                else:
                    st.warning("⚠️ Please fill all fields")
    
    with tab2:
        st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Create Account</h3>', unsafe_allow_html=True)
        
        with st.form("signup_form"):
            new_email = st.text_input("**Email Address**", placeholder="Enter your email")
            new_password = st.text_input("**Password**", type="password", placeholder="Create a password (min 4 chars)")
            confirm_password = st.text_input("**Confirm Password**", type="password", placeholder="Re-enter your password")
            signup_btn = st.form_submit_button("**Create Account**")
            
            if signup_btn:
                if new_email and new_password and confirm_password:
                    if new_password != confirm_password:
                        st.error("❌ Passwords don't match!")
                    elif len(new_password) < 4:
                        st.error("❌ Password must be at least 4 characters")
                    else:
                        if signup_user(new_email, new_password):
                            st.success(" Account created successfully! Please login.")
                        else:
                            st.error("❌ Email already exists!")
                else:
                    st.warning("⚠️ Please fill all fields")

def home_page():
    # Homepage header
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("""
        <div style='margin-bottom: 1rem;'>
            <div class="main-header" style='font-size: 2.5rem; text-align: left; margin: 0;'> iRMC aura</div>
            <p style='color: #666; font-size: 1.1rem; margin: 0;'>
                Welcome back, <strong style="color: #175CFF;">{}</strong>
            </p>
        </div>
        """.format(st.session_state.email), unsafe_allow_html=True)
    
    with col2:
        if st.button("**Logout**", use_container_width=True):
            logout_user()
            st.rerun()
    
    st.markdown("---")
    
    # App boxes
    st.markdown('<h3 style="color: #175CFF; text-align: center; margin-bottom: 2rem;">Aura AI Applications</h3>', unsafe_allow_html=True)
    
    # Create 3 rows for apps (2 apps in each row)
    # Row 1
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>Chatbot</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Chat with your documents using advanced AI technology.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Chatbot", key="rag_btn", use_container_width=True):
            st.switch_page("pages/2_Chatbot.py")
    
    with col2:
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'> GraphX</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Advanced document analysis with knowledge graphs.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch GraphX", key="graph_btn", use_container_width=True):
            st.switch_page("pages/3_🌐_Graph_RAG.py")
    
    # Row 2
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>SQL_Assistant</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Convert English to SQL queries instantly.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch SQL_Assistant", key="sql_btn", use_container_width=True):
            st.switch_page("pages/4_📊_SQL_Assistant.py")
    
    with col2:
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>QueryCraft</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Query and analyze CSV files with natural language.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch QueryCraft", key="csv_btn", use_container_width=True):
            st.switch_page("pages/5_📁_CSV_SQL_Assistant.py")
    
    # Row 3 - NEW APP
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>DataGiene</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>Generate Data</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch DataGiene", key="research_btn", use_container_width=True):
            st.switch_page("pages/6_🔢_Synthetic_Data_Generator.py")
    
    with col2:
        st.markdown("""
        <div class="app-box">
            <div>
                <h4 style='color: #175CFF; margin: 0 0 0.5rem 0;'>🚀 Coming Soon</h4>
                <p style='color: #555; margin: 0; font-size: 0.9rem;'>More AI tools and applications in development.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Explore More", key="more_btn", use_container_width=True, disabled=True):
            pass

def main():
    # Check if user is logged in
    if not check_session():
        login_page()
    else:
        home_page()

if __name__ == "__main__":
    main()
