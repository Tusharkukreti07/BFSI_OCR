import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
import cv2
import pytesseract
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import json
import requests
from io import StringIO
import re

# Set page configuration
st.set_page_config(
    page_title="ApnaBank - Your Financial Partner",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:support@apnabank.com',
        'Report a bug': 'mailto:bugs@apnabank.com',
        'About': 'ApnaBank - Your Financial Partner'
    }
)

# Load the pre-trained model
@st.cache_resource
def load_model():
    try:
        with open('cibil_model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'cibil_model.pkl' is in the current directory.")
        return None

# Database setup
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String)
    email = Column(String, unique=True)
    age = Column(Integer)
    income = Column(Float)
    employment_type = Column(String)
    college = Column(String)
    college_rank = Column(Integer)
    existing_loans = Column(Integer, default=0)
    credit_card_usage = Column(Float, default=0)
    repayment_history = Column(Float, default=0)
    cibil_score = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)
    
    transactions = relationship("Transaction", back_populates="user")

class Transaction(Base):
    __tablename__ = 'transactions'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'))
    month = Column(String)
    total_income = Column(Float)
    total_expense = Column(Float)
    loan_emi = Column(Float)
    credit_card_payment = Column(Float)
    savings = Column(Float)
    created_at = Column(DateTime, default=datetime.now)
    
    user = relationship("User", back_populates="transactions")

# Create database engine and session
engine = create_engine('sqlite:///apnabank.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# Load Animation JSON
@st.cache_data
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

@st.cache_data
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.error(f"Failed to load animation: {e}")
        return None

# Create improved fallback animations
def get_fallback_animation(animation_type="general"):
    # More detailed fallback animations for different contexts
    if animation_type == "banking":
        return {
            "v": "5.7.8",
            "fr": 30,
            "ip": 0,
            "op": 60,
            "w": 400,
            "h": 400,
            "nm": "Banking Animation",
            "ddd": 0,
            "assets": [],
            "layers": [
                {
                    "ddd": 0,
                    "ind": 1,
                    "ty": 4,
                    "nm": "Bank",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100},
                        "r": {"a": 0, "k": 0},
                        "p": {"a": 0, "k": [200, 200, 0]},
                        "a": {"a": 0, "k": [0, 0, 0]},
                        "s": {
                            "a": 1,
                            "k": [
                                {"t": 0, "s": [100, 100, 100], "h": 0},
                                {"t": 30, "s": [110, 110, 100], "h": 0},
                                {"t": 60, "s": [100, 100, 100], "h": 0}
                            ]
                        }
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "rc",
                            "d": 1,
                            "s": {"a": 0, "k": [100, 80]},
                            "p": {"a": 0, "k": [0, 10]},
                            "r": {"a": 0, "k": 5},
                            "nm": "Building",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0.102, 0.2, 0.396, 1]},
                            "o": {"a": 0, "k": 100},
                            "r": 1,
                            "bm": 0,
                            "nm": "Fill",
                            "hd": False
                        },
                        {
                            "ty": "tr",
                            "p": {"a": 0, "k": [0, -30]},
                            "a": {"a": 0, "k": [0, 0]},
                            "s": {"a": 0, "k": [100, 100]},
                            "r": {"a": 0, "k": 0},
                            "o": {"a": 0, "k": 100},
                            "sk": {"a": 0, "k": 0},
                            "sa": {"a": 0, "k": 0}
                        }
                    ]
                },
                {
                    "ddd": 0,
                    "ind": 2,
                    "ty": 4,
                    "nm": "Roof",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100},
                        "r": {"a": 0, "k": 0},
                        "p": {"a": 0, "k": [200, 150, 0]},
                        "a": {"a": 0, "k": [0, 0, 0]},
                        "s": {"a": 0, "k": [100, 100, 100]}
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "sh",
                            "d": 1,
                            "ks": {
                                "a": 0,
                                "k": {
                                    "i": [[0, 0], [0, 0], [0, 0]],
                                    "o": [[0, 0], [0, 0], [0, 0]],
                                    "v": [[-60, 0], [0, -40], [60, 0]],
                                    "c": True
                                }
                            },
                            "nm": "Triangle",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0.945, 0.768, 0.113, 1]},
                            "o": {"a": 0, "k": 100},
                            "r": 1,
                            "bm": 0,
                            "nm": "Fill",
                            "hd": False
                        }
                    ]
                },
                {
                    "ddd": 0,
                    "ind": 3,
                    "ty": 4,
                    "nm": "Coin",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100},
                        "r": {"a": 1, "k": [
                            {"t": 0, "s": [0], "h": 0},
                            {"t": 60, "s": [360], "h": 0}
                        ]},
                        "p": {"a": 1, 
                             "k": [
                                {"t": 0, "s": [150, 250, 0], "h": 0},
                                {"t": 30, "s": [250, 250, 0], "h": 0},
                                {"t": 60, "s": [150, 250, 0], "h": 0}
                             ]
                        },
                        "a": {"a": 0, "k": [0, 0, 0]},
                        "s": {"a": 0, "k": [100, 100, 100]}
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "el",
                            "d": 1,
                            "s": {"a": 0, "k": [40, 40]},
                            "p": {"a": 0, "k": [0, 0]},
                            "nm": "Circle",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0.945, 0.768, 0.113, 1]},
                            "o": {"a": 0, "k": 100},
                            "r": 1,
                            "bm": 0,
                            "nm": "Fill",
                            "hd": False
                        }
                    ]
                }
            ]
        }
    elif animation_type == "loan":
        # Simple loan animation fallback
        return {
            "v": "5.7.8",
            "fr": 30,
            "ip": 0,
            "op": 60,
            "w": 400,
            "h": 400,
            "nm": "Loan Animation",
            "ddd": 0,
            "assets": [],
            "layers": [
                {
                    "ddd": 0,
                    "ind": 1,
                    "ty": 4,
                    "nm": "Document",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100},
                        "r": {"a": 0, "k": 0},
                        "p": {"a": 0, "k": [200, 200, 0]},
                        "a": {"a": 0, "k": [0, 0, 0]},
                        "s": {"a": 1, "k": [
                            {"t": 0, "s": [100, 100, 100], "h": 0},
                            {"t": 30, "s": [110, 110, 100], "h": 0},
                            {"t": 60, "s": [100, 100, 100], "h": 0}
                        ]}
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "rc",
                            "d": 1,
                            "s": {"a": 0, "k": [80, 120]},
                            "p": {"a": 0, "k": [0, 0]},
                            "r": {"a": 0, "k": 5},
                            "nm": "Paper",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0.929, 0.929, 0.929, 1]},
                            "o": {"a": 0, "k": 100},
                            "r": 1,
                            "bm": 0,
                            "nm": "Fill",
                            "hd": False
                        }
                    ]
                },
                {
                    "ddd": 0,
                    "ind": 2,
                    "ty": 4,
                    "nm": "Line1",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100},
                        "r": {"a": 0, "k": 0},
                        "p": {"a": 0, "k": [200, 180, 0]},
                        "a": {"a": 0, "k": [0, 0, 0]},
                        "s": {"a": 0, "k": [100, 100, 100]}
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "rc",
                            "d": 1,
                            "s": {"a": 0, "k": [60, 8]},
                            "p": {"a": 0, "k": [0, 0]},
                            "r": {"a": 0, "k": 4},
                            "nm": "Line",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0.2, 0.2, 0.2, 1]},
                            "o": {"a": 0, "k": 100},
                            "r": 1,
                            "bm": 0,
                            "nm": "Fill",
                            "hd": False
                        }
                    ]
                }
            ]
        }
    else:
        # Default general animation
        return {
            "v": "5.7.8",
            "fr": 30,
            "ip": 0,
            "op": 60,
            "w": 400,
            "h": 400,
            "nm": "Fallback Animation",
            "ddd": 0,
            "assets": [],
            "layers": [
                {
                    "ddd": 0,
                    "ind": 1,
                    "ty": 4,
                    "nm": "Circle",
                    "sr": 1,
                    "ks": {
                        "o": {"a": 0, "k": 100},
                        "r": {"a": 0, "k": 0},
                        "p": {"a": 0, "k": [200, 200, 0]},
                        "a": {"a": 0, "k": [0, 0, 0]},
                        "s": {
                            "a": 1,
                            "k": [
                                {"t": 0, "s": [100, 100, 100], "h": 0},
                                {"t": 30, "s": [150, 150, 100], "h": 0},
                                {"t": 60, "s": [100, 100, 100], "h": 0}
                            ]
                        }
                    },
                    "ao": 0,
                    "shapes": [
                        {
                            "ty": "rc",
                            "d": 1,
                            "s": {"a": 0, "k": [100, 100]},
                            "p": {"a": 0, "k": [0, 0]},
                            "r": {"a": 0, "k": 50},
                            "nm": "Rectangle Path",
                            "hd": False
                        },
                        {
                            "ty": "fl",
                            "c": {"a": 0, "k": [0.102, 0.2, 0.396, 1]},
                            "o": {"a": 0, "k": 100},
                            "r": 1,
                            "bm": 0,
                            "nm": "Fill",
                            "hd": False
                        }
                    ]
                }
            ]
        }

# Load animation assets with improved fallbacks and updated URLs
fallback_animation = get_fallback_animation()
banking_fallback = get_fallback_animation("banking")
loan_fallback = get_fallback_animation("loan")

# Update animation URLs to more reliable sources
banking_animation = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_qgah66oj.json") or banking_fallback
loan_animation = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_rntlc8f4.json") or loan_fallback
analytics_animation = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_tno6cg2w.json") or fallback_animation
pay_animation = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_qgah66oj.json") or fallback_animation
stock_animation = load_lottieurl("https://assets8.lottiefiles.com/packages/lf20_4kx2q32n.json") or fallback_animation
video_animation = load_lottieurl("https://assets10.lottiefiles.com/temp/lf20_nXwOJj.json") or fallback_animation

# Add a helper function to render animations with proper error handling
def render_animation(animation_data, height=300, key=None):
    try:
        st_lottie(animation_data, height=height, key=key)
    except Exception as e:
        st.error(f"Error displaying animation: {e}")
        # Display a simple colored box as absolute fallback
        st.markdown(f"""
        <div style="
            height: {height}px; 
            background-color: #1A3365; 
            border-radius: 10px; 
            display: flex; 
            align-items: center; 
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 24px;">
            ApnaBank
        </div>
        """, unsafe_allow_html=True)

# CSS for animations and styling
st.markdown("""
<style>
    /* Override Streamlit's default theme */
    .stApp {
        background-color: white;
    }
    
    /* Sidebar background */
    .css-1d391kg, .css-1lcbmhc, section[data-testid="stSidebar"] {
        background-color: #f9f9f9 !important;
    }
    
    /* Text color adjustments */
    .stMarkdown, p, h1, h2, h3, h4, h5, h6, span, div, .stText, .stTitle {
        color: #333333 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #1A3365;
        color: white;
    }
    
    /* Adjust input fields */
    .stTextInput > div > div > input, .stNumberInput > div > div > input, .stDateInput > div > div > input, .stTimeInput > div > div > input {
        background-color: white;
        color: #333333;
    }
    
    /* Main styles */
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1A3365;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2C4C8C;
        margin: 15px 0;
    }
    
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);
    }
    
    .feature-icon {
        font-size: 2rem;
        color: #1A3365;
        margin-bottom: 10px;
    }
    
    .feature-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1A3365;
    }
    
    .feature-desc {
        color: #555;
        font-size: 0.9rem;
    }
    
    /* Navigation bar styling */
    .navbar {
        background-color: #1A3365;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        
    }
    
    .nav-item {
        color: black; /* Change font color to black */
        text-decoration: none;
        margin: 0 15px;
        font-weight: 500;
        transition: color 0.3s ease;
    }
    
    .nav-item:hover {
        color: #FFD700;
    }
    
    /* Login form styling */
    .login-form {
        max-width: 400px;
        margin: 0 auto;
        padding: 30px;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Dashboard styling */
    .dashboard-tile {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        height: 100%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1A3365;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    
    /* Animation class for scrolling effects */
    .scroll-animation {
        opacity: 0;
        transform: translateY(20px);
        transition: opacity 0.8s ease, transform 0.8s ease;
    }
    
    .scroll-animation.visible {
        opacity: 1;
        transform: translateY(0);
    }
    
    /* Staggered animation delays */
    .delay-1 { transition-delay: 0.1s; }
    .delay-2 { transition-delay: 0.2s; }
    .delay-3 { transition-delay: 0.3s; }
    .delay-4 { transition-delay: 0.4s; }
    .delay-5 { transition-delay: 0.5s; }
    
    /* Custom button styles */
    .custom-button {
        background-color: #1A3365;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: 500;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    
    .custom-button:hover {
        background-color: #2C4C8C;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1.4rem;
        }
    }
</style>

<script>
    // JavaScript for scroll animations
    document.addEventListener('DOMContentLoaded', function() {
        const animatedElements = document.querySelectorAll('.scroll-animation');
        
        function checkScroll() {
            animatedElements.forEach(element => {
                const elementTop = element.getBoundingClientRect().top;
                const windowHeight = window.innerHeight;
                
                if (elementTop < windowHeight - 100) {
                    element.classList.add('visible');
                }
            });
        }
        
        window.addEventListener('scroll', checkScroll);
        checkScroll(); // Check on load
    });
</script>
""", unsafe_allow_html=True)

# Session state initialization
if 'is_logged_in' not in st.session_state:
    st.session_state.is_logged_in = False
    
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
    
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Navigation bar
def render_navbar():
    # Title and logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/clouds/100/000000/bank-building.png", width=80)
    with col2:
        st.markdown('<h1 class="main-header">ApnaBank</h1>', unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Supervised Learning", "Unsupervised Learning", "Semi-Supervised Learning", "Loan Checker", "VideoText Extractor", "Dashboard"],
        icons=["house", "clipboard-data", "lightbulb", "graph-up", "calculator", "camera-video", "speedometer"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "#1A3365", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "#e6f0ff" ,},
            "nav-link-selected": {"background-color": "#1A3365"},
        }
    )
    
    if selected == "Home":
        st.session_state.page = 'home'
    elif selected == "Supervised Learning":
        st.session_state.page = 'supervised'
    elif selected == "Unsupervised Learning":
        st.session_state.page = 'unsupervised'
    elif selected == "Semi-Supervised Learning":
        st.session_state.page = 'semi_supervised'
    elif selected == "Loan Checker":
        st.session_state.page = 'loan_checker'
    elif selected == "VideoText Extractor":
        st.session_state.page = 'video_text'
    elif selected == "Dashboard":
        if st.session_state.is_logged_in:
            st.session_state.page = 'dashboard'
        else:
            st.warning("Please login to access the dashboard")
            st.session_state.page = 'login'
    
    # Login/Logout button
    if st.session_state.is_logged_in:
        if st.button("Logout"):
            st.session_state.is_logged_in = False
            st.session_state.current_user = None
            st.session_state.page = 'home'
            st.experimental_rerun()
    else:
        col1, col2 = st.columns([9, 1])
        with col2:
            if st.button("Login"):
                st.session_state.page = 'login'
                st.experimental_rerun()

# Home page content
def render_home():
    # Hero section
    st.markdown('<div class="scroll-animation">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h2 class="sub-header">Welcome to ApnaBank</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.2rem; color: #555;">
            Your trusted financial partner for all banking needs. ApnaBank offers innovative solutions
            with cutting-edge technology to make your banking experience smooth and efficient.
        </p>
        """, unsafe_allow_html=True)
        
        # Replace HTML buttons with functional Streamlit buttons
        col_btn1, col_btn2, col_spacing = st.columns([1.2, 1.2, 2])
        with col_btn1:
            if st.button("Open Account", key="open_account_btn", use_container_width=True):
                st.session_state.page = 'register'
                st.experimental_rerun()
        with col_btn2:
            if st.button("Learn More", key="learn_more_btn", type="secondary", use_container_width=True):
                st.session_state.page = 'supervised'  # Redirect to any informational page
                st.experimental_rerun()
    
    with col2:
        render_animation(banking_animation, height=300, key="banking")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Services section
    st.markdown('<div class="scroll-animation delay-1">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center; margin-top: 50px;">Our Services</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Personal Banking</div>
            <p class="feature-desc">
                Manage your daily finances with our comprehensive personal banking solutions including savings accounts, fixed deposits, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">🏢</div>
            <div class="feature-title">Corporate Banking</div>
            <p class="feature-desc">
                Tailored financial solutions for businesses of all sizes. From current accounts to business loans and cash management services.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Investment Services</div>
            <p class="feature-desc">
                Grow your wealth with our expert investment advice and services, including mutual funds, stocks, and portfolio management.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # AI-powered features
    st.markdown('<div class="scroll-animation delay-2">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center; margin-top: 50px;">AI-Powered Banking</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        render_animation(analytics_animation, height=250, key="analytics")
    with col2:
        st.markdown("""
        <p style="font-size: 1.1rem; color: #555;">
            At ApnaBank, we leverage advanced AI technologies to provide smarter financial services:
        </p>
        <ul style="font-size: 1rem; color: #555;">
            <li><strong>Personalized Financial Insights:</strong> Get tailored recommendations based on your spending patterns.</li>
            <li><strong>Intelligent Loan Assessment:</strong> Our AI evaluates your eligibility based on comprehensive data analysis.</li>
            <li><strong>Automated Document Processing:</strong> Extract information from payslips and financial documents instantly.</li>
            <li><strong>Fraud Detection:</strong> Stay protected with our advanced AI security systems.</li>
        </ul>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Testimonials
    st.markdown('<div class="scroll-animation delay-3">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center; margin-top: 50px;">What Our Customers Say</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div>⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic;">
                "ApnaBank's AI-powered insights have helped me save more than I ever thought possible. The personalized recommendations are spot on!"
            </p>
            <p style="font-weight: bold;">- Priya Sharma</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div>⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic;">
                "The loan approval process was incredibly smooth. I was impressed by how quickly my application was processed using their AI system."
            </p>
            <p style="font-weight: bold;">- Rajesh Patel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div>⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic;">
                "The VideoText Extractor tool saved me hours of manual data entry. I simply uploaded my documents and all information was extracted instantly."
            </p>
            <p style="font-weight: bold;">- Amit Singh</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Continuing from the previous code...

# Navigation bar
def render_navbar():
    # Title and logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://img.icons8.com/clouds/100/000000/bank-building.png", width=80)
    with col2:
        st.markdown('<h1 class="main-header">ApnaBank</h1>', unsafe_allow_html=True)
    
    selected = option_menu(
        menu_title=None,
        options=["Home", "Supervised Learning", "Unsupervised Learning", "Semi-Supervised Learning", "Loan Checker", "VideoText Extractor", "Dashboard"],
        icons=["house", "clipboard-data", "lightbulb", "graph-up", "calculator", "camera-video", "speedometer"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#f0f2f6"},
            "icon": {"color": "#1A3365", "font-size": "14px"}, 
            "nav-link": {"font-size": "14px", "text-align": "center", "margin":"0px", "--hover-color": "#e6f0ff"},
            "nav-link-selected": {"background-color": "#1A3365"},
        }
    )
    
    if selected == "Home":
        st.session_state.page = 'home'
    elif selected == "Supervised Learning":
        st.session_state.page = 'supervised'
    elif selected == "Unsupervised Learning":
        st.session_state.page = 'unsupervised'
    elif selected == "Semi-Supervised Learning":
        st.session_state.page = 'semi_supervised'
    elif selected == "Loan Checker":
        st.session_state.page = 'loan_checker'
    elif selected == "VideoText Extractor":
        st.session_state.page = 'video_text'
    elif selected == "Dashboard":
        if st.session_state.is_logged_in:
            st.session_state.page = 'dashboard'
        else:
            st.warning("Please login to access the dashboard")
            st.session_state.page = 'login'

# AI-powered features continuation
def render_home():
    # Hero section
    st.markdown('<div class="scroll-animation">', unsafe_allow_html=True)
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown('<h2 class="sub-header">Welcome to ApnaBank</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p style="font-size: 1.2rem; color: #555;">
            Your trusted financial partner for all banking needs. ApnaBank offers innovative solutions
            with cutting-edge technology to make your banking experience smooth and efficient.
        </p>
        """, unsafe_allow_html=True)
        
        # Replace HTML buttons with functional Streamlit buttons
        col_btn1, col_btn2, col_spacing = st.columns([1.2, 1.2, 2])
        with col_btn1:
            if st.button("Open Account", key="open_account_btn", use_container_width=True):
                st.session_state.page = 'register'
                st.experimental_rerun()
        with col_btn2:
            if st.button("Learn More", key="learn_more_btn", type="secondary", use_container_width=True):
                st.session_state.page = 'supervised'  # Redirect to any informational page
                st.experimental_rerun()
    
    with col2:
        render_animation(banking_animation, height=300, key="banking")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Services section
    st.markdown('<div class="scroll-animation delay-1">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center; margin-top: 50px;">Our Services</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">💰</div>
            <div class="feature-title">Personal Banking</div>
            <p class="feature-desc">
                Manage your daily finances with our comprehensive personal banking solutions including savings accounts, fixed deposits, and more.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">🏢</div>
            <div class="feature-title">Corporate Banking</div>
            <p class="feature-desc">
                Tailored financial solutions for businesses of all sizes. From current accounts to business loans and cash management services.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div class="feature-icon">📊</div>
            <div class="feature-title">Investment Services</div>
            <p class="feature-desc">
                Grow your wealth with our expert investment advice and services, including mutual funds, stocks, and portfolio management.
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # AI-powered features
    st.markdown('<div class="scroll-animation delay-2">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center; margin-top: 50px;">AI-Powered Banking</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        render_animation(analytics_animation, height=250, key="analytics")
    with col2:
        st.markdown("""
        <p style="font-size: 1.1rem; color: #555;">
            At ApnaBank, we leverage advanced AI technologies to provide smarter financial services:
        </p>
        <ul style="font-size: 1rem; color: #555;">
            <li><strong>Personalized Financial Insights:</strong> Get tailored recommendations based on your spending patterns.</li>
            <li><strong>Intelligent Loan Assessment:</strong> Our AI evaluates your eligibility based on comprehensive data analysis.</li>
            <li><strong>Automated Document Processing:</strong> Extract information from payslips and financial documents instantly.</li>
            <li><strong>Fraud Detection:</strong> Stay protected with our advanced AI security systems.</li>
        </ul>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Testimonials
    st.markdown('<div class="scroll-animation delay-3">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center; margin-top: 50px;">What Our Customers Say</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <div>⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic;">
                "ApnaBank's AI-powered insights have helped me save more than I ever thought possible. The personalized recommendations are spot on!"
            </p>
            <p style="font-weight: bold;">- Priya Sharma</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <div>⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic;">
                "The loan approval process was incredibly smooth. I was impressed by how quickly my application was processed using their AI system."
            </p>
            <p style="font-weight: bold;">- Rajesh Patel</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="card">
            <div>⭐⭐⭐⭐⭐</div>
            <p style="font-style: italic;">
                "The VideoText Extractor tool saved me hours of manual data entry. I simply uploaded my documents and all information was extracted instantly."
            </p>
            <p style="font-weight: bold;">- Amit Singh</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Login page
def render_login():
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center;">Login to Your Account</h2>', unsafe_allow_html=True)
    
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        login_button = st.button("Login", key="login_btn")
    
    if login_button:
        if username and password:
            # Validate credentials
            session = Session()
            user = session.query(User).filter_by(username=username).first()
            
            if user and user.password == password:
                st.session_state.is_logged_in = True
                st.session_state.current_user = user
                st.session_state.page = 'dashboard'
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
            
            session.close()
        else:
            st.warning("Please enter both username and password")
    
    st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('<a href="#" style="color: #1A3365; text-decoration: none;">Forgot Password?</a>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('Don\'t have an account?', unsafe_allow_html=True)
    if st.button("Register"):
        st.session_state.page = 'register'
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Registration page
def render_register():
    st.markdown('<div class="login-form">', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header" style="text-align: center;">Create Your Account</h2>', unsafe_allow_html=True)
    
    name = st.text_input("Full Name")
    username = st.text_input("Username")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        register_button = st.button("Register", key="register_btn")
    
    if register_button:
        if not all([name, username, email, password, confirm_password]):
            st.warning("Please fill in all fields")
        elif password != confirm_password:
            st.error("Passwords do not match")
        else:
            # Create new user
            session = Session()
            
            # Check if username or email already exists
            existing_user = session.query(User).filter((User.username == username) | (User.email == email)).first()
            
            if existing_user:
                st.error("Username or email already exists")
            else:
                new_user = User(
                    name=name,
                    username=username,
                    email=email,
                    password=password
                )
                
                session.add(new_user)
                session.commit()
                st.success("Registration successful! Please log in.")
                st.session_state.page = 'login'
                st.experimental_rerun()
            
            session.close()
    
    st.markdown('<div style="text-align: center; margin-top: 20px;">', unsafe_allow_html=True)
    st.markdown('Already have an account?', unsafe_allow_html=True)
    if st.button("Login"):
        st.session_state.page = 'login'
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Dashboard
def render_dashboard():
    if not st.session_state.is_logged_in:
        st.warning("Please login to access the dashboard")
        st.session_state.page = 'login'
        st.experimental_rerun()
        return
    
    st.markdown('<h2 class="sub-header">Your Financial Dashboard</h2>', unsafe_allow_html=True)
    
    # User info
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://img.icons8.com/clouds/100/000000/user.png", width=100)
    with col2:
        st.markdown(f"<h3>Welcome, {st.session_state.current_user.name}!</h3>", unsafe_allow_html=True)
        st.markdown(f"<p>Email: {st.session_state.current_user.email}</p>", unsafe_allow_html=True)
        st.markdown(f"<p>CIBIL Score: {st.session_state.current_user.cibil_score}</p>", unsafe_allow_html=True)
    
    # Financial overview
    st.markdown('<h3 class="sub-header">Financial Overview</h3>', unsafe_allow_html=True)
    
    # Get user's transactions
    session = Session()
    transactions = session.query(Transaction).filter_by(user_id=st.session_state.current_user.id).all()
    session.close()
    
    if transactions:
        # Convert to dataframe
        df = pd.DataFrame([{
            'month': t.month,
            'total_income': t.total_income,
            'total_expense': t.total_expense,
            'loan_emi': t.loan_emi,
            'credit_card_payment': t.credit_card_payment,
            'savings': t.savings
        } for t in transactions])
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="dashboard-tile">
                <p class="metric-label">Total Income</p>
                <p class="metric-value">₹{:,.2f}</p>
            </div>
            """.format(df['total_income'].sum()), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="dashboard-tile">
                <p class="metric-label">Total Expenses</p>
                <p class="metric-value">₹{:,.2f}</p>
            </div>
            """.format(df['total_expense'].sum()), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="dashboard-tile">
                <p class="metric-label">Total Savings</p>
                <p class="metric-value">₹{:,.2f}</p>
            </div>
            """.format(df['savings'].sum()), unsafe_allow_html=True)
        
        with col4:
            savings_ratio = df['savings'].sum() / df['total_income'].sum() * 100 if df['total_income'].sum() > 0 else 0
            st.markdown("""
            <div class="dashboard-tile">
                <p class="metric-label">Savings Ratio</p>
                <p class="metric-value">{:.1f}%</p>
            </div>
            """.format(savings_ratio), unsafe_allow_html=True)
            
        # Transaction history
        st.markdown('<h3 class="sub-header">Monthly Transaction History</h3>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True)
        
        # Visualizations
        st.markdown('<h3 class="sub-header">Spending Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Income vs Expenses bar chart
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['month'],
                y=df['total_income'],
                name='Income',
                marker_color='#1A3365'
            ))
            fig.add_trace(go.Bar(
                x=df['month'],
                y=df['total_expense'],
                name='Expenses',
                marker_color='#FF6B6B'
            ))
            fig.update_layout(
                title='Income vs Expenses by Month',
                xaxis_title='Month',
                yaxis_title='Amount (₹)',
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Expense breakdown pie chart
            expenses = {
                'Loan EMI': df['loan_emi'].sum(),
                'Credit Card': df['credit_card_payment'].sum(),
                'Other Expenses': df['total_expense'].sum() - df['loan_emi'].sum() - df['credit_card_payment'].sum()
            }
            
            fig = go.Figure(data=[go.Pie(
                labels=list(expenses.keys()),
                values=list(expenses.values()),
                hole=.4,
                marker_colors=['#1A3365', '#4A6DB5', '#FF6B6B']
            )])
            fig.update_layout(
                title='Expense Breakdown',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Savings trend
        fig = px.line(
            df,
            x='month',
            y='savings',
            markers=True,
            line_shape='spline',
            title='Monthly Savings Trend'
        )
        fig.update_traces(line_color='#1A3365')
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("No transaction data available. Please add your financial data to see insights.")
        
        # Sample data form
        st.markdown('<h3 class="sub-header">Add Transaction Data</h3>', unsafe_allow_html=True)
        
        with st.form("transaction_form"):
            month = st.selectbox("Month", ["January", "February", "March", "April", "May", "June", 
                                         "July", "August", "September", "October", "November", "December"])
            total_income = st.number_input("Total Income", min_value=0.0, format="%.2f")
            total_expense = st.number_input("Total Expenses", min_value=0.0, format="%.2f")
            loan_emi = st.number_input("Loan EMI", min_value=0.0, format="%.2f")
            credit_card_payment = st.number_input("Credit Card Payment", min_value=0.0, format="%.2f")
            
            submitted = st.form_submit_button("Save Transaction")
            
            if submitted:
                savings = total_income - total_expense
                
                session = Session()
                new_transaction = Transaction(
                    user_id=st.session_state.current_user.id,
                    month=month,
                    total_income=total_income,
                    total_expense=total_expense,
                    loan_emi=loan_emi,
                    credit_card_payment=credit_card_payment,
                    savings=savings
                )
                
                session.add(new_transaction)
                session.commit()
                session.close()
                
                st.success("Transaction data saved successfully!")
                st.experimental_rerun()

# Supervised Learning Page
def render_supervised():
    st.markdown('<h2 class="sub-header">Supervised Learning Services</h2>', unsafe_allow_html=True)
    
    # Tabs for different services
    tab1, tab2, tab3 = st.tabs(["Pay", "Invoice", "Salary Slip"])
    
    with tab1:
        st.markdown('<h3 class="sub-header">Payment Services</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p>Our AI-powered payment system uses supervised learning algorithms to:</p>
            <ul>
                <li>Detect and prevent fraudulent transactions</li>
                <li>Provide personalized spending insights</li>
                <li>Optimize payment routing for faster processing</li>
                <li>Predict future expenses based on historical patterns</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h4>Make a Payment</h4>', unsafe_allow_html=True)
        
        with st.form("payment_form"):
            recipient = st.text_input("Recipient Name/Account")
            amount = st.number_input("Amount", min_value=0.0, format="%.2f")
            payment_date = st.date_input("Payment Date")
            description = st.text_area("Description")
            
            submitted = st.form_submit_button("Process Payment")
            
            if submitted:
                if not st.session_state.is_logged_in:
                    st.warning("Please login to process payments")
                else:
                    # Simulate payment processing
                    with st.spinner("Processing payment..."):
                        # Simulating ML fraud detection
                        import time
                        time.sleep(2)
                        
                        st.success(f"Payment of ₹{amount:.2f} to {recipient} scheduled successfully!")
                        st.info("Our AI system has verified this transaction as legitimate.")
    
    with tab2:
        st.markdown('<h3 class="sub-header">Invoice Management</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p>Our invoice management system uses supervised learning to:</p>
            <ul>
                <li>Extract key information from invoice documents</li>
                <li>Classify invoices by vendor and category</li>
                <li>Detect anomalies in billing patterns</li>
                <li>Forecast upcoming invoice amounts</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h4>Upload Invoice</h4>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload invoice PDF or image", type=["pdf", "png", "jpg", "jpeg"])
        
        if uploaded_file is not None:
            # Display preview
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Invoice Preview", use_column_width=True)
            else:
                st.info(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Process Invoice"):
                with st.spinner("Extracting invoice data..."):
                    # Simulating ML processing
                    import time
                    time.sleep(3)
                    
                    # Sample extracted data
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("### Extracted Information")
                        st.markdown("**Vendor:** ABC Corporation")
                        st.markdown("**Invoice #:** INV-2023-45678")
                        st.markdown("**Date:** 2023-03-15")
                        st.markdown("**Amount:** ₹12,500.00")
                    
                    with col2:
                        st.markdown("### AI Analysis")
                        st.markdown("**Category:** Office Supplies")
                        st.markdown("**Confidence Score:** 94%")
                        st.markdown("**Payment Due:** 2023-04-15")
                        st.markdown("**Anomaly Detection:** No irregularities detected")
    
    with tab3:
        st.markdown('<h3 class="sub-header">Salary Slip Analysis</h3>', unsafe_allow_html=True)
        st.markdown("""
        <div class="card">
            <p>Our salary slip analyzer uses supervised learning to:</p>
            <ul>
                <li>Extract salary components and deductions</li>
                <li>Track salary growth over time</li>
                <li>Compare compensation to industry benchmarks</li>
                <li>Provide tax optimization suggestions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<h4>Upload Salary Slip</h4>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload your salary slip", type=["pdf", "png", "jpg", "jpeg"], key="salary_slip")
        
        if uploaded_file is not None:
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Salary Slip Preview", use_column_width=True)
            else:
                st.info(f"File uploaded: {uploaded_file.name}")
            
            if st.button("Analyze Salary Slip"):
                with st.spinner("Analyzing salary components..."):
                    # Simulating ML processing
                    import time
                    time.sleep(3)
                    
                    # Sample analysis
                    st.markdown("### Salary Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        # Basic salary breakdown
                        salary_data = {
                            'Basic Salary': 45000,
                            'HRA': 22500,
                            'Conveyance': 1600,
                            'Special Allowance': 15000,
                            'Medical Allowance': 1250
                        }
                        
                        fig = go.Figure(data=[go.Bar(
                            x=list(salary_data.keys()),
                            y=list(salary_data.values()),
                            marker_color='#1A3365'
                        )])
                        fig.update_layout(
                            title='Salary Components',
                            xaxis_title='Component',
                            yaxis_title='Amount (₹)',
                            height=400
                        )
                        st.plotly_chart(fig)
                    
                    with col2:
                        # Deductions
                        deduction_data = {
                            'PF': 5400,
                            'Income Tax': 7500,
                            'Professional Tax': 200,
                            'Health Insurance': 1800
                        }
                        
                        fig = go.Figure(data=[go.Pie(
                            labels=list(deduction_data.keys()),
                            values=list(deduction_data.values()),
                            hole=.4,
                            marker_colors=['#1A3365', '#4A6DB5', '#6384D3', '#FF6B6B']
                        )])
                        fig.update_layout(
                            title='Deductions',
                            height=400
                        )
                        st.plotly_chart(fig)
                    
                    # Insights
                    st.markdown("### AI Insights")
                    st.info("📈 Your salary is 15% above the industry average for your role.")
                    st.info("💡 You could save approximately ₹12,000 annually by increasing your 80C investments.")
                    st.info("📊 Based on your salary growth pattern, you're projected to reach ₹1,20,000 monthly by next year.")

# Unsupervised Learning Page
def render_unsupervised():
    st.markdown('<h2 class="sub-header">Unsupervised Learning Services</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>Our unsupervised learning services leverage advanced AI algorithms to discover patterns and insights without pre-labeled data.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 class="sub-header">LLM-Based Financial Insights</h3>', unsafe_allow_html=True)
    
    # Simulated LLM interface for financial advice
    user_query = st.text_area("Ask about your finances:", placeholder="E.g., How can I improve my credit score? Or What's the best way to save for retirement?")
    
    if st.button("Get AI Insights"):
        with st.spinner("Analyzing your query..."):
            # Simulating AI processing
            import time
            time.sleep(2)
            
            if "credit score" in user_query.lower():
                st.markdown("### How to Improve Your Credit Score")
                st.markdown("""
                Based on our analysis of financial patterns, here are personalized recommendations:
                
                1. **Pay bills on time**: Late payments are one of the biggest factors affecting your score.
                2. **Reduce credit utilization**: Try to use less than 30% of your available credit.
                3. **Don't close old credit accounts**: Length of credit history matters.
                4. **Diversify your credit mix**: Having different types of credit (credit cards, loans) can help.
                5. **Check your credit report regularly**: Dispute any errors you find.
                
                Our AI has identified that users with profiles similar to yours have seen an average 40-point improvement in 6 months by following these steps.
                """)
            
            elif "save" in user_query.lower() and "retirement" in user_query.lower():
                st.markdown("### Retirement Savings Strategy")
                st.markdown("""
                Based on clustering analysis of successful retirement savers, here's a personalized approach:
                
                1. **Start with employer matching**: If your employer offers a retirement plan with matching contributions, contribute at least enough to get the full match.
                2. **Follow the 15% rule**: Aim to save at least 15% of your pre-tax income for retirement.
                3. **Diversify investments**: Consider a mix of stocks, bonds, and other assets based on your age and risk tolerance.
                4. **Increase contributions over time**: Gradually increase your savings rate, especially after raises.
                5. **Consider tax-advantaged accounts**: Maximize contributions to tax-advantaged retirement accounts.
                
                Our pattern analysis shows that users with similar financial profiles who followed this approach accumulated 1.8x more retirement savings than average.
                """)
            
            elif "invest" in user_query.lower():
                st.markdown("### Investment Strategy Insights")
                st.markdown("""
                Our unsupervised learning algorithms have clustered investment patterns and identified optimal strategies for your profile:
                
                1. **Asset allocation**: Based on users with similar risk profiles, a mix of 60% stocks, 30% bonds, and 10% alternative investments may be optimal.
                2. **Sector distribution**: Technology, healthcare, and green energy sectors show promising growth patterns for your investment horizon.
                3. **Investment timing**: Market pattern analysis suggests small, regular investments rather than large lump sums show better long-term results.
                4. **Risk management**: Our algorithms suggest maintaining an emergency fund of 6 months' expenses before increasing investment allocation.
                
                Users with similar profiles have seen an average annual return of 12.3% using a pattern-based investment approach.
                """)
            
            else:
                st.markdown("### Financial Insights")
                st.markdown("""
                Our AI has analyzed your query and thousands of similar financial patterns. Here are some general insights:
                
                1. **Budget optimization**: Our clustering analysis shows that the most financially successful users follow the 50/30/20 rule - 50% on needs, 30% on wants, and 20% on savings.
                2. **Debt management**: Pattern recognition suggests prioritizing high-interest debt first while maintaining minimum payments on other debts.
                3. **Financial habits**: Successful financial patterns show consistent monitoring of expenses, regular financial reviews, and automated savings.
                4. **Future planning**: Long-term financial success correlates strongly with setting specific financial goals and regularly tracking progress.
                
                For more personalized insights, please ask a specific financial question related to credit, savings, investments, or debt management.
                """)
    
    # Spending pattern analysis section
    st.markdown('<h3 class="sub-header">Spending Pattern Analysis</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
        <p>Upload your bank statement to discover hidden patterns in your spending behavior using our clustering algorithms.</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_statement = st.file_uploader("Upload bank statement (CSV format)", type=["csv"])
    
    if uploaded_statement is not None:
        try:
            # Sample data for demonstration when user uploads a file
            data = {
                'Category': ['Groceries', 'Dining', 'Entertainment', 'Shopping', 'Transport', 'Utilities', 'Rent', 'Healthcare'],
                'Amount': [12500, 8700, 5400, 9200, 4500, 3800, 25000, 2800]
            }
            
            df = pd.DataFrame(data)
            
            st.success("Bank statement processed successfully!")
            
            # Display spending clusters
            st.markdown("### Your Spending Clusters")
            
            fig = px.pie(
                df, 
                values='Amount', 
                names='Category',
                title='Spending Distribution by Category',
                color_discrete_sequence=px.colors.sequential.Blues_r
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomaly detection
            st.markdown("### Spending Anomaly Detection")
            
            # Simulate anomaly scores
            anomaly_data = {
                'Transaction': ['Grocery Store 03/05', 'Restaurant 03/12', 'Online Shopping 03/15', 'Electronics Store 03/20', 'Subscription 03/25'],
                'Amount': [2500, 4800, 12500, 35000, 1500],
                'Anomaly_Score': [0.12, 0.18, 0.58, 0.92, 0.08]
            }
            
            anomaly_df = pd.DataFrame(anomaly_data)
            
            fig = px.bar(
                anomaly_df,
                x='Transaction',
                y='Anomaly_Score',
                color='Anomaly_Score',
                color_continuous_scale='Blues',
                title='Transaction Anomaly Scores'
            )
            
            fig.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Anomaly Threshold")
            st.plotly_chart(fig, use_container_width=True)
            
            # Highlight anomalies
            st.markdown("### Unusual Spending Patterns Detected")
            anomalies = anomaly_df[anomaly_df['Anomaly_Score'] > 0.5]
            
            for _, row in anomalies.iterrows():
                st.warning(f"Unusual transaction detected: {row['Transaction']} - ₹{row['Amount']} with anomaly score of {row['Anomaly_Score']:.2f}")
            
            # Spending recommendations based on clusters
            st.markdown("### AI Recommendations")
            st.info("📊 Based on spending patterns of similar users, you could save approximately ₹8,500 monthly by optimizing your shopping and dining expenses.")
            st.info("💡 Your spending patterns indicate potential savings opportunities in the Entertainment category compared to users with similar income.")
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

# Semi-Supervised Learning Page
def render_semi_supervised():
    st.markdown('<h2 class="sub-header">Semi-Supervised Learning Services</h2>', unsafe_allow_html=True)
    
    # Tabs for stocks and API
    tab1, tab2 = st.tabs(["Stock Analysis", "Financial API Integration"])
    
    with tab1:
        st.markdown('<h3 class="sub-header">AI-Powered Stock Analysis</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            render_animation(stock_animation, height=200, key="stock_animation")
        
        with col2:
            st.markdown("""
            <div class="card">
                <p>Our semi-supervised learning model combines labeled market data with vast amounts of unlabeled financial data to predict stock trends with higher accuracy.</p>
                <ul>
                    <li>Pattern recognition in market movements</li>
                    <li>Sentiment analysis from financial news</li>
                    <li>Technical indicator optimization</li>
                    <li>Risk assessment based on market volatility</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Stock selector
        stock_options = ["RELIANCE", "TCS", "HDFC", "ICICI", "INFY", "BAJFINANCE", "SBIN", "LT", "HCLTECH", "ADANIENT"]
        selected_stock = st.selectbox("Select a stock to analyze:", stock_options)
        
        # Time period
        time_period = st.radio("Select time period:", ["1 Month", "3 Months", "6 Months", "1 Year"], horizontal=True)
        
        if st.button("Analyze Stock"):
            with st.spinner(f"Analyzing {selected_stock} with our semi-supervised AI models..."):
                # Simulate processing time
                import time
                time.sleep(2)
                
                # Generate sample stock data
                import numpy as np
                
                days = 30
                if time_period == "3 Months":
                    days = 90
                elif time_period == "6 Months":
                    days = 180
                elif time_period == "1 Year":
                    days = 365
                
                # Generate dates
                end_date = pd.Timestamp.today()
                date_range = pd.date_range(end=end_date, periods=days)
                
                # Generate price data
                start_price = np.random.randint(1000, 3000)
                
                # Add some randomness and trend
                np.random.seed(42)  # For reproducibility
                daily_returns = np.random.normal(0.0005, 0.015, size=days)
                
                # Add a trend based on the stock selected
                if selected_stock in ["RELIANCE", "TCS", "HDFC"]:
                    daily_returns += 0.001  # Positive trend
                elif selected_stock in ["ADANIENT"]:
                    daily_returns -= 0.0005  # Negative trend
                
                # Calculate price series
                price_series = [start_price]
                for ret in daily_returns:
                    price_series.append(price_series[-1] * (1 + ret))
                
                price_series = price_series[1:]  # Remove the initial seed
                
                # Create DataFrame
                stock_data = pd.DataFrame({
                    'Date': date_range,
                    'Price': price_series,
                })
                
                # Calculate moving averages
                stock_data['MA_20'] = stock_data['Price'].rolling(window=20).mean()
                stock_data['MA_50'] = stock_data['Price'].rolling(window=50).mean()
                
                # Create plotly figure
                fig = go.Figure()
                
                # Add price line
                fig.add_trace(go.Scatter(
                    x=stock_data['Date'],
                    y=stock_data['Price'],
                    name='Price',
                    line=dict(color='#1A3365')
                ))
                
                # Add moving averages
                if days >= 20:
                    fig.add_trace(go.Scatter(
                        x=stock_data['Date'],
                        y=stock_data['MA_20'],
                        name='20-Day MA',
                        line=dict(color='#FF6B6B', dash='dot')
                    ))
                
                if days >= 50:
                    fig.add_trace(go.Scatter(
                        x=stock_data['Date'],
                        y=stock_data['MA_50'],
                        name='50-Day MA',
                        line=dict(color='#4CAF50', dash='dot')
                    ))
                
                # Add trend prediction region
                future_days = int(days * 0.2)  # 20% of the selected period for prediction
                last_date = stock_data['Date'].iloc[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
                
                # Generate future predictions with some uncertainty
                last_price = stock_data['Price'].iloc[-1]
                
                # Calculate trend from the last 30 days
                trend = 0
                if len(stock_data) > 30:
                    trend = (stock_data['Price'].iloc[-1] - stock_data['Price'].iloc[-30]) / stock_data['Price'].iloc[-30]
                else:
                    trend = (stock_data['Price'].iloc[-1] - stock_data['Price'].iloc[0]) / stock_data['Price'].iloc[0]
                
                # Adjust trend based on stock
                if selected_stock in ["RELIANCE", "TCS", "HDFC"]:
                    trend += 0.002
                elif selected_stock in ["ADANIENT"]:
                    trend -= 0.002
                
                # Create prediction with uncertainty
                future_daily_returns = np.random.normal(trend/days, 0.015, size=future_days)
                future_prices = [last_price]
                for ret in future_daily_returns:
                    future_prices.append(future_prices[-1] * (1 + ret))
                
                future_prices = future_prices[1:]  # Remove the seed
                
                # Calculate upper and lower bounds
                uncertainty = 0.05  # 5% uncertainty
                upper_bound = [price * (1 + uncertainty * (i/future_days)) for i, price in enumerate(future_prices)]
                lower_bound = [price * (1 - uncertainty * (i/future_days)) for i, price in enumerate(future_prices)]
                
                # Add prediction line
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=future_prices,
                    name='AI Prediction',
                    line=dict(color='#FFD700', dash='solid')
                ))
                
                # Add uncertainty range
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=upper_bound,
                    fill=None,
                    mode='lines',
                    line_color='rgba(255, 215, 0, 0)',
                    showlegend=False
                ))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=lower_bound,
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(255, 215, 0, 0)',
                    fillcolor='rgba(255, 215, 0, 0.2)',
                    name='Prediction Range'
                ))
                
                # Update layout
                fig.update_layout(
                    title=f'{selected_stock} Stock Analysis with AI Predictions',
                    xaxis_title='Date',
                    yaxis_title='Price (₹)',
                    height=500
                )

# Loan Checker Page
def render_loan_checker():
    st.markdown('<h2 class="sub-header">Loan Eligibility Checker</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_animation(loan_animation, height=250, key="loan_animation")
    
    with col2:
        st.markdown("""
        <div class="card">
            <p style="font-size: 1.1rem;">
                Our advanced AI-powered loan eligibility system uses your financial data to determine loan eligibility instantly.
            </p>
            <ul>
                <li>Upload your recent payslip to assess your financial health</li>
                <li>Our AI extracts key information automatically</li>
                <li>Get your CIBIL score prediction and loan eligibility</li>
                <li>Special consideration for graduates from Tier-1 institutions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Document upload section
    st.markdown('<h3 class="sub-header">Upload Your Payslip</h3>', unsafe_allow_html=True)
    
    # File uploader for payslip
    uploaded_file = st.file_uploader("Choose a payslip (PDF or image)", type=["pdf", "png", "jpg", "jpeg"])
    
    # Educational background form
    st.markdown("<h4>Educational Background</h4>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        college = st.selectbox("College/University", [
            "Select your college/university",
            "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
            "NIT Trichy", "NIT Surathkal", "NIT Warangal", 
            "BITS Pilani", "BITS Goa", "BITS Hyderabad",
            "Delhi University", "Mumbai University", "Pune University",
            "Other"
        ])
    with col2:
        college_rank = st.number_input("Your rank in college", min_value=1, max_value=5000, value=100)
    
    # Process button
    process_button = st.button("Check Loan Eligibility")
    
    if process_button:
        if uploaded_file is None:
            st.warning("Please upload your payslip to check eligibility")
        else:
            with st.spinner("Processing document and calculating eligibility..."):
                # Extract text from payslip using OCR
                extracted_text = ""
                income = 0
                
                try:
                    # For image files
                    if uploaded_file.type.startswith('image'):
                        image = Image.open(uploaded_file)
                        image_np = np.array(image)
                        extracted_text = pytesseract.image_to_string(image_np)
                    
                    # For PDF files (would require pdf2image in a real implementation)
                    elif uploaded_file.type == 'application/pdf':
                        st.info("PDF processing: In a full implementation, we would use pdf2image or PyPDF2")
                        extracted_text = "Sample PDF extraction - Basic Salary: Rs. 45,000\nHRA: Rs. 18,000\nDA: Rs. 9,000\nNet Pay: Rs. 65,000"
                        
                    # Display extracted text (for debugging)
                    with st.expander("View Extracted Text"):
                        st.text(extracted_text)
                    
                    # Process the extracted text to find income and other financial details
                    # This is a simplified version - a real implementation would use regex or NLP
                    
                    # Let's simulate finding salary information
                    salary_match = re.search(r"(?:salary|pay|income)\s*(?::|is|rs\.?|inr)?\s*(?:rs\.?|inr)?\s*([0-9,]+)", 
                                           extracted_text.lower())
                    
                    if salary_match:
                        income_str = salary_match.group(1).replace(',', '')
                        income = float(income_str)
                    else:
                        # If we can't find it, use a mock value
                        income = 65000
                    
                    # Display extracted information
                    st.markdown("<h4>Extracted Financial Information</h4>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.info(f"📄 Detected Monthly Income: ₹{income:,.2f}")
                        
                        # Other financial details (mock data for demo)
                        deductions = income * 0.15
                        net_pay = income - deductions
                        
                        st.info(f"📄 Detected Deductions: ₹{deductions:,.2f}")
                        st.info(f"📄 Detected Net Pay: ₹{net_pay:,.2f}")
                    
                    with col2:
                        if uploaded_file.type.startswith('image'):
                            st.image(uploaded_file, caption="Uploaded Payslip", width=300)
                        else:
                            st.success(f"Processed {uploaded_file.name}")
                    
                    # Load the pre-trained CIBIL model
                    model = load_model()
                    
                    if model is None:
                        st.error("Could not load the CIBIL prediction model. Using fallback evaluation.")
                        # Fallback logic
                        cibil_score = int(min(max(income / 200, 550), 900))
                    else:
                        # Prepare features for model prediction
                        # In a real scenario, these would be extracted from the document
                        # Here we're creating mock data based on the income
                        features = {
                            'age': 30,  # mock value
                            'income': income,
                            'employment_type': 'Salaried',
                            'existing_loans': 1,  # mock value
                            'credit_card_usage': 0.3,  # mock value
                            'repayment_history': 0.9,  # mock value
                        }
                        
                        # Convert to DataFrame for model
                        X = pd.DataFrame([features])
                        
                        # In a real app, use the actual model prediction
                        # predicted_cibil = model.predict(X)[0]
                        
                        # For demo purposes, calculate a mock score based on income
                        cibil_score = int(min(max(income / 200 + np.random.randint(-50, 50), 550), 900))
                    
                    # Check if the user is from a Tier-1 college
                    tier1_colleges = [
                        "IIT Bombay", "IIT Delhi", "IIT Madras", "IIT Kanpur", "IIT Kharagpur",
                        "NIT Trichy", "NIT Surathkal", "NIT Warangal", 
                        "BITS Pilani", "BITS Goa", "BITS Hyderabad"
                    ]
                    
                    is_tier1 = college in tier1_colleges
                    is_top_ranker = college_rank <= 100
                    
                    # Calculate loan eligibility
                    # Standard rule: CIBIL > 700 is eligible
                    # Special rule: Tier-1 college + top ranker is eligible regardless of CIBIL
                    standard_eligible = cibil_score >= 700
                    special_eligible = is_tier1 and is_top_ranker
                    
                    is_eligible = standard_eligible or special_eligible
                    
                    # Set max loan amount based on income
                    max_loan_amount = income * 36 if is_eligible else 0
                    
                    # Display results
                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown("<h3 class='sub-header'>Loan Eligibility Results</h3>", unsafe_allow_html=True)
                    
                    # Display CIBIL score with color coding
                    score_color = "#28a745" if cibil_score >= 750 else "#ffc107" if cibil_score >= 650 else "#dc3545"
                    
                    col1, col2, col3 = st.columns([1, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 10px; border: 2px solid {score_color}; text-align: center;">
                            <h4 style="margin-bottom: 10px;">Predicted CIBIL Score</h4>
                            <h2 style="color: {score_color}; margin: 10px 0;">{cibil_score}</h2>
                            <p>{'Excellent' if cibil_score >= 750 else 'Good' if cibil_score >= 700 else 'Fair' if cibil_score >= 650 else 'Poor'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        eligibility_color = "#28a745" if is_eligible else "#dc3545"
                        eligibility_text = "Eligible" if is_eligible else "Not Eligible"
                        
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 10px; border: 2px solid {eligibility_color}; text-align: center;">
                            <h4 style="margin-bottom: 10px;">Loan Eligibility</h4>
                            <h2 style="color: {eligibility_color}; margin: 10px 0;">{eligibility_text}</h2>
                            <p>{'Approved' if is_eligible else 'Not Approved'}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div style="background-color: white; padding: 20px; border-radius: 10px; border: 2px solid #1A3365; text-align: center;">
                            <h4 style="margin-bottom: 10px;">Maximum Loan Amount</h4>
                            <h2 style="color: #1A3365; margin: 10px 0;">₹{max_loan_amount:,.0f}</h2>
                            <p>Based on your income</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Show explanation for special eligibility
                    if special_eligible and not standard_eligible:
                        st.success("🎓 Special Eligibility: You qualify for our educational institution special program! As a top performer from a Tier-1 institution, you're eligible despite having a lower CIBIL score.")
                    
                    # Additional recommendations
                    st.markdown("<h4>Personalized Recommendations</h4>", unsafe_allow_html=True)
                    
                    if is_eligible:
                        st.markdown("""
                        <div class="card">
                            <h5 style="color: #28a745;">✅ Congratulations! Here are your loan options:</h5>
                            <ul>
                                <li><strong>Personal Loan:</strong> Up to 36 times your monthly income</li>
                                <li><strong>Home Loan:</strong> Up to 60 times your monthly income with property as collateral</li>
                                <li><strong>Education Loan:</strong> Special rates for advanced studies</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="card">
                            <h5 style="color: #dc3545;">Steps to improve your eligibility:</h5>
                            <ul>
                                <li><strong>Improve your credit score</strong> by paying bills on time</li>
                                <li><strong>Reduce existing debt</strong> before applying again</li>
                                <li><strong>Apply with a co-applicant</strong> to enhance eligibility</li>
                                <li><strong>Consider a secured loan</strong> with collateral</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    st.error(f"An error occurred while processing your document: {e}")
                    st.info("Please make sure you've uploaded a valid payslip document.")

# Footer component
def render_footer():
    st.markdown("""
    <footer style="background-color: #1A3365; color: white; padding: 40px 0; margin-top: 50px; border-top: 5px solid #FFD700;">
        <div style="max-width: 1200px; margin: 0 auto; padding: 0 20px;">
            <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                <!-- Company Information -->
                <div style="flex: 1; min-width: 250px; margin-bottom: 20px; padding-right: 20px;">
                    <h3 style="color: white; font-size: 1.5rem; margin-bottom: 20px;">ApnaBank</h3>
                    <p style="color: #ccc; margin-bottom: 15px;">Your trusted financial partner offering innovative banking solutions with cutting-edge AI technology.</p>
                    <p style="color: #ccc;"><strong>CIN:</strong> L65110MH1994PLC078243</p>
                </div>
                
                <!-- Contact Information -->
                <div style="flex: 1; min-width: 250px; margin-bottom: 20px;">
                    <h3 style="color: white; font-size: 1.2rem; margin-bottom: 20px;">Contact Us</h3>
                    <p style="color: #ccc; margin-bottom: 10px;">
                        <i class="fas fa-phone-alt" style="margin-right: 10px;"></i> Customer Care: 1800-258-9999
                    </p>
                    <p style="color: #ccc; margin-bottom: 10px;">
                        <i class="fas fa-phone-alt" style="margin-right: 10px;"></i> Toll Free: 1800-123-4567
                    </p>
                    <p style="color: #ccc; margin-bottom: 10px;">
                        <i class="fas fa-envelope" style="margin-right: 10px;"></i> Email: care@apnabank.com
                    </p>
                    <p style="color: #ccc;">
                        <i class="fas fa-map-marker-alt" style="margin-right: 10px;"></i> ApnaBank Tower, Bandra Kurla Complex, Mumbai - 400051
                    </p>
                </div>
                
                <!-- Quick Links -->
                <div style="flex: 1; min-width: 250px; margin-bottom: 20px;">
                    <h3 style="color: white; font-size: 1.2rem; margin-bottom: 20px;">Quick Links</h3>
                    <ul style="list-style: none; padding: 0; margin: 0;">
                        <li style="margin-bottom: 10px;"><a href="#" style="color: #ccc; text-decoration: none; transition: color 0.3s;">About Us</a></li>
                        <li style="margin-bottom: 10px;"><a href="#" style="color: #ccc; text-decoration: none; transition: color 0.3s;">Products & Services</a></li>
                        <li style="margin-bottom: 10px;"><a href="#" style="color: #ccc; text-decoration: none; transition: color 0.3s;">Interest Rates</a></li>
                        <li style="margin-bottom: 10px;"><a href="#" style="color: #ccc; text-decoration: none; transition: color 0.3s;">Careers</a></li>
                        <li><a href="#" style="color: #ccc; text-decoration: none; transition: color 0.3s;">Locate Branch</a></li>
                    </ul>
                </div>
                
                <!-- Connect with us -->
                <div style="flex: 1; min-width: 250px; margin-bottom: 20px;">
                    <h3 style="color: white; font-size: 1.2rem; margin-bottom: 20px;">Connect With Us</h3>
                    <div style="display: flex; gap: 15px; margin-bottom: 20px;">
                        <a href="#" style="color: white; background-color: #3b5998; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; border-radius: 50%; text-decoration: none;">
                            <span>f</span>
                        </a>
                        <a href="#" style="color: white; background-color: #1da1f2; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; border-radius: 50%; text-decoration: none;">
                            <span>t</span>
                        </a>
                        <a href="#" style="color: white; background-color: #0077b5; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; border-radius: 50%; text-decoration: none;">
                            <span>in</span>
                        </a>
                        <a href="#" style="color: white; background-color: #ff0000; width: 36px; height: 36px; display: flex; align-items: center; justify-content: center; border-radius: 50%; text-decoration: none;">
                            <span>▶</span>
                        </a>
                    </div>
                    
                    <h4 style="color: white; font-size: 1.1rem; margin-bottom: 10px;">Mobile Banking</h4>
                    <div style="display: flex; gap: 10px;">
                        <img src="https://img.icons8.com/color/48/000000/google-play.png" style="width: 120px; height: auto;"/>
                        <img src="https://img.icons8.com/color/48/000000/app-store.png" style="width: 120px; height: auto;"/>
                    </div>
                </div>
            </div>
            
            <!-- Bottom Footer -->
            <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid rgba(255, 255, 255, 0.1);">
                <div style="display: flex; flex-wrap: wrap; justify-content: space-between;">
                    <div style="margin-bottom: 10px;">
                        <p style="color: #ccc; margin: 0;">© 2023 ApnaBank. All Rights Reserved.</p>
                    </div>
                    <div>
                        <a href="#" style="color: #ccc; text-decoration: none; margin-right: 20px;">Terms & Conditions</a>
                        <a href="#" style="color: #ccc; text-decoration: none; margin-right: 20px;">Privacy Policy</a>
                        <a href="#" style="color: #ccc; text-decoration: none;">Disclaimer</a>
                    </div>
                </div>
            </div>
        </div>
    </footer>
    
    <script src="https://kit.fontawesome.com/a076d05399.js"></script>
    """, unsafe_allow_html=True)

# Main execution block
if __name__ == "__main__":
    # First load the model in the background
    model = load_model()
    
    # Render the navigation bar
    render_navbar()
    
    # Render the appropriate page based on session state
    if st.session_state.page == 'home':
        render_home()
    elif st.session_state.page == 'login':
        render_login()
    elif st.session_state.page == 'register':
        render_register()
    elif st.session_state.page == 'dashboard':
        render_dashboard()
    elif st.session_state.page == 'supervised':
        render_supervised()
    elif st.session_state.page == 'unsupervised':
        render_unsupervised()
    elif st.session_state.page == 'semi_supervised':
        render_semi_supervised()
    elif st.session_state.page == 'loan_checker':
        render_loan_checker()
    elif st.session_state.page == 'video_text':
        # You'll need to implement this function
        st.markdown('<h2 class="sub-header">Video Text Extractor</h2>', unsafe_allow_html=True)
        st.info("This feature is coming soon!")
    
    # Render the footer
    render_footer()