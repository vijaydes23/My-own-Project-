import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import time
import random

# --- 1. CONFIGURATION & MOCK DATA SETUP ---

# Configure Streamlit page settings
st.set_page_config(
    page_title="Student Career Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define Career Profiles and Skills
CAREER_PROFILES = [
    'Data Analyst', 'Data Scientist', 'Software Developer', 'Web Developer',
    'Cloud Engineer', 'Cybersecurity Analyst', 'Product Manager', 'Business Analyst',
    'AI Engineer', 'Machine Learning Engineer', 'Digital Marketer', 'Graphic Designer',
    'UI/UX Designer', 'System Administrator', 'Financial Analyst'
]

SKILLS = [
    'Python', 'SQL', 'Power BI', 'Excel', 'Machine Learning', 'Deep Learning',
    'Cloud Computing', 'Cybersecurity', 'UI/UX Design', 'Product Management',
    'Blockchain', 'Finance', 'Digital Marketing', 'DevOps', 'Communication',
    'Leadership', 'Problem Solving', 'Visual Design'  # FIX: Added 'Visual Design' to fix KeyError
]

# Education Levels for Label Encoding
EDUCATION_LEVELS = ['10th', '12th', 'Graduation', 'Postgrad']

# Map career to expected average salary (in INR lakhs/year) and growth factor
CAREER_MAP = {
    'Data Analyst': {'base_salary': 6.0, 'growth_factor': 1.2, 'demand': 0.7},
    'Data Scientist': {'base_salary': 15.0, 'growth_factor': 1.8, 'demand': 0.6},
    'Software Developer': {'base_salary': 10.0, 'growth_factor': 1.5, 'demand': 0.9},
    'Web Developer': {'base_salary': 7.0, 'growth_factor': 1.3, 'demand': 0.8},
    'Cloud Engineer': {'base_salary': 12.0, 'growth_factor': 1.7, 'demand': 0.9},
    'Cybersecurity Analyst': {'base_salary': 9.0, 'growth_factor': 1.6, 'demand': 0.8},
    'Product Manager': {'base_salary': 18.0, 'growth_factor': 1.9, 'demand': 0.5},
    'Business Analyst': {'base_salary': 8.0, 'growth_factor': 1.1, 'demand': 0.7},
    'AI Engineer': {'base_salary': 16.0, 'growth_factor': 2.0, 'demand': 0.6},
    'Machine Learning Engineer': {'base_salary': 14.0, 'growth_factor': 1.9, 'demand': 0.6},
    'Digital Marketer': {'base_salary': 5.0, 'growth_factor': 1.0, 'demand': 0.7},
    'Graphic Designer': {'base_salary': 4.0, 'growth_factor': 0.9, 'demand': 0.5},
    'UI/UX Designer': {'base_salary': 8.0, 'growth_factor': 1.4, 'demand': 0.7},
    'System Administrator': {'base_salary': 6.0, 'growth_factor': 1.0, 'demand': 0.6},
    'Financial Analyst': {'base_salary': 9.0, 'growth_factor': 1.1, 'demand': 0.6}
}

# --- 2. DATA GENERATION AND MODEL TRAINING ---

@st.cache_data
def generate_and_train_model():
    """Generates synthetic data and trains the Classification and Regression models."""
    N = 1000  # Number of samples
    data = []

    # Encoders
    le_career = LabelEncoder()
    le_education = LabelEncoder()
    le_education.fit(EDUCATION_LEVELS)

    for i in range(N):
        # Academic & Personal Features
        age = random.randint(20, 26)
        gender = random.choice(['Male', 'Female', 'Other'])
        edu = random.choice(EDUCATION_LEVELS)
        score_10 = random.randint(50, 95)
        score_12 = random.randint(50, 95)
        cgpa = round(random.uniform(5.0, 9.9), 1)

        # Interests & Work Type
        interests = random.sample(SKILLS[:10], random.randint(1, 4)) # Tech interests influence
        work_type = random.choice(['Remote', 'On-site'])
        future_goal = random.choice(['Job', 'Higher Studies', 'Startup'])

        # Skills (Weighted towards interests)
        user_skills = {skill: 0 for skill in SKILLS}
        for skill in interests:
            user_skills[skill] = 1 # Core skills based on interest
        for skill in SKILLS:
            if user_skills[skill] == 0 and random.random() < 0.2: # Randomly add non-core skills
                user_skills[skill] = 1

        # Determine Target Career (based on weighted score)
        tech_score = (user_skills['Python'] + user_skills['Machine Learning'] + user_skills['Cloud Computing']) * 10
        # FIX: Replaced 'Graphic Designer' (Job Profile) with 'Visual Design' (Skill)
        design_score = (user_skills['UI/UX Design'] + user_skills['Visual Design']) * 8
        finance_score = (user_skills['Finance'] + user_skills['Excel']) * 7
        comm_score = (user_skills['Communication'] + user_skills['Leadership']) * 5

        scores = {
            'Data Analyst': tech_score * 0.4 + comm_score * 0.3 + finance_score * 0.3,
            'Software Developer': tech_score * 0.5 + cgpa * 5 + comm_score * 0.2,
            'Product Manager': comm_score * 0.6 + finance_score * 0.3 + tech_score * 0.1,
            'UI/UX Designer': design_score * 0.7 + comm_score * 0.2,
            'Financial Analyst': finance_score * 0.7 + comm_score * 0.3,
            'Graphic Designer': design_score * 0.6 + comm_score * 0.1, # Added scoring for consistency
        }

        # Select the highest scoring career, with some randomness
        target_career = max(scores, key=scores.get) if scores else random.choice(CAREER_PROFILES)
        
        # Salary Calculation (with noise)
        career_meta = CAREER_MAP[target_career]
        base = career_meta['base_salary']
        # Salary is based on career base, CGPA, and number of tech skills
        num_skills = sum(user_skills.values())
        salary = (base * career_meta['growth_factor']) + (cgpa * 0.5) + (num_skills * 0.2) + random.uniform(-1.0, 1.0)
        salary = max(3.0, salary) # Min salary cap

        # Append to dataset
        row = {
            'Age': age, 'Gender': gender, 'Education': edu, '10th %': score_10,
            '12th %': score_12, 'CGPA': cgpa, 'Work Type': work_type,
            'Future Goal': future_goal, 'Target Career': target_career,
            'Estimated Salary': round(salary, 2)
        }
        row.update(user_skills)
        data.append(row)

    df = pd.DataFrame(data)

    # --- Preprocessing ---
    X = df.drop(['Target Career', 'Estimated Salary', 'Gender', 'Work Type', 'Future Goal'], axis=1)
    y_class = df['Target Career']
    y_reg = df['Estimated Salary']

    X['Education_Encoded'] = le_education.transform(X['Education'])
    X = X.drop('Education', axis=1)

    # --- Training Classification Model ---
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_class, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_c, y_train_c)
    
    # --- Training Regression Model ---
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = LinearRegression() # Simpler model for salary
    reg.fit(X_train_r, y_train_r)

    return clf, reg, X.columns, le_career, le_education, df

# Load models and data
clf, reg, feature_names, le_career, le_education, df_full = generate_and_train_model()

# --- 3. UTILITY AND MOCK AI FUNCTIONS ---

def mock_resume_parser(uploaded_file):
    """Mocks PDF parsing and extracts random skills."""
    if uploaded_file is not None:
        st.info("Parsing resume... (Simulated extraction)")
        time.sleep(1) # Simulate I/O time
        
        # Return a realistic subset of skills based on the profile
        mock_skills = random.sample([
            'Python', 'SQL', 'Communication', 'Problem Solving', 'Excel', 'Leadership', 'Cloud Computing'
        ], random.randint(3, 7))
        return mock_skills
    return []

def preprocess_user_data(user_input, all_features):
    """Converts user input dictionary into a feature vector for the models."""
    processed = {}

    # Education encoding
    edu_encoded = le_education.transform([user_input['education']])[0]
    
    # Base features
    processed['Age'] = user_input['age']
    processed['10th %'] = user_input['score_10']
    processed['12th %'] = user_input['score_12']
    processed['CGPA'] = user_input['cgpa']
    processed['Education_Encoded'] = edu_encoded
    
    # Skills and Interests (One-Hot Encoded)
    all_skills_and_interests = SKILLS # SKILLS includes both tech and soft
    
    for feature in all_skills_and_interests:
        # Check if the feature is a skill or an interest
        is_active = 1 if (feature in user_input['technical_skills'] or feature in user_input['soft_skills'] or feature in user_input['interests']) else 0
        processed[feature] = is_active

    # Create the final feature vector (must match training feature order)
    feature_vector = [processed[col] for col in all_features]
    
    return np.array(feature_vector).reshape(1, -1)


def generate_ai_recommendations(career_role, confidence_score):
    """
    Simulates a detailed LLM call to generate personalized recommendations.
    """
    base_salary = CAREER_MAP[career_role]['base_salary']
    
    recommendations = {
        'top_matches': [
            career_role,
            random.choice([c for c in CAREER_PROFILES if c != career_role]),
            random.choice([c for c in CAREER_PROFILES if c != career_role and c != recommendations['top_matches'][1]])
        ],
        'suggested_courses': {
            'Data Analyst': [
                ("IBM Data Analyst Professional Certificate", "https://coursera.org/data-analyst-ibm"),
                ("SQL - The Complete Guide (Udemy)", "https://udemy.com/sql-complete-guide"),
                ("FreeCodeCamp Data Analysis with Python", "https://youtube.com/freecodecamp-data-analysis")
            ],
            'Software Developer': [
                ("The Complete 2024 Web Development Bootcamp (Udemy)", "https://udemy.com/web-dev-bootcamp"),
                ("Introduction to Computer Science (CS50)", "https://cs50.harvard.edu/"),
                ("Codecademy Pro - Backend Engineering Path", "https://codecademy.com/backend-path")
            ],
            'Product Manager': [
                ("Google Project Management: Professional Certificate", "https://coursera.org/project-management-google"),
                ("Product School Certifications", "https://productschool.com"),
                ("Roadmap to Product Management (YouTube)", "https://youtube.com/pm-roadmap")
            ],
            'Graphic Designer': [ # Added for consistency
                ("Graphic Design Masterclass (Udemy)", "https://udemy.com/graphic-design-masterclass"),
                ("Adobe Illustrator Essential Training (LinkedIn Learning)", "https://linkedin.com/illustrator"),
                ("The Futur - Brand Strategy Fundamentals", "https://youtube.com/thefutur")
            ],
            # Fallback for others
            'default': [
                ("General Career Success (Skillshare)", "https://skillshare.com/careers"),
                ("Effective Communication Skills (LinkedIn Learning)", "https://linkedin.com/communication")
            ]
        },
        'in_demand_skills': {
            'Data Analyst': ['SQL (Advanced)', 'Python (Pandas/NumPy)', 'Power BI/Tableau', 'Data Storytelling'],
            'Software Developer': ['One Backend Framework (Node/Django)', 'Cloud (AWS/Azure)', 'Version Control (Git)', 'Data Structures & Algorithms'],
            'Product Manager': ['Market Research', 'Agile/Scrum', 'Stakeholder Management', 'Wireframing (Figma)'],
            'UI/UX Designer': ['Figma/Sketch', 'Prototyping', 'User Research', 'Interaction Design'],
            'Financial Analyst': ['Advanced Excel Modeling', 'Valuation', 'Financial Statement Analysis', 'Python for Finance'],
            'Graphic Designer': ['Adobe Creative Suite', 'Branding Fundamentals', 'Typography', 'Visual Design'],
        },
        'career_roadmap': {
            'Data Analyst': [
                "Phase 1: Master Excel & Basic SQL (3 months)",
                "Phase 2: Learn Python for Data Science (Pandas, NumPy) (3 months)",
                "Phase 3: Deep Dive into Power BI or Tableau (2 months)",
                "Phase 4: Build a professional project portfolio (Kaggle, personal projects) (4 months)",
                "Phase 5: Apply for internships/Entry-Level DA roles (Ongoing)"
            ],
            'Software Developer': [
                "Phase 1: Solidify DSA and Core Programming (6 months)",
                "Phase 2: Choose a Stack (e.g., MERN/Python Django) and build simple apps (4 months)",
                "Phase 3: Master Git and CI/CD basics (1 month)",
                "Phase 4: Contribute to Open Source or build a complex full-stack project (4 months)",
                "Phase 5: Job search focusing on mid-size tech companies (Ongoing)"
            ]
        }
    }

    # Generate the personalized content
    skills_list = recommendations['in_demand_skills'].get(career_role, recommendations['in_demand_skills']['Data Analyst'])
    roadmap_list = recommendations['career_roadmap'].get(career_role, recommendations['career_roadmap']['Data Analyst'])
    courses = recommendations['suggested_courses'].get(career_role, recommendations['suggested_courses']['default'])
    
    # Custom LLM response structure (mocked)
    llm_response = f"""
    ### 🎯 Prediction Justification & Confidence
    The prediction for **{career_role}** is made with a high confidence score of **{confidence_score:.1f}%** because your profile strongly aligns with the necessary technical and soft skills. Your **{career_role}** aptitude score was significantly boosted by your focus on **{', '.join(st.session_state.user_data['technical_skills'][:3])}** and your strong academic background (CGPA: {st.session_state.user_data['cgpa']}).

    ### 📚 In-Demand Skills to Acquire
    To succeed in this role, focus intensely on the following:
    - **{skills_list[0]}**: This is the fundamental tool for {career_role}.
    - **{skills_list[1]}**: Essential for large-scale projects and advanced implementation.
    - **{skills_list[2]}**: Translating data/ideas into actionable insights/products.
    - **{skills_list[3]}**: Critical for team collaboration and executive presentations.

    ### 🧭 Personalized Career Roadmap
    This roadmap is designed to take you from your current stage to a mid-level professional in {career_role}:
    1. **{roadmap_list[0]}**
    2. **{roadmap_list[1]}**
    3. **{roadmap_list[2]}**
    4. **{roadmap_list[3]}**
    5. **{roadmap_list[4]}**

    ### 💡 Suggested Online Courses
    | Platform | Course Title | Link |
    | :--- | :--- | :--- |
    """
    
    for title, link in courses:
        llm_response += f"| {title.split(' ')[0]} | {title} | [Enroll Here]({link}) |\n"
        
    return llm_response, recommendations['top_matches']

def generate_radar_chart(user_skills):
    """Generates a Skill Radar Chart using Matplotlib."""
    # Select a subset of core skills
    categories = ['Python', 'SQL', 'Machine Learning', 'Cloud Computing', 'Communication', 'Leadership']
    
    # Calculate user skill score (0 or 1)
    user_data = [1 if s in user_skills else 0 for s in categories]
    
    # Calculate professional average (mock baseline)
    pro_avg = [0.7, 0.8, 0.5, 0.4, 0.9, 0.6] 
    
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    user_data += user_data[:1]
    pro_avg += pro_avg[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, user_data, linewidth=2, linestyle='solid', label='Your Profile', color='#3b82f6')
    ax.fill(angles, user_data, 'b', alpha=0.1)
    
    ax.plot(angles, pro_avg, linewidth=2, linestyle='dashed', label='Industry Average', color='#ef4444')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticks([0.2, 0.5, 0.8, 1.0])
    ax.set_yticklabels(['20%', '50%', '80%', '100%'], color="grey", size=8)
    ax.set_ylim(0, 1)
    
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title("Skill Strength Radar", size=14, y=1.1)
    return fig

# --- 4. AUTHENTICATION (MOCK) ---

def init_session_state():
    """Initializes necessary session state variables."""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'users_db' not in st.session_state:
        # Mock database of users
        st.session_state.users_db = {
            "demo": {"password": "password", "name": "Demo User"}
        }
    if 'prediction_data' not in st.session_state:
        st.session_state.prediction_data = None
    if 'user_data' not in st.session_state:
        st.session_state.user_data = None

def login_form():
    """Renders the login/signup form."""
    st.sidebar.subheader("🔒 Authentication")
    
    choice = st.sidebar.radio("Mode", ["Login", "Sign Up"], horizontal=True)

    with st.sidebar.form("auth_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        name = ""
        if choice == "Sign Up":
            name = st.text_input("Full Name")
        
        submitted = st.form_submit_button(choice)

        if submitted:
            if choice == "Login":
                if username in st.session_state.users_db and st.session_state.users_db[username]['password'] == password:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Welcome back, {st.session_state.users_db[username]['name']}!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid Username or Password")
            
            elif choice == "Sign Up":
                if username in st.session_state.users_db:
                    st.error("Username already exists.")
                elif len(username) < 3 or len(password) < 6:
                    st.error("Username must be >= 3 chars, Password >= 6 chars.")
                else:
                    st.session_state.users_db[username] = {"password": password, "name": name}
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"Account created successfully! Welcome, {name}!")
                    st.experimental_rerun()

def logout():
    """Handles user logout."""
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.prediction_data = None
    st.session_state.user_data = None
    st.info("Logged out successfully.")
    st.experimental_rerun()

# --- 5. VISUALIZATIONS ---

def create_career_comparison_chart():
    """Creates a Career Comparison Graph (Salary vs. Growth)."""
    comparison_data = [
        {'Career': c, 'Avg Salary (LPA)': CAREER_MAP[c]['base_salary'], 'Growth Factor': CAREER_MAP[c]['growth_factor']}
        for c in CAREER_PROFILES
    ]
    df_comp = pd.DataFrame(comparison_data)
    
    fig = px.scatter(
        df_comp, 
        x='Avg Salary (LPA)', 
        y='Growth Factor', 
        color='Career',
        size='Avg Salary (LPA)',
        hover_name='Career',
        title='Career Comparison: Average Salary vs. Market Growth',
        labels={'Avg Salary (LPA)': 'Average Starting Salary (LPA)', 'Growth Factor': 'Market Growth Potential (Mock Index)'},
        template='plotly_white'
    )
    fig.update_layout(height=500)
    return fig

def create_salary_distribution_plot():
    """Shows the general salary distribution for various careers."""
    fig = px.box(
        df_full, 
        x='Target Career', 
        y='Estimated Salary', 
        color='Target Career', 
        title='Simulated Salary Distribution Across Career Fields',
        labels={'Target Career': 'Career Field', 'Estimated Salary': 'Estimated Salary (LPA)'},
        template='plotly_white'
    )
    fig.update_layout(height=600, showlegend=False, xaxis_title='')
    return fig

def create_demand_trends_chart():
    """Shows mock job demand trends over time."""
    mock_demand_data = {
        'Year': [2021, 2022, 2023, 2024],
        'Data Scientist': [0.4, 0.5, 0.6, 0.7],
        'Cloud Engineer': [0.5, 0.6, 0.75, 0.9],
        'Digital Marketer': [0.8, 0.7, 0.6, 0.55],
        'Cybersecurity Analyst': [0.3, 0.5, 0.7, 0.85],
        'Web Developer': [0.7, 0.7, 0.65, 0.6]
    }
    df_demand = pd.DataFrame(mock_demand_data)
    
    fig = px.line(
        df_demand, 
        x='Year', 
        y=list(df_demand.columns[1:]), 
        title='Job Demand Trends (Mock Data)',
        labels={'value': 'Demand Index (Mock)', 'variable': 'Career Field'},
        template='plotly_white'
    )
    fig.update_traces(mode='lines+markers')
    fig.update_layout(height=500)
    return fig


# --- 6. CORE APP PAGES ---

def home_page():
    """The main input form for user data."""
    st.title("🎓 Student Career Prediction System")
    st.markdown("---")
    
    with st.expander("📚 Instructions"):
        st.markdown("""
        **Welcome!** Please fill out the form below accurately. Our system uses a machine learning model,
        trained on synthesized data, to predict your ideal career path and estimated salary.
        """)
        
    # Initialize the main form container
    with st.form(key='career_form'):
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Personal & Academic Info")
            name = st.text_input("Full Name", value=st.session_state.user_data.get('name', '') if st.session_state.user_data else "", key='name')
            age = st.number_input("Age (Years)", min_value=18, max_value=40, value=22, key='age')
            gender = st.selectbox("Gender", ["Male", "Female", "Other"], key='gender')
        
        with col2:
            st.subheader("Educational Scores")
            education = st.selectbox("Highest Education Level", EDUCATION_LEVELS, key='education')
            score_10 = st.slider("10th Grade Percentage", 50, 100, 75, key='score_10')
            score_12 = st.slider("12th Grade Percentage", 50, 100, 70, key='score_12')
            cgpa = st.slider("CGPA (out of 10)", 5.0, 10.0, 8.0, 0.1, key='cgpa')
            
        with col3:
            st.subheader("Career Preferences")
            work_type = st.radio("Preferred Work Type", ["Remote", "On-site"], key='work_type', horizontal=True)
            future_goal = st.selectbox("Future Goal", ["Job", "Higher Studies", "Startup"], key='future_goal')
            interests = st.multiselect("Primary Interests (Max 5)", SKILLS, key='interests', default=SKILLS[:2])
            
        st.markdown("---")
        
        col_skills_1, col_skills_2, col_skills_3 = st.columns(3)
        
        with col_skills_1:
            st.subheader("Technical Skills")
            tech_skills = st.multiselect("Select Technical Skills", [s for s in SKILLS if s not in ['Communication', 'Leadership', 'Problem Solving']], key='tech_skills')
        
        with col_skills_2:
            st.subheader("Soft Skills")
            soft_skills = st.multiselect("Select Soft Skills", ['Communication', 'Leadership', 'Problem Solving'], key='soft_skills')

        with col_skills_3:
            st.subheader("Upload Resume")
            uploaded_file = st.file_uploader("Upload PDF Resume", type=["pdf"])
            
            if uploaded_file:
                # Mock parsing
                extracted_skills = mock_resume_parser(uploaded_file)
                st.success(f"Extracted {len(extracted_skills)} key skills!")
                # Automatically add extracted skills to the form (if not already present)
                for skill in extracted_skills:
                    if skill in [s for s in SKILLS if s not in ['Communication', 'Leadership', 'Problem Solving']] and skill not in tech_skills:
                        tech_skills.append(skill)
                    elif skill in ['Communication', 'Leadership', 'Problem Solving'] and skill not in soft_skills:
                        soft_skills.append(skill)

        st.markdown("---")
        
        submit_button = st.form_submit_button(label='🚀 Predict My Career Path')

    if submit_button:
        # Check if basic fields are filled
        if not name or not education or cgpa < 5.0:
            st.error("Please ensure your Name, Education Level, and CGPA are filled correctly.")
            return

        # Store input data in session state
        st.session_state.user_data = {
            'name': name, 'age': age, 'gender': gender, 'education': education,
            'score_10': score_10, 'score_12': score_12, 'cgpa': cgpa,
            'work_type': work_type, 'future_goal': future_goal,
            'technical_skills': tech_skills, 'soft_skills': soft_skills,
            'interests': interests
        }

        # --- PREDICTION LOGIC ---
        with st.spinner("Analyzing profile and running AI prediction..."):
            
            # 1. Preprocess input
            input_vector = preprocess_user_data(st.session_state.user_data, feature_names)
            
            # 2. Predict Career Role (Classification)
            prediction_proba = clf.predict_proba(input_vector)[0]
            confidence_score = np.max(prediction_proba) * 100
            career_role_index = np.argmax(prediction_proba)
            career_role = clf.classes_[career_role_index]
            
            # 3. Predict Salary (Regression)
            estimated_salary_lpa = reg.predict(input_vector)[0]
            estimated_salary_lpa = max(3.0, round(estimated_salary_lpa, 2)) # Ensure min salary cap
            
            # 4. Generate AI Recommendations
            ai_recommendation_text, top_matches = generate_ai_recommendations(career_role, confidence_score)
            
            # 5. Store results
            st.session_state.prediction_data = {
                'career_role': career_role,
                'salary_lpa': estimated_salary_lpa,
                'confidence': confidence_score,
                'ai_text': ai_recommendation_text,
                'top_matches': top_matches,
                'all_skills': tech_skills + soft_skills
            }
        
        st.success(f"Prediction Complete! Your ideal career is **{career_role}**.")
        # Automatically switch to results page
        st.session_state.current_page = "Prediction Results"
        st.experimental_rerun()


def results_page():
    """Displays the prediction results, graphs, and recommendations."""
    if not st.session_state.prediction_data:
        st.warning("Please submit your profile in the 'Home' tab first to see results.")
        st.session_state.current_page = "Home"
        return

    data = st.session_state.prediction_data
    
    st.title(f"🎉 Your Personalized Career Report")
    st.subheader(f"Results for {st.session_state.user_data['name']}")
    st.markdown("---")
    
    # 1. Core Metrics Display
    st.markdown(f"### Predicted Career: **{data['career_role']}** 🎓")
    
    col_metric_1, col_metric_2, col_metric_3 = st.columns(3)
    
    with col_metric_1:
        st.metric(
            label="Estimated Salary Package",
            value=f"₹ {data['salary_lpa']} LPA",
            delta="15% Above Avg (Simulated)",
            delta_color="normal"
        )
    
    with col_metric_2:
        st.metric(
            label="Prediction Confidence Score",
            value=f"{data['confidence']:.1f} %",
            delta_color="off"
        )
        st.progress(data['confidence'] / 100)

    with col_metric_3:
        st.metric(
            label="Top Alternative Match",
            value=data['top_matches'][1],
            delta="High Growth Potential"
        )
        
    st.markdown("---")
    
    # 2. Visualizations
    st.subheader("📊 Profile Visualization")
    
    viz_col_1, viz_col_2 = st.columns([1, 2])
    
    with viz_col_1:
        st.pyplot(generate_radar_chart(data['all_skills']))
        
    with viz_col_2:
        st.plotly_chart(create_career_comparison_chart(), use_container_width=True)

    st.markdown("---")

    # 3. AI Recommendations
    st.subheader("🤖 AI-Powered Career Roadmap & Recommendations")
    
    # Use the generated LLM text
    st.markdown(data['ai_text'], unsafe_allow_html=True)
    
    # 4. Action Buttons
    st.markdown("---")
    if st.button("⬇️ Download Full Report (CSV Mock)", key="download_report_btn"):
        mock_report = pd.DataFrame({
            'Metric': ['Name', 'Predicted Career', 'Estimated Salary (LPA)', 'Confidence (%)'],
            'Value': [st.session_state.user_data['name'], data['career_role'], data['salary_lpa'], round(data['confidence'], 1)]
        })
        st.download_button(
            label="Download Data",
            data=mock_report.to_csv().encode('utf-8'),
            file_name=f"{st.session_state.user_data['name']}_career_report.csv",
            mime='text/csv',
            key='csv_download_btn'
        )
        st.success("Report generation mock complete.")


def insights_page():
    """Displays data-driven insights and plots."""
    st.title("💡 Career Market Data Insights")
    st.markdown("Explore trends and distributions across different career fields.")
    st.markdown("---")
    
    st.subheader("💰 Salary Distribution Plot")
    st.plotly_chart(create_salary_distribution_plot(), use_container_width=True)
    
    st.subheader("📈 Job Demand Trends")
    st.plotly_chart(create_demand_trends_chart(), use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Data Insights Tab")
    st.info("""
    This tab showcases the underlying data distribution used by the model. 
    A higher salary spread (taller box plot) indicates greater variability in earnings, often based on factors like company size and location.
    The mock demand trends illustrate how specific technical roles (like Cloud Engineer) have seen significant growth.
    """)

def chatbot_page():
    """AI Career Chatbot Assistant (Mock LLM Chat)."""
    st.title("💬 AI Career Assistant Chatbot")
    st.markdown("Ask me anything about career paths, learning resources, or salaries!")
    st.markdown("---")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hello! I am your AI Career Assistant. How can I help you today?"}]

    # Display chat messages
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Mock LLM response generator
    def mock_llm_chat(prompt):
        prompt_lower = prompt.lower()
        if "salary" in prompt_lower:
            return "Salaries for Data Scientists in India typically range from 8 LPA for freshers up to 35+ LPA for senior roles, heavily dependent on skills like Deep Learning and Cloud experience."
        elif "course" in prompt_lower or "learn" in prompt_lower:
            return "For Software Development, I highly recommend starting with Harvard's free CS50 course to build a strong foundation in computer science and problem-solving."
        elif "roadmap" in prompt_lower:
            return "A good career roadmap involves: 1. Skill Acquisition, 2. Portfolio Building, 3. Networking, and 4. Targeted Applications. Focus on one phase at a time."
        elif "thank" in prompt_lower:
            return "You're very welcome! Let me know if you have any other questions."
        else:
            return "That's a great question. Based on current trends, having a strong portfolio with 3-5 real-world projects is the single most important factor for securing a tech job."

    # User input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        # Generate and display assistant response
        with st.spinner("Thinking..."):
            response = mock_llm_chat(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)


def about_page():
    """About the app and developer section."""
    st.title("ℹ️ About the App & Developer")
    st.markdown("---")
    
    st.subheader("🔬 How The Model Works")
    st.markdown("""
    This **Student Career Prediction and Recommendation System** utilizes a two-model approach:
    
    1.  **Career Prediction (Random Forest Classifier):** Predicts the best-fit career role based on your academic scores, education level, and technical/soft skills. Feature importance heavily weighs **CGPA** and **Relevant Technical Skills (e.g., Python, SQL)**.
    2.  **Salary Estimation (Linear Regression):** Estimates your potential starting salary based on the predicted role's base salary, your CGPA, and the number of highly-demanded skills you possess.
    3.  **AI Recommendation:** The personalized advice, roadmap, and course links are generated by a simulated Large Language Model (LLM) that structures its output based on the primary predicted career.
    
    **Note:** The models are trained on a comprehensive synthetic dataset of 1000 student profiles, making the prediction logic robust for demonstration purposes.
    """)
    
    st.subheader("👨‍💻 Developed by Suraj Kumar (Mock)")
    st.markdown("""
    **Developer:** Suraj Kumar
    
    **Title:** BCA Student | Data Analyst Enthusiast
    
    This project was built using Python, Streamlit, and scikit-learn as a demonstration of applying machine learning and AI concepts to solve real-world career planning challenges.
    
    ---
    
    **Connect with me:**
    
    * **LinkedIn:** [linkedin.com/in/surajkumar](https://www.linkedin.com/in/surajkumar) (Mock Link)
    * **GitHub:** [github.com/surajkumar-dev](https://github.com/surajkumar-dev) (Mock Link)
    """)

# --- 7. MAIN APP EXECUTION ---

def main_app():
    """Controls the page flow and sidebar navigation."""
    
    init_session_state()

    # --- Sidebar Content ---
    
    if st.session_state.authenticated:
        st.sidebar.markdown(f"**Welcome, {st.session_state.users_db[st.session_state.username]['name']}**")
        if st.sidebar.button("Logout", on_click=logout):
            pass # Logout handled by the function
            
        # Add page navigation after successful login
        st.sidebar.markdown("---")
        
        pages = {
            "Home (Input Form)": home_page,
            "Prediction Results": results_page,
            "Career Market Insights": insights_page,
            "AI Career Chatbot": chatbot_page,
            "About App & Developer": about_page
        }
        
        # Set default page on first run or if page state is missing
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "Home (Input Form)"
            
        # Handle page change from the sidebar selectbox
        selected_page_name = st.sidebar.radio(
            "Navigation", 
            list(pages.keys()), 
            index=list(pages.keys()).index(st.session_state.current_page)
        )
        
        # Update current page state if changed via sidebar
        if selected_page_name != st.session_state.current_page:
             st.session_state.current_page = selected_page_name
             
        # Call the selected page function
        pages[st.session_state.current_page]()
        
    else:
        # Show login/signup if not authenticated
        st.sidebar.title("Student Career System")
        login_form()
        
    # --- Always visible content ---
    st.sidebar.markdown("""
    ---
    🌐 **Tech Stack:** Python, Streamlit, scikit-learn, Pandas, Plotly.
    """)
    

if __name__ == '__main__':
    main_app()
