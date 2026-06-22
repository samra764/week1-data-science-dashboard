import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    
    page_title="Titanic Survival Analysis",
    page_icon="🚢",
    layout="wide"
)
import base64

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("titanic_bg.jpg")

st.markdown(f"""
<style>

/* ── CINEMATIC BACKGROUND IMAGE ── */
.stApp {{
    background-image: 
        linear-gradient(
            rgba(0, 0, 0, 0.55),
            rgba(0, 5, 20, 0.70)
        ),
        url("data:image/jpg;base64,{img_base64}");
    background-size: cover;
    background-position: center top;
    background-attachment: fixed;
    background-repeat: no-repeat;
}}

/* ── headings ── */
h1, h2, h3, h4, h5, h6 {{
    color: #e8f4f8 !important;
    text-shadow: 2px 2px 12px rgba(0,0,0,0.9);
}}

p, li {{
    color: #c8dfe8 !important;
}}

/* ── sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(
        180deg,
        rgba(0, 8, 25, 0.92) 0%,
        rgba(0, 15, 40, 0.95) 100%
    ) !important;
    border-right: 1px solid rgba(100, 180, 220, 0.25) !important;
}}

/* ── metric cards — glass effect ── */
[data-testid="metric-container"] {{
    background: rgba(0, 15, 40, 0.65) !important;
    border: 1px solid rgba(100, 180, 220, 0.35) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}}

[data-testid="metric-container"] label {{
    color: #7eb8d4 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.8rem !important;
}}

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: rgba(0, 15, 40, 0.60) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(100, 180, 220, 0.20) !important;
}}

.stTabs [data-baseweb="tab"] {{
    color: #7eb8d4 !important;
    font-weight: 600 !important;
}}

.stTabs [aria-selected="true"] {{
    background: rgba(100, 180, 220, 0.25) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}}

/* ── chart containers — frosted glass ── */
[data-testid="stImage"],
.element-container {{
    background: rgba(0, 10, 30, 0.50) !important;
    border-radius: 12px !important;
    border: 1px solid rgba(100, 180, 220, 0.15) !important;
    backdrop-filter: blur(8px) !important;
}}

/* ── dataframe ── */
[data-testid="stDataFrame"] {{
    background: rgba(0, 15, 40, 0.60) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(100, 180, 220, 0.20) !important;
}}

/* ── divider ── */
hr {{
    border-color: rgba(100, 180, 220, 0.25) !important;
}}

/* ── scrollbar ── */
::-webkit-scrollbar {{
    width: 6px;
}}
::-webkit-scrollbar-track {{
    background: rgba(0, 10, 30, 0.5);
}}
::-webkit-scrollbar-thumb {{
    background: rgba(100, 180, 220, 0.4);
    border-radius: 3px;
}}

</style>
""", unsafe_allow_html=True)
@st.cache_data
def load_data():
    # ── paste your Cell 2 code here ──
    df = pd.read_csv("dataset.csv")

    # ── paste your Cell 3 cleaning code here ──
    df['Age'] = df.groupby(['Pclass','Sex'])['Age'].transform(
        lambda x: x.fillna(x.median())
    )
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df.drop(columns=['Cabin','Name','Ticket'], inplace=True)

    # ── paste your Cell 5 feature engineering here ──
    df['AgeGroup'] = pd.cut(df['Age'],
                             bins=[0,12,18,35,60,100],
                             labels=['Child','Teen','Young Adult','Middle Age','Senior'])
    df['FamilySize']    = df['SibSp'] + df['Parch'] + 1
    df['IsAlone']       = (df['FamilySize'] == 1).astype(int)
    df['SurvivedLabel'] = df['Survived'].map({1:'Survived', 0:'Perished'})
    df['ClassLabel']    = df['Pclass'].map({1:'1st Class',
                                            2:'2nd Class',
                                            3:'3rd Class'})
    return df

df = load_data()

st.title("🚢 RMS Titanic — Survival Analysis Dashboard")
st.markdown("Exploring survival patterns using real passenger data.")
with st.sidebar:
    st.header("🎛️ Filters")

    sex_filter = st.multiselect(
        "Gender",
        options=df['Sex'].unique(),
        default=df['Sex'].unique()
    )
    class_filter = st.multiselect(
        "Passenger Class",
        options=sorted(df['Pclass'].unique()),
        default=sorted(df['Pclass'].unique())
    )
    age_range = st.slider(
        "Age Range",
        min_value=int(df['Age'].min()),
        max_value=int(df['Age'].max()),
        value=(int(df['Age'].min()), int(df['Age'].max()))
    )

mask = (
    df['Sex'].isin(sex_filter) &
    df['Pclass'].isin(class_filter) &
    df['Age'].between(age_range[0], age_range[1])
)
dff = df[mask]
st.markdown("---")

total     = len(dff)
survived  = dff['Survived'].sum()
perished  = total - survived
surv_rate = (survived / total * 100) if total > 0 else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("👥 Total Passengers", f"{total}")
col2.metric("✅ Survived",         f"{survived}")
col3.metric("❌ Perished",         f"{perished}")
col4.metric("📈 Survival Rate",    f"{surv_rate:.1f}%")

st.markdown("---")
tab1, tab2, tab3, tab4 = st.tabs([
    "📋 Dataset Overview",
    "📊 Survival Analysis",
    "👥 Demographics",
    "💡 Insights"
])
with tab1:
    st.header("Dataset Overview")

    st.subheader("Raw Data")
    st.dataframe(dff.head(10), use_container_width=True)

    st.subheader("Statistical Summary")
    st.write(dff.describe())
with tab2:
    st.header("Survival Analysis")

    # ── Row 1: Pie + Bar by Class (side by side) ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Overall Survival")
        fig, ax = plt.subplots(figsize=(4, 3))   # ← smaller size
        counts = dff['SurvivedLabel'].value_counts()
        ax.pie(counts,
               labels=counts.index,
               autopct='%1.1f%%',
               colors=['#2ecc71', '#e74c3c'],
               textprops={'fontsize': 10})        # ← smaller text
        ax.set_title("Survived vs Perished", fontsize=11)
        plt.tight_layout()                        # ← fits neatly
        st.pyplot(fig, use_container_width=True)  # ← fills column width
        plt.close()

    with col2:
        st.subheader("Survival by Class")
        fig, ax = plt.subplots(figsize=(4, 3))   # ← same size as pie
        surv_by_class = dff.groupby('ClassLabel')['Survived'].mean() * 100
        ax.bar(surv_by_class.index,
               surv_by_class.values,
               color=['#3498db', '#f39c12', '#e74c3c'],
               width=0.5)
        ax.set_ylabel("Survival Rate (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title("Survival Rate by Class", fontsize=11)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Row 2: Gender bar chart (contained, not full width) ──
    st.subheader("Survival by Gender")
    col3, col4 = st.columns(2)        # ← put gender chart in a column

    with col3:                         # ← only takes half the page
        fig, ax = plt.subplots(figsize=(4, 3))   # ← same small size
        surv_gender = dff.groupby('Sex')['Survived'].mean() * 100
        ax.bar(['Female', 'Male'],
               surv_gender.values,
               color=['#e91e8c', '#3498db'],
               width=0.4)
        ax.set_ylabel("Survival Rate (%)", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_title("Survival Rate by Gender", fontsize=11)
        ax.tick_params(labelsize=9)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with col4:                         # ← empty column keeps layout neat
        st.empty()
with tab3:
    st.header("Passenger Demographics")

    col1, col2 = st.columns(2)

    # ── take your Cell 9 histogram ──
    with col1:
        st.subheader("Age Distribution")
        fig, ax = plt.subplots(figsize=(5,4))
        ax.hist(dff[dff['Survived']==1]['Age'].dropna(),
                bins=20, alpha=0.7, color='#2ecc71', label='Survived')
        ax.hist(dff[dff['Survived']==0]['Age'].dropna(),
                bins=20, alpha=0.7, color='#e74c3c', label='Perished')
        ax.set_xlabel("Age")
        ax.set_ylabel("Count")
        ax.legend()
        ax.set_title("Age Distribution by Survival")
        st.pyplot(fig)      # ← was plt.show() in Jupyter
        plt.close()         # ← add this line always

    # ── take your Cell 10 scatter plot ──
    with col2:
        st.subheader("Age vs Fare")
        fig, ax = plt.subplots(figsize=(5,4))
        ax.scatter(dff[dff['Survived']==0]['Age'],
                   dff[dff['Survived']==0]['Fare'],
                   c='#e74c3c', alpha=0.5, label='Perished', s=30)
        ax.scatter(dff[dff['Survived']==1]['Age'],
                   dff[dff['Survived']==1]['Fare'],
                   c='#2ecc71', alpha=0.5, label='Survived', s=30)
        ax.set_xlabel("Age")
        ax.set_ylabel("Fare (£)")
        ax.legend()
        ax.set_title("Age vs Fare by Survival")
        st.pyplot(fig)      # ← was plt.show() in Jupyter
        plt.close()   
with tab4:
    st.header("💡 Key Insights")

    f_surv  = dff[dff['Sex']=='female']['Survived'].mean() * 100
    m_surv  = dff[dff['Sex']=='male']['Survived'].mean() * 100
    c1_surv = dff[dff['Pclass']==1]['Survived'].mean() * 100
    c3_surv = dff[dff['Pclass']==3]['Survived'].mean() * 100

    st.markdown(f"""
    - 📌 **Women First:** Female survival was **{f_surv:.0f}%**
      vs only **{m_surv:.0f}%** for males.

    - 📌 **Class Privilege:** 1st class survived at **{c1_surv:.0f}%**
      vs only **{c3_surv:.0f}%** in 3rd class.

    - 📌 **Children Prioritized:** Children under 12 had relatively
      higher survival — the crew enforced women and children first.

    - 📌 **Fare Reflects Access:** Higher fare passengers were on
      upper decks closer to lifeboats — survival was higher.
    """)      # ← add this line always