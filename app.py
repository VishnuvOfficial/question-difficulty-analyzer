import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================
# LOAD SAVED MODEL FILES
# ==========================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))
le = pickle.load(open("label_encoder.pkl", "rb"))

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Question Difficulty Predictor",
    page_icon="📘",
    layout="wide"
)

# ==========================
# HEADER
# ==========================
st.markdown(
    """
    <h1 style='text-align:center;'>📘 Question Difficulty Predictor</h1>
    <p style='text-align:center;font-size:18px;'>
    AI-based NLP system to classify exam questions into Easy, Medium, or Hard
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# ==========================
# MAIN LAYOUT
# ==========================
col1, col2 = st.columns([2, 1])

# ==========================
# INPUT + PREDICTION
# ==========================
with col1:
    question = st.text_area(
        "✍️ Enter your question",
        height=180,
        placeholder="Example: Explain Bayes theorem with an example"
    )

    if st.button("🔍 Predict Difficulty"):
        if question.strip() == "":
            st.warning("Please enter a question")
        else:
            q_vec = tfidf.transform([question])
            pred = model.predict(q_vec)
            result = le.inverse_transform(pred)[0]

            if result.lower() == "easy":
                st.success(f"🟢 Difficulty Level: {result}")
            elif result.lower() == "medium":
                st.warning(f"🟡 Difficulty Level: {result}")
            else:
                st.error(f"🔴 Difficulty Level: {result}")

# ==========================
# PROJECT DETAILS
# ==========================
with col2:
    st.subheader("📊 Project Details")
    st.info("""
    **Domain:** NLP & Machine Learning  
    **Algorithm:** Naive Bayes  
    **Feature Extraction:** TF-IDF  
    **Classes:** Easy / Medium / Hard  
    """)

    st.subheader("📝 Sample Questions")
    st.code("""
    • What is Python?
    • Explain Bayes theorem.
    • Design a distributed system architecture.
    """)

st.divider()

# ==========================
# MODEL PERFORMANCE
# ==========================
st.subheader("📈 Model Performance")

# Replace with your real accuracy value
accuracy_value = 85
st.metric("Accuracy", f"{accuracy_value}%")

# ==========================
# DATASET VISUALIZATIONS
# ==========================
st.subheader(f"📊 Dataset Analysis (5000 rows)")

# Load dataset
df = pd.read_csv("question_difficulty_dataset_5000.csv")
df = df.dropna().reset_index(drop=True)  # ensure no missing rows

# Add question length column safely
df["length"] = df["question_text"].apply(lambda x: len(str(x)))

# Create two columns for graphs
col3, col4 = st.columns(2)

with col3:
    st.subheader("Difficulty Distribution")
    fig1, ax1 = plt.subplots()
    sns.countplot(x="difficulty_level", data=df, ax=ax1)
    ax1.set_xlabel("Difficulty Level")
    ax1.set_ylabel("Number of Questions")
    st.pyplot(fig1)

with col4:
    st.subheader("Question Length Analysis")
    fig2, ax2 = plt.subplots()
    sns.boxplot(x="difficulty_level", y="length", data=df, ax=ax2)
    ax2.set_xlabel("Difficulty Level")
    ax2.set_ylabel("Question Length")
    st.pyplot(fig2)

st.divider()

# ==========================
# FOOTER
# ==========================
st.success("✅ Major Project – Ready for Demo, Viva & Resume")
