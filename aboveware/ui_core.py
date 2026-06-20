import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# Page Config
# -----------------------
st.set_page_config(
    page_title="AI Information Extractor",
    page_icon="🤖",
    layout="wide"
)

# -----------------------
# Custom CSS
# -----------------------
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}

.stTextArea textarea {
    font-size: 16px;
}

.result-box {
    background-color: #262730;
    padding: 20px;
    border-radius: 10px;
    border: 1px solid #4A4A4A;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------
# Load Environment
# -----------------------
load_dotenv()

# -----------------------
# Model
# -----------------------
@st.cache_resource
def load_model():
    return ChatMistralAI(
        model="mistral-small-latest"
    )

model = load_model()

# -----------------------
# Prompt
# -----------------------
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an information extraction system.

Analyze the content and identify:

1. What is the primary subject?
2. What type of content is this?
3. Who are the important people?
4. What are the important organizations?
5. What are the important locations?
6. What are the major events?
7. What factual claims are made?
8. What statistics or numbers are mentioned?
9. What future developments are mentioned?
10. What are the key takeaways?

Then generate:

### Executive Summary
(5-10 sentences)

### Structured Information
(Organized under meaningful headings)
"""
    ),
    (
        "human",
        """
Extract the information from the following content:

{paragraph}
"""
    )
])

# -----------------------
# Header
# -----------------------
st.markdown(
    '<div class="title">🤖 AI Information Extractor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Powered by LangChain + Mistral AI</div>',
    unsafe_allow_html=True
)

# -----------------------
# Input
# -----------------------
paragraph = st.text_area(
    "Paste your content below",
    height=250,
    placeholder="Paste article, report, blog post, news content..."
)

col1, col2 = st.columns([1, 4])

with col1:
    extract_btn = st.button(
        "🚀 Extract",
        use_container_width=True
    )

# -----------------------
# Processing
# -----------------------
if extract_btn:

    if not paragraph.strip():
        st.warning("Please enter some content.")
        st.stop()

    with st.spinner("Analyzing Content..."):

        final_prompt = prompt.invoke(
            {"paragraph": paragraph}
        )

        response = model.invoke(final_prompt)

    st.success("Analysis Complete!")

    st.markdown("---")

    st.subheader("📑 Extracted Insights")

    st.markdown(response.content)

    st.download_button(
        label="📥 Download Report",
        data=response.content,
        file_name="analysis_report.txt",
        mime="text/plain"
    ) 