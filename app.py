from __future__ import annotations

import streamlit as st

from src.personalization import PersonalizationEngine
from src.rag_pipeline import LearningAssistant


st.set_page_config(
    page_title="AI Doubt Solving + Personalized Learning",
    page_icon="AI",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_assistant() -> LearningAssistant:
    return LearningAssistant()


@st.cache_resource(show_spinner=False)
def load_personalization() -> PersonalizationEngine:
    return PersonalizationEngine()


assistant = load_assistant()
personalization = load_personalization()

st.title("AI Doubt Solving + Personalized Learning System")
st.caption(
    "Ask a question, retrieve relevant study material, and get topic-level recommendations."
)

with st.sidebar:
    st.subheader("System")
    st.write(f"Embedding model: `{assistant.kb.model_name}`")
    st.write(f"Knowledge chunks: `{assistant.kb.document_count}`")
    st.write(
        "Answer mode: "
        + ("`OpenAI + Retrieval`" if assistant.generator.uses_openai else "`Retrieval Fallback`")
    )

question = st.text_area(
    "Enter your doubt",
    placeholder="Example: Why does photosynthesis need chlorophyll?",
    height=120,
)

top_k = st.slider("Retrieved context chunks", min_value=2, max_value=5, value=3)

if st.button("Solve Doubt", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            result = assistant.answer_question(question=question, top_k=top_k)
            personalization.log_interaction(result)

        left, right = st.columns([2, 1])

        with left:
            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Retrieved Context")
            for idx, item in enumerate(result["context"], start=1):
                with st.expander(f"Chunk {idx}: {item['title']}"):
                    st.write(f"Topic: `{item['topic']}`")
                    st.write(item["content"])
                    st.write(f"Similarity score: `{item['score']:.3f}`")

        with right:
            st.subheader("Question Analysis")
            st.metric("Detected topic", result["topic"])
            st.metric("Confidence", f"{result['confidence']:.2f}")
            st.metric("Sources used", len(result["context"]))

            st.subheader("Suggested Quiz")
            for prompt in result["quiz_questions"]:
                st.write(f"- {prompt}")

st.divider()

st.header("Personalized Learning Insights")
history_df = personalization.load_history()

if history_df.empty:
    st.info("No interactions yet. Ask a question to generate recommendations.")
else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Weak Areas")
        weak_areas = personalization.identify_weak_areas(history_df)
        if weak_areas:
            for item in weak_areas:
                st.write(
                    f"- `{item['topic']}`: {item['count']} questions, average confidence `{item['avg_confidence']:.2f}`"
                )
        else:
            st.write("No weak areas identified yet.")

    with col2:
        st.subheader("Study Plan")
        for line in personalization.build_study_plan(history_df):
            st.write(f"- {line}")

    st.subheader("Recent Question History")
    st.dataframe(history_df.sort_values("timestamp", ascending=False), use_container_width=True)
