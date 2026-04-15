import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from llm_config import render_llm_selector, build_llm
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MailMind — AI Email Assistant",
    page_icon="✉️",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.block-container { max-width: 960px; padding: 2rem 2rem 5rem; }
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #0D0D14;
  border-right: 1px solid #1C1C2E;
}
div[data-testid="stSidebarContent"] { padding: 1.5rem 1rem; }

.brand {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 4px;
}
.brand-icon {
  width: 36px; height: 36px;
  background: linear-gradient(135deg, #6366F1, #A855F7);
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 18px;
}
.brand-name { font-size: 1.2rem; font-weight: 700; color: #F1F5F9; }
.brand-sub { font-size: 0.75rem; color: #4B5563; margin-bottom: 1.2rem; }

.section-label {
  font-size: 0.65rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase;
  color: #374151; margin: 1rem 0 0.4rem;
}

/* Example pills */
.example-pill {
  background: #13131A;
  border: 1px solid #1C1C2E;
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 0.78rem;
  color: #6B7280;
  margin-bottom: 6px;
  cursor: pointer;
  line-height: 1.4;
}
.example-pill:hover { border-color: #6366F1; color: #A5B4FC; }

/* Main header */
.page-header {
  display: flex; align-items: center; gap: 12px;
  margin-bottom: 1.5rem;
}
.page-title { font-size: 1.5rem; font-weight: 700; color: #F1F5F9; margin: 0; }
.page-sub { font-size: 0.85rem; color: #4B5563; margin: 2px 0 0; }

/* Email input area */
.email-box-label {
  font-size: 0.72rem; font-weight: 600;
  letter-spacing: 0.06em; text-transform: uppercase;
  color: #374151; margin-bottom: 6px;
}

/* Classification card */
.class-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-bottom: 1.2rem;
}
.class-card {
  background: #13131A;
  border: 1px solid #1C1C2E;
  border-radius: 10px;
  padding: 14px 12px;
  text-align: center;
}
.class-card-icon { font-size: 1.3rem; margin-bottom: 6px; }
.class-card-label {
  font-size: 0.65rem; color: #4B5563;
  text-transform: uppercase; letter-spacing: 0.06em;
  margin-bottom: 4px;
}
.class-card-value { font-size: 0.9rem; font-weight: 600; color: #E2E8F0; }

/* Priority badge */
.badge {
  display: inline-block;
  border-radius: 20px;
  padding: 3px 10px;
  font-size: 0.72rem;
  font-weight: 600;
}
.badge-high { background: rgba(239,68,68,0.15); color: #F87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-medium { background: rgba(251,191,36,0.12); color: #FCD34D; border: 1px solid rgba(251,191,36,0.25); }
.badge-low { background: rgba(16,185,129,0.12); color: #34D399; border: 1px solid rgba(16,185,129,0.25); }

/* Draft card */
.draft-wrapper {
  background: #13131A;
  border: 1px solid #1C1C2E;
  border-radius: 12px;
  padding: 1.2rem 1.4rem;
  margin-top: 0.5rem;
}
.draft-header {
  display: flex; justify-content: space-between; align-items: center;
  margin-bottom: 1rem;
}
.draft-title { font-size: 0.78rem; font-weight: 600; color: #6366F1; text-transform: uppercase; letter-spacing: 0.06em; }
.draft-meta { font-size: 0.72rem; color: #374151; }

/* Reasoning box */
.reasoning-box {
  background: #0D0D14;
  border: 1px solid #1C1C2E;
  border-left: 3px solid #6366F1;
  border-radius: 6px;
  padding: 10px 14px;
  font-size: 0.8rem;
  color: #6B7280;
  line-height: 1.6;
  font-style: italic;
}

/* Textarea override */
textarea {
  background: #13131A !important;
  border: 1px solid #1C1C2E !important;
  border-radius: 10px !important;
  color: #E2E8F0 !important;
  font-size: 0.87rem !important;
  line-height: 1.6 !important;
}
textarea:focus {
  border-color: #6366F1 !important;
  box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
}

/* Buttons */
.stButton > button {
  border-radius: 8px !important;
  font-weight: 500 !important;
  font-size: 0.85rem !important;
  transition: all 0.15s !important;
}
</style>
""", unsafe_allow_html=True)

# ── Schema ────────────────────────────────────────────────────────────────────
class EmailAnalysis(BaseModel):
    category: Literal["Support", "Sales", "Complaint", "Inquiry", "Spam", "Internal", "Other"]
    priority: Literal["High", "Medium", "Low"]
    sentiment: Literal["Positive", "Neutral", "Negative", "Frustrated", "Urgent"]
    summary: str = Field(description="One sentence summary of the email")
    key_points: list[str] = Field(description="2-4 key points or requests from the email")
    suggested_subject: str = Field(description="Subject line for the reply")
    reasoning: str = Field(description="Brief explanation of classification and tone choices")

CATEGORY_ICONS = {
    "Support": "🛠️", "Sales": "💼", "Complaint": "⚠️",
    "Inquiry": "❓", "Spam": "🚫", "Internal": "🏢", "Other": "📬",
}

EXAMPLES = [
    ("Angry customer", "Hi, I've been waiting 3 weeks for my order #84921 and still nothing. I want a full refund immediately. This is completely unacceptable and I've already told my friends to avoid your company."),
    ("Sales inquiry", "Hello, we're a 50-person company evaluating AI tools for our support team. Could you share pricing and whether you offer enterprise contracts? We'd like to schedule a demo this week if possible."),
    ("Simple support", "Hey, I can't seem to reset my password. I've tried the forgot password link but the email isn't arriving. Can you help?"),
    ("Partnership", "Hi there, I'm the head of partnerships at TechCorp. We think there's a strong fit between our platforms and would love to explore a co-marketing arrangement. Are you open to a call?"),
]

# ── LLM chains ────────────────────────────────────────────────────────────────
def get_analysis_chain(provider: str, model: str, api_key: str | None):
    llm = build_llm(provider, model, api_key, temperature=0.1)
    parser = JsonOutputParser(pydantic_object=EmailAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert email analyst for a business. Analyze the email and return a JSON object "
         "matching this schema exactly:\n{format_instructions}\n\n"
         "Be precise with priority: High = needs response within hours, Medium = within a day, Low = within a week."),
        ("human", "Analyze this email:\n\n{email}"),
    ])

    return (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | llm
        | parser
    )


def get_draft_chain(provider: str, model: str, api_key: str | None):
    llm = build_llm(provider, model, api_key, temperature=0.6, streaming=True)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a professional email writer. Write a {tone} reply to the email below.\n"
         "Guidelines:\n"
         "- Address all key points: {key_points}\n"
         "- Match the appropriate urgency for a {priority} priority email\n"
         "- Be concise but thorough\n"
         "- End with a clear next step\n"
         "- Do NOT include a subject line in the body\n"
         "- Sign off as 'The Support Team'"),
        ("human", "Original email:\n\n{email}"),
    ])
    return prompt | llm | StrOutputParser()


TONE_MAP = {
    "Complaint": "empathetic and solution-focused",
    "Support":   "helpful and clear",
    "Sales":     "professional and value-focused",
    "Inquiry":   "informative and engaging",
    "Internal":  "direct and collegial",
    "Other":     "professional and friendly",
    "Spam":      "brief and dismissive",
}

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("analysis", None), ("draft", None), ("email_text", ""), ("last_processed", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="brand">
      <div class="brand-icon">✉️</div>
      <span class="brand-name">MailMind</span>
    </div>
    <div class="brand-sub">AI-powered email intelligence</div>
    """, unsafe_allow_html=True)

    provider, model, api_key, is_configured = render_llm_selector(default_temp=0.3)

    st.markdown('<div class="section-label">Try an example</div>', unsafe_allow_html=True)
    for label, body in EXAMPLES:
        if st.button(label, use_container_width=True, key=f"ex_{label}"):
            st.session_state.email_text = body
            st.session_state.analysis = None
            st.session_state.draft = None
            st.rerun()

    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.75rem;color:#374151;line-height:1.6;">
    Classifies emails by category, priority, and sentiment — then drafts a professional reply tailored to the situation.
    </div>
    """, unsafe_allow_html=True)

# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
  <div>
    <div class="page-title">Email Intelligence</div>
    <div class="page-sub">Classify, prioritize, and draft replies — instantly</div>
  </div>
</div>
""", unsafe_allow_html=True)

if not is_configured:
    st.info("Choose a provider and enter your API key (or select Local Ollama) in the sidebar to get started.")
    st.stop()

# ── Email input ───────────────────────────────────────────────────────────────
col_input, col_controls = st.columns([3, 1])

with col_input:
    st.markdown('<div class="email-box-label">Email content</div>', unsafe_allow_html=True)
    email_text = st.text_area(
        "email", label_visibility="collapsed",
        value=st.session_state.email_text,
        height=200,
        placeholder="Paste an email here, or pick an example from the sidebar…",
        key="email_input",
    )

with col_controls:
    st.write("")
    st.write("")
    analyze_clicked = st.button("Analyze →", type="primary", use_container_width=True)
    if st.session_state.analysis:
        if st.button("Clear", use_container_width=True):
            st.session_state.analysis = None
            st.session_state.draft = None
            st.session_state.email_text = ""
            st.rerun()

# ── Run analysis ──────────────────────────────────────────────────────────────
if analyze_clicked and email_text.strip():
    st.session_state.email_text = email_text
    with st.spinner("Analyzing…"):
        try:
            chain = get_analysis_chain(provider, model, api_key)
            result = chain.invoke({"email": email_text})
            st.session_state.analysis = result
            st.session_state.draft = None
            st.session_state.last_processed = email_text
        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.stop()

# ── Show results ──────────────────────────────────────────────────────────────
if st.session_state.analysis:
    a = st.session_state.analysis
    # Handle both dict and Pydantic object
    if isinstance(a, dict):
        category = a.get("category", "Other")
        priority = a.get("priority", "Medium")
        sentiment = a.get("sentiment", "Neutral")
        summary = a.get("summary", "")
        key_points = a.get("key_points", [])
        suggested_subject = a.get("suggested_subject", "")
        reasoning = a.get("reasoning", "")
    else:
        category = a.category
        priority = a.priority
        sentiment = a.sentiment
        summary = a.summary
        key_points = a.key_points
        suggested_subject = a.suggested_subject
        reasoning = a.reasoning

    priority_class = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(priority, "badge-medium")
    cat_icon = CATEGORY_ICONS.get(category, "📬")

    st.divider()

    # Classification grid
    st.markdown(
        f'<div class="class-grid">'
        f'<div class="class-card"><div class="class-card-icon">{cat_icon}</div>'
        f'<div class="class-card-label">Category</div><div class="class-card-value">{category}</div></div>'
        f'<div class="class-card"><div class="class-card-icon">🎯</div>'
        f'<div class="class-card-label">Priority</div>'
        f'<div class="class-card-value"><span class="badge {priority_class}">{priority}</span></div></div>'
        f'<div class="class-card"><div class="class-card-icon">💬</div>'
        f'<div class="class-card-label">Sentiment</div><div class="class-card-value">{sentiment}</div></div>'
        f'<div class="class-card"><div class="class-card-icon">⏱️</div>'
        f'<div class="class-card-label">Respond by</div>'
        f'<div class="class-card-value">{"2 hrs" if priority == "High" else "24 hrs" if priority == "Medium" else "1 week"}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Summary + key points
    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown(f"**Summary**")
        st.markdown(f"<div style='font-size:0.88rem;color:#94A3B8;line-height:1.6;'>{summary}</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown(f"**Key points**")
        for point in key_points:
            st.markdown(f"<div style='font-size:0.82rem;color:#94A3B8;margin-bottom:4px;'>• {point}</div>", unsafe_allow_html=True)

    # Reasoning
    st.markdown(
        f'<div class="reasoning-box" style="margin-top:1rem;">'
        f'<strong style="color:#6366F1;font-style:normal;">AI reasoning:</strong> {reasoning}'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # Draft reply section
    st.markdown("**Draft reply**")
    tone = TONE_MAP.get(category, "professional and friendly")

    col_draft, col_opts = st.columns([3, 1])
    with col_opts:
        custom_tone = st.selectbox(
            "Tone", ["Auto-detect", "Professional", "Empathetic", "Concise", "Detailed"],
            label_visibility="collapsed",
        )
        generate_clicked = st.button("Generate draft →", type="primary", use_container_width=True)

    if generate_clicked or (st.session_state.draft is None and category != "Spam"):
        if custom_tone != "Auto-detect":
            tone = custom_tone.lower()
        with col_draft:
            with st.spinner("Drafting reply…"):
                draft_chain = get_draft_chain(provider, model, api_key)
                draft = draft_chain.invoke({
                    "email": st.session_state.last_processed,
                    "tone": tone,
                    "key_points": ", ".join(key_points),
                    "priority": priority,
                })
                st.session_state.draft = draft

    if st.session_state.draft:
        with col_draft:
            st.markdown(
                f'<div class="draft-wrapper">'
                f'<div class="draft-header">'
                f'<span class="draft-title">Draft reply</span>'
                f'<span class="draft-meta">Subject: {suggested_subject}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            edited = st.text_area(
                "draft_edit", label_visibility="collapsed",
                value=st.session_state.draft,
                height=280,
                key="draft_textarea",
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if st.button("Copy draft", use_container_width=True):
                st.toast("Draft copied to clipboard!")
