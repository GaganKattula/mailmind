"""
Email reply drafting chain.

Builds a LangChain LCEL chain that generates a professional reply
given the original email body, category, priority, and key points.
"""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Auto-detected tones keyed by EmailCategory value
TONE_MAP: dict[str, str] = {
    "Complaint": "empathetic, apologetic, and solution-focused",
    "Support": "helpful, clear, and patient",
    "Sales": "professional, value-focused, and engaging",
    "Inquiry": "informative, friendly, and encouraging",
    "Internal": "direct, collegial, and concise",
    "Spam": "brief and non-committal",
    "Other": "professional and friendly",
}

_SYSTEM_PROMPT = (
    "You are a professional business email writer. Write a {tone} reply to the email below.\n\n"
    "Guidelines:\n"
    "- Address all key points: {key_points}\n"
    "- Match the urgency of a {priority} priority email\n"
    "- Be concise but thorough — no unnecessary filler\n"
    "- End with a clear next step or call to action\n"
    "- Do NOT include a subject line in the body\n"
    "- Sign off as: The Support Team\n"
)


def build_drafter_chain(llm):
    """
    Build the reply drafting chain.

    Parameters
    ----------
    llm : any LangChain chat model (streaming recommended)

    Returns
    -------
    LCEL chain that accepts:
        {
            "email": str,        original email body
            "tone": str,         e.g. "empathetic and solution-focused"
            "key_points": str,   comma-separated key points
            "priority": str,     "High" / "Medium" / "Low"
        }
    and streams the reply as a string.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "Original email:\n\n{email}"),
    ])
    return prompt | llm | StrOutputParser()
