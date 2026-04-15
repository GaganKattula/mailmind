"""
Email classification chain.

Builds a LangChain LCEL chain that takes an email body and returns
a structured EmailAnalysis object.
"""
from __future__ import annotations

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

from .schemas import EmailAnalysis

_SYSTEM_PROMPT = (
    "You are an expert email analyst for a business. Analyze the incoming email "
    "and return a JSON object matching the schema exactly.\n\n"
    "{format_instructions}\n\n"
    "Priority guide:\n"
    "  High   — needs a response within 2 hours (angry customers, outages, urgent deadlines)\n"
    "  Medium — respond within 24 hours (sales leads, general support)\n"
    "  Low    — respond within a week (cold outreach, low-urgency inquiries)\n"
)


def build_classifier_chain(llm):
    """
    Build the email classification chain.

    Parameters
    ----------
    llm : any LangChain chat model

    Returns
    -------
    LCEL chain that accepts {"email": str} and returns dict matching EmailAnalysis
    """
    parser = JsonOutputParser(pydantic_object=EmailAnalysis)
    prompt = ChatPromptTemplate.from_messages([
        ("system", _SYSTEM_PROMPT),
        ("human", "Analyze this email:\n\n{email}"),
    ]).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser
