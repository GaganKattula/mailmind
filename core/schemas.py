"""
Pydantic schemas for email classification output.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class EmailCategory(str, Enum):
    SUPPORT = "Support"
    SALES = "Sales"
    COMPLAINT = "Complaint"
    INQUIRY = "Inquiry"
    SPAM = "Spam"
    INTERNAL = "Internal"
    OTHER = "Other"


class EmailSentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"
    FRUSTRATED = "Frustrated"
    URGENT = "Urgent"


class RiskLevel(str, Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class EmailAnalysis(BaseModel):
    """Structured output from the email classification chain."""

    category: EmailCategory = Field(
        description="Primary category of the email"
    )
    priority: RiskLevel = Field(
        description="Response priority: High = within 2 hrs, Medium = 24 hrs, Low = 1 week"
    )
    sentiment: EmailSentiment = Field(
        description="Emotional tone of the sender"
    )
    summary: str = Field(
        description="One-sentence plain-English summary of the email"
    )
    key_points: list[str] = Field(
        description="2–4 key requests, issues, or information items",
        min_length=1,
        max_length=6,
    )
    suggested_subject: str = Field(
        description="Suggested subject line for the reply"
    )
    reasoning: str = Field(
        description="Brief explanation of the classification and priority choices"
    )
