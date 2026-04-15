"""MailMind core — schemas, classifier, and response drafter."""
from .schemas import EmailAnalysis, RiskLevel, EmailCategory, EmailSentiment
from .classifier import build_classifier_chain
from .drafter import build_drafter_chain, TONE_MAP

__all__ = [
    "EmailAnalysis", "RiskLevel", "EmailCategory", "EmailSentiment",
    "build_classifier_chain", "build_drafter_chain", "TONE_MAP",
]
