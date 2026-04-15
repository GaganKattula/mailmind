"""Unit tests for core/schemas.py — no API calls required."""
import pytest
from pydantic import ValidationError

from core.schemas import EmailAnalysis, EmailCategory, EmailSentiment, RiskLevel


class TestEmailCategory:
    def test_all_values_exist(self):
        expected = {"Support", "Sales", "Complaint", "Inquiry", "Spam", "Internal", "Other"}
        actual = {c.value for c in EmailCategory}
        assert actual == expected


class TestRiskLevel:
    def test_ordering_values(self):
        assert RiskLevel.HIGH.value == "High"
        assert RiskLevel.MEDIUM.value == "Medium"
        assert RiskLevel.LOW.value == "Low"


class TestEmailAnalysis:
    def _valid_payload(self) -> dict:
        return {
            "category": "Complaint",
            "priority": "High",
            "sentiment": "Frustrated",
            "summary": "Customer wants a refund for a late order.",
            "key_points": ["Order not arrived", "Wants full refund"],
            "suggested_subject": "Re: Your Order — We're On It",
            "reasoning": "Complaint with frustrated tone and explicit refund demand → High priority.",
        }

    def test_valid_payload_parses(self):
        analysis = EmailAnalysis(**self._valid_payload())
        assert analysis.category == EmailCategory.COMPLAINT
        assert analysis.priority == RiskLevel.HIGH

    def test_missing_required_field_raises(self):
        payload = self._valid_payload()
        del payload["category"]
        with pytest.raises(ValidationError):
            EmailAnalysis(**payload)

    def test_invalid_category_raises(self):
        payload = self._valid_payload()
        payload["category"] = "NotACategory"
        with pytest.raises(ValidationError):
            EmailAnalysis(**payload)

    def test_empty_key_points_raises(self):
        payload = self._valid_payload()
        payload["key_points"] = []
        with pytest.raises(ValidationError):
            EmailAnalysis(**payload)

    def test_too_many_key_points_raises(self):
        payload = self._valid_payload()
        payload["key_points"] = [f"point {i}" for i in range(10)]
        with pytest.raises(ValidationError):
            EmailAnalysis(**payload)
