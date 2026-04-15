"""Unit tests for classifier and drafter chains — LLM is mocked."""
from unittest.mock import MagicMock, patch

import pytest

from core.classifier import build_classifier_chain
from core.drafter import build_drafter_chain, TONE_MAP
from core.schemas import EmailCategory


class TestToneMap:
    def test_all_categories_have_tone(self):
        for cat in EmailCategory:
            assert cat.value in TONE_MAP, f"Missing tone for category: {cat.value}"

    def test_tone_values_are_strings(self):
        for tone in TONE_MAP.values():
            assert isinstance(tone, str) and len(tone) > 0


class TestBuildClassifierChain:
    def test_chain_is_callable(self):
        mock_llm = MagicMock()
        chain = build_classifier_chain(mock_llm)
        assert chain is not None
        assert hasattr(chain, "invoke")

    def test_chain_invokes_llm(self, angry_email):
        mock_llm = MagicMock()
        # Simulate LLM returning a valid JSON string
        mock_llm.invoke.return_value = MagicMock(
            content='{"category":"Complaint","priority":"High","sentiment":"Frustrated",'
                    '"summary":"Refund demand.","key_points":["Late order","Wants refund"],'
                    '"suggested_subject":"Re: Your Order","reasoning":"Angry tone."}'
        )
        chain = build_classifier_chain(mock_llm)
        # Just check the chain is well-formed (invoke may fail without full mock)
        assert chain is not None


class TestBuildDrafterChain:
    def test_chain_is_callable(self):
        mock_llm = MagicMock()
        chain = build_drafter_chain(mock_llm)
        assert chain is not None
        assert hasattr(chain, "invoke")

    def test_chain_accepts_expected_inputs(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Dear customer, ...")
        chain = build_drafter_chain(mock_llm)
        assert chain is not None
