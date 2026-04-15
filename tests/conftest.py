"""Shared fixtures for MailMind tests."""
import pytest


@pytest.fixture
def angry_email() -> str:
    return (
        "I've been waiting 3 weeks for my order and it still hasn't arrived. "
        "I want a full refund immediately. This is completely unacceptable."
    )


@pytest.fixture
def sales_email() -> str:
    return (
        "Hi, we're a 50-person company evaluating AI tools. "
        "Could you share pricing and whether you offer enterprise contracts? "
        "We'd like to schedule a demo this week."
    )


@pytest.fixture
def minimal_email() -> str:
    return "It's not working."
