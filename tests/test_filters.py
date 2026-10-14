"""Tests for credit / watermark detection."""

import pytest

from src.utils.text_filters import is_credit_text


@pytest.mark.parametrize("text", [
    "READ THIS CHAPTER AT WWW.MANGASITE.COM",
    "Read at mangasite.com",
    "https://example.org/series/12",
    "Translated by SomeGroup",
    "Join our discord",
    "All rights reserved",
    "© 2024 Publisher",
    "Scanlated by Team X, cleaned by Y",
])
def test_credit_texts_are_detected(text):
    assert is_credit_text(text)


@pytest.mark.parametrize("text", [
    "I will find you.",
    "What are you doing?",
    "Wait... what?! No way.",
    "It costs 3.5 dollars",
    "Follow me!",
])
def test_dialogue_is_not_credit(text):
    assert not is_credit_text(text)
