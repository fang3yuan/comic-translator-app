"""Tests for the Gemini translator (fake HTTP, no network) and Arabic post-processing."""

import json

import pytest

from src.config import Settings
from src.translators import (
    ConfigurationError,
    GeminiTranslator,
    TranslateItem,
    TranslationError,
    create_translator,
    diagnose,
    parse_keys,
)
from src.translators.gemini import extract_json, parse_translations, suggest_max_chars
from src.utils.arabic_utils import contains_arabic, postprocess_translation, shape_arabic


class Resp:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = status, body, headers or {}, json.dumps(body)

    def json(self):
        return self._body


def ok(pairs):
    """A successful Gemini answer containing the given {id: text}."""
    payload = json.dumps({"translations": [{"id": i, "text": t} for i, t in pairs.items()]}, ensure_ascii=False)
    return Resp(200, {"candidates": [{"content": {"parts": [{"text": payload}]}}]})


def err(status, message="x", extra=None):
    return Resp(status, {"error": {"code": status, "message": message, **(extra or {})}})


class Server:
    """Scripted fake of requests.post: each call pops the next handler (or echoes ids)."""

    def __init__(self, *script):
        self.script, self.calls = list(script), []

    def __call__(self, url, headers=None, data=None, timeout=None):
        body = json.loads(data)
        self.calls.append({"url": url, "key": headers["x-goog-api-key"], "body": body})
        if self.script:
            step = self.script.pop(0)
            return step(body) if callable(step) else step
        items = json.loads(body["contents"][0]["parts"][0]["text"])["items"]
        return ok({i["id"]: f"ar:{i['text']}" for i in items})


def make(server, keys="KEY1234567890", **kw):
    clock = {"t": 1000.0}
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        clock["t"] += s

    tr = GeminiTranslator(keys, poster=server, sleep=sleep, clock=lambda: clock["t"], **kw)
    tr.sleeps = sleeps
    return tr


ITEMS = [TranslateItem(0, "Hello there", "dialogue", 16), TranslateItem(1, "BOOM", "sfx", 8)]


def test_parse_keys_accepts_lines_commas_and_duplicates():
    assert parse_keys("A1\nB2, C3  A1;D4") == ["A1", "B2", "C3", "D4"]
    assert parse_keys(["A1", "B2\nC3"]) == ["A1", "B2", "C3"]
    assert parse_keys("") == [] and parse_keys(None) == []


def test_extract_json_tolerates_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": [1, 2]} bye') == {"a": [1, 2]}
    with pytest.raises(ValueError):
        extract_json("no json here")
    assert parse_translations({"translations": [{"id": "3", "text": " مرحبا "}, {"id": 4, "text": ""}, "junk"]}) == {3: "مرحبا"}


def test_whole_page_goes_in_one_request_with_length_hints():
    server = Server()
    out = make(server).translate_items(ITEMS)
    assert out == {0: "ar:Hello there", 1: "ar:BOOM"}
    assert len(server.calls) == 1
    sent = json.loads(server.calls[0]["body"]["contents"][0]["parts"][0]["text"])
    assert [i["id"] for i in sent["items"]] == [0, 1] and sent["items"][0]["max_chars"] == 16
    assert sent["items"][1]["role"] == "sfx"
    cfg = server.calls[0]["body"]["generationConfig"]
    assert cfg["responseMimeType"] == "application/json" and "responseSchema" in cfg
    assert "max_chars" in server.calls[0]["body"]["systemInstruction"]["parts"][0]["text"]
    assert server.calls[0]["url"].endswith(":generateContent")


def test_custom_notes_reach_the_prompt():
    server = Server()
    make(server, notes="Keep the name Luffy").translate_items(ITEMS)
    assert "Keep the name Luffy" in server.calls[0]["body"]["systemInstruction"]["parts"][0]["text"]


def test_missing_blocks_are_requested_again():
    server = Server(ok({0: "واحد"}))  # block 1 is missing from the first answer
    out = make(server).translate_items(ITEMS)
    assert out == {0: "واحد", 1: "ar:BOOM"}
    second = json.loads(server.calls[1]["body"]["contents"][0]["parts"][0]["text"])
    assert [i["id"] for i in second["items"]] == [1]  # only the missing one


def test_gives_up_when_blocks_never_come_back():
    server = Server(*[ok({0: "واحد"})] * 5)
    with pytest.raises(TranslationError):
        make(server, retries=2).translate_items(ITEMS)


def test_rate_limited_key_rotates_to_the_next_key():
    server = Server(err(429, "quota", {"details": [{"retryDelay": "40s"}]}))
    tr = make(server, keys="KEY_AAAAAAAAAA\nKEY_BBBBBBBBBB")
    assert tr.translate_items(ITEMS) == {0: "ar:Hello there", 1: "ar:BOOM"}
    assert [c["key"] for c in server.calls][0] != [c["key"] for c in server.calls][1]
    tr.translate_items(ITEMS)  # the limited key is skipped while it cools down
    assert server.calls[-1]["key"] == server.calls[1]["key"]


def test_single_key_waits_for_retry_delay_then_succeeds():
    server = Server(err(429, "quota", {"details": [{"retryDelay": "12s"}]}))
    tr = make(server)
    assert tr.translate_items(ITEMS)[0] == "ar:Hello there"
    assert any(s >= 12 for s in tr.sleeps)


def test_daily_quota_is_reported_instead_of_waiting_for_hours():
    server = Server(err(429, "Quota exceeded for metric ... PerDay"))
    with pytest.raises(TranslationError) as exc:
        make(server).translate_items(ITEMS)
    assert "limit" in str(exc.value).lower()


def test_invalid_key_is_dropped_and_the_next_one_used():
    server = Server(err(400, "API key not valid. Please pass a valid API key."))
    tr = make(server, keys="BADKEY_0000000\nGOODKEY_111111")
    assert tr.translate_items(ITEMS)[1] == "ar:BOOM"
    assert server.calls[-1]["key"] == "GOODKEY_111111"


def test_all_keys_invalid_is_a_configuration_error():
    server = Server(err(403, "forbidden"), err(403, "forbidden"))
    with pytest.raises(ConfigurationError):
        make(server, keys="AAAAAAAAAAAA\nBBBBBBBBBBBB").translate_items(ITEMS)


def test_no_key_explains_where_to_get_one():
    with pytest.raises(ConfigurationError) as exc:
        make(Server(), keys="").translate_items(ITEMS)
    assert "aistudio.google.com" in str(exc.value)


def test_unknown_model_falls_back_to_the_next_one():
    server = Server(err(404, "models/foo is not found for API version v1beta"))
    tr = make(server, model="foo")
    tr.translate_items(ITEMS)
    assert "foo" in server.calls[0]["url"] and "foo" not in server.calls[1]["url"]


def test_server_rejecting_optional_fields_is_handled_once():
    server = Server(err(400, "Invalid JSON payload: Unknown name 'responseSchema'"))
    tr = make(server)
    tr.translate_items(ITEMS)
    assert "responseSchema" in server.calls[0]["body"]["generationConfig"]
    assert "responseSchema" not in server.calls[1]["body"]["generationConfig"]
    tr.translate_items([TranslateItem(5, "Again")])
    assert "responseSchema" not in server.calls[-1]["body"]["generationConfig"]  # remembered


def test_temporary_server_errors_are_retried():
    server = Server(err(503, "overloaded"), err(500, "oops"))
    assert make(server, retries=3).translate_items(ITEMS)[0] == "ar:Hello there"
    assert len(server.calls) == 3


def test_requests_are_spaced_by_min_interval():
    server = Server()
    tr = make(server, min_interval=1.0)
    tr.translate_items(ITEMS)
    tr.translate_items([TranslateItem(9, "Another page")])
    assert any(abs(s - 1.0) < 0.01 for s in tr.sleeps)  # one second between the two requests


def test_big_pages_are_split_into_chunks():
    server = Server()
    items = [TranslateItem(i, f"line {i}") for i in range(85)]
    out = make(server).translate_items(items)
    assert len(out) == 85 and len(server.calls) == 3  # 40 + 40 + 5


def test_blocked_prompt_is_explained():
    blocked = Resp(200, {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []})
    with pytest.raises(TranslationError) as exc:
        make(Server(blocked)).translate_items(ITEMS)
    assert "SAFETY" in str(exc.value)


def test_key_never_leaks_in_error_text():
    def boom(*a, **k):
        raise ConnectionError("Max retries exceeded with url ...?key=KEY1234567890")

    with pytest.raises(TranslationError) as exc:
        make(boom, retries=1).translate_items(ITEMS)
    assert "KEY1234567890" not in str(exc.value)


def test_check_keys_reports_each_key():
    server = Server(ok({0: "مرحبا"}), err(400, "API key not valid"))
    checks = make(server, keys="GOODKEY_111111\nBADKEY_2222222").check_keys()
    assert [c.ok for c in checks] == [True, False] and "…" in checks[0].label


def test_create_translator_and_diagnose_from_settings():
    s = Settings({"translation": {"gemini_keys": ["K1_AAAAAAAAAA", "K2_BBBBBBBBBB"], "model": "m1", "notes": "n"}})
    tr = create_translator(s)
    assert len(tr.keys) == 2 and tr.models[0] == "m1" and tr.notes == "n"
    empty = diagnose(Settings())
    assert len(empty) == 1 and not empty[0].ok


def test_length_budget_is_close_to_the_english_length():
    assert suggest_max_chars("Hello") >= 8
    assert suggest_max_chars("x" * 100) < 130


def test_postprocess_punctuation_and_digits():
    out = postprocess_translation("هل أنت بخير? نعم, أنا بخير...", arabic_digits=False)
    assert "؟" in out and "،" in out and "…" in out
    assert postprocess_translation("عندي 3 تفاحات", arabic_digits=True).count("٣") == 1


def test_shape_arabic_changes_text_and_keeps_arabic():
    shaped = shape_arabic("مرحبا")
    assert shaped != "مرحبا" and contains_arabic(shaped) and shaped.count("\n") == 0
