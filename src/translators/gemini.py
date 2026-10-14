"""Gemini translator: one request per page, several API keys, structured JSON answers.

All the speech bubbles of a page are sent together (so pronouns, names and tone stay
consistent) and the model must answer with ``{"translations": [{"id": n, "text": "..."}]}``.
The answer is validated; anything missing is asked again. Requests are spaced by
``min_interval`` seconds (default 1) and keys are rotated when one hits its limit.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.translators.base import ConfigurationError, TranslationError, explain_error

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ("gemini-flash-latest", "gemini-2.5-flash")
KEY_URL = "https://aistudio.google.com/apikey"
MAX_ITEMS_PER_REQUEST = 40

SYSTEM_PROMPT = """\
You are a professional manga and comic translator (English to Arabic).
You receive ALL the text blocks of ONE page, in reading order, as JSON.

Rules:
1. Translate every block into natural Modern Standard Arabic that matches its role:
   dialogue = lively and natural, narration = literary, whisper = soft, sfx = a short Arabic
   sound word (or a transliteration), title/caption = concise.
2. The Arabic MUST fit the same speech bubble. Never write more than "max_chars" characters
   (hard limit). If the literal translation is longer, rephrase and shorten it. Never drop the meaning.
3. Use the whole page as context for pronouns, gender, names and tone. Keep names consistent.
4. Keep punctuation meaningful (use the Arabic question mark and comma). Keep "..." as "...".
5. Output Arabic script only: no notes, no quotes around the text, no emoji, no transliteration of
   ordinary words. Proper names may be transliterated.
6. Do not merge, split, skip or reorder blocks. Every input "id" must appear exactly once.
7. Reply with JSON only, exactly in this shape:
   {"translations":[{"id":<number>,"text":"<Arabic>"}]}
"""


@dataclass
class TranslateItem:
    """One text block to translate.

    Attributes:
        id: Block id (unique inside the page).
        text: English source text.
        role: ``dialogue`` / ``narration`` / ``sfx`` / ``whisper``.
        max_chars: Hard length limit for the Arabic text (``None`` = no hint).
    """

    id: int
    text: str
    role: str = "dialogue"
    max_chars: Optional[int] = None


@dataclass
class KeyCheck:
    """Result of testing one API key.

    Attributes:
        label: Masked key (``AIza…xyz1``).
        ok: Whether a test request succeeded.
        message: Translation sample or the explained error.
    """

    label: str
    ok: bool
    message: str


def mask_key(key: str) -> str:
    """Return a safe-to-display version of an API key."""
    return f"{key[:4]}…{key[-4:]}" if len(key) > 10 else "…"


def parse_keys(raw: Any) -> List[str]:
    """Normalise keys typed by the user (one per line, or separated by commas/spaces).

    Args:
        raw: A string or a list of strings.

    Returns:
        Unique non-empty keys in the original order.
    """
    parts = re.split(r"[\s,;]+", raw) if isinstance(raw, str) else [p for r in (raw or []) for p in re.split(r"[\s,;]+", str(r))]
    seen: Dict[str, None] = {}
    for p in parts:
        if p.strip():
            seen.setdefault(p.strip(), None)
    return list(seen)


def extract_json(text: str) -> Any:
    """Parse the model's answer, tolerating code fences and surrounding prose.

    Args:
        text: Raw text returned by the model.

    Returns:
        Decoded JSON value.

    Raises:
        ValueError: If no JSON object/array can be found.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = text.find(open_c), text.rfind(close_c)
        if 0 <= a < b:
            try:
                return json.loads(text[a: b + 1])
            except ValueError:
                continue
    raise ValueError("the answer is not valid JSON")


def parse_translations(data: Any) -> Dict[int, str]:
    """Extract ``{id: text}`` from the structured answer.

    Args:
        data: Decoded JSON (``{"translations": [...]}`` or a bare list).

    Returns:
        Mapping of block id to Arabic text (blank texts are dropped).
    """
    rows = data.get("translations") if isinstance(data, dict) else data
    out: Dict[int, str] = {}
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        try:
            rid = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        text = str(row.get("text", "")).strip()
        if text:
            out[rid] = text
    return out


def suggest_max_chars(text: str) -> int:
    """Length budget (characters) for the Arabic translation of ``text``."""
    n = len(text.strip())
    return max(6, int(n * 1.15) + (6 if n <= 14 else 3))


@dataclass
class _Key:
    value: str
    cooldown_until: float = 0.0
    dead: str = ""  # reason, when the key is unusable


@dataclass
class _Call:
    """Outcome of a single HTTP request."""

    status: int
    body: Any
    text: str = ""
    retry_after: float = 0.0


class GeminiTranslator:
    """Translate a page with the Gemini API (free keys from Google AI Studio)."""

    name = "Gemini"

    def __init__(
        self,
        keys: Any,
        model: str = DEFAULT_MODEL,
        min_interval: float = 1.0,
        timeout: float = 90.0,
        retries: int = 3,
        notes: str = "",
        poster: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the translator.

        Args:
            keys: API keys (list or text with one key per line).
            model: Model code name; unknown names fall back to ``FALLBACK_MODELS``.
            min_interval: Minimum seconds between two requests (whatever the key).
            timeout: Network timeout per request.
            retries: Attempts for temporary failures (5xx, timeouts, bad answers).
            notes: Extra instructions from the user (tone, glossary...).
            poster: HTTP POST function (``requests.post`` compatible), injectable for tests.
            sleep: Sleep function (injectable for tests).
            clock: Monotonic clock (injectable for tests).
        """
        self.keys = [_Key(k) for k in parse_keys(keys)]
        self.models = [m for m in dict.fromkeys([model.strip() or DEFAULT_MODEL, *FALLBACK_MODELS])]
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.notes = notes.strip()
        self._post = poster
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last = -1e9
        self._rr = 0
        self._model_idx = 0
        self._off: List[str] = []  # optional request features the server rejected
        self.last_used: Optional[str] = None
        self.requests_sent = 0

    # -- low level ------------------------------------------------------
    @property
    def model(self) -> str:
        """Model currently in use."""
        return self.models[min(self._model_idx, len(self.models) - 1)]

    def _throttle(self) -> None:
        """Keep at least ``min_interval`` seconds between requests."""
        with self._lock:
            wait = self.min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()

    def _pick_key(self) -> _Key:
        """Next usable key (round robin); waits briefly if all are cooling down.

        Raises:
            ConfigurationError: If no key is set or all keys are invalid.
            TranslationError: If every key is out of quota for too long.
        """
        if not self.keys:
            raise ConfigurationError(
                "No Gemini API key. Get a free key at aistudio.google.com/apikey and paste it in the "
                "Translation tab. / لا يوجد مفتاح Gemini. احصل على مفتاح مجاني من aistudio.google.com/apikey "
                "والصقه في تبويب Translation.")
        alive = [k for k in self.keys if not k.dead]
        if not alive:
            raise ConfigurationError("All API keys were rejected by Google: " + "; ".join(
                f"{mask_key(k.value)}: {k.dead}" for k in self.keys) + " / كل المفاتيح مرفوضة، تأكد من نسخها كاملة.")
        now = self._clock()
        ready = [k for k in alive if k.cooldown_until <= now]
        if not ready:
            soonest = min(alive, key=lambda k: k.cooldown_until)
            wait = soonest.cooldown_until - now
            if wait > 75:
                raise TranslationError(
                    f"Every key reached its limit (try again in about {int(wait // 60) + 1} min, or add a key "
                    "from another Google account). / كل المفاتيح وصلت حدها، حاول بعد قليل أو أضف مفتاحًا من "
                    "حساب Google آخر.")
            self._sleep(wait + 0.2)
            return soonest
        key = ready[self._rr % len(ready)]
        self._rr += 1
        return key

    def _body(self, user_json: str, schema: bool, thinking: bool) -> Dict[str, Any]:
        cfg: Dict[str, Any] = {"temperature": 0.3, "responseMimeType": "application/json"}
        if schema:
            cfg["responseSchema"] = {
                "type": "OBJECT",
                "properties": {"translations": {"type": "ARRAY", "items": {
                    "type": "OBJECT",
                    "properties": {"id": {"type": "INTEGER"}, "text": {"type": "STRING"}},
                    "required": ["id", "text"]}}},
                "required": ["translations"],
            }
        if thinking:
            cfg["thinkingConfig"] = {"thinkingBudget": 0}
        system = SYSTEM_PROMPT + (f"\nExtra instructions from the user:\n{self.notes}\n" if self.notes else "")
        safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in (
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user_json}]}],
                "generationConfig": cfg, "safetySettings": safety}

    def _send(self, key: _Key, body: Dict[str, Any]) -> _Call:
        """POST one request (no retry logic here)."""
        post = self._post
        if post is None:
            try:
                import requests
            except ImportError as exc:  # pragma: no cover
                raise ConfigurationError("The 'requests' package is missing - open Setup and press Install.") from exc
            post = requests.post
        self._throttle()
        self.requests_sent += 1
        resp = post(f"{API_ROOT}/{self.model}:generateContent", headers={"x-goog-api-key": key.value,
                    "Content-Type": "application/json"}, data=json.dumps(body), timeout=self.timeout)
        try:
            payload = resp.json()
        except ValueError:
            payload = None
        retry_after = 0.0
        try:
            retry_after = float(resp.headers.get("Retry-After", 0))
        except (AttributeError, TypeError, ValueError):
            pass
        return _Call(int(resp.status_code), payload, getattr(resp, "text", "") or "", retry_after)

    @staticmethod
    def _error_message(call: _Call) -> str:
        err = call.body.get("error") if isinstance(call.body, dict) else None
        return str((err or {}).get("message") or call.text or f"HTTP {call.status}")[:300]

    @staticmethod
    def _retry_delay(call: _Call) -> float:
        """Seconds Google asks us to wait (from ``retryDelay`` / ``Retry-After``), 0 if unknown."""
        if call.retry_after:
            return call.retry_after
        m = re.search(r'"?retryDelay"?\s*[:=]\s*"?([\d.]+)s', json.dumps(call.body) if call.body else call.text)
        return float(m.group(1)) if m else 0.0

    def _request(self, user_json: str) -> str:
        """Send the page and return the model's raw text, handling keys/models/errors.

        Raises:
            ConfigurationError: Invalid keys / unavailable model.
            TranslationError: Network or quota problems after all attempts.
        """
        attempts, last_error = 0, ""
        while attempts < self.retries + len(self.keys) + len(self.models) + 2:
            attempts += 1
            key = self._pick_key()
            body = self._body(user_json, "schema" not in self._off, "thinking" not in self._off)
            try:
                call = self._send(key, body)
            except ConfigurationError:
                raise
            except Exception as exc:  # noqa: BLE001 - network errors
                last_error = explain_error(exc).replace(key.value, mask_key(key.value))
                logger.warning("Gemini request failed: %s", last_error)
                self._sleep(min(8.0, 1.5 * attempts))
                continue
            s = call.status
            if s == 200:
                text = self._answer_text(call)
                if text is None:
                    raise TranslationError(self._block_reason(call))
                key.cooldown_until = 0.0
                self.last_used = f"{self.name} ({self.model})"
                return text
            msg = self._error_message(call)
            low = msg.lower()
            if s == 429:
                delay = self._retry_delay(call) or (3600.0 if "perday" in low.replace(" ", "") or "per day" in low else 30.0)
                key.cooldown_until = self._clock() + delay + 0.5
                last_error = f"{mask_key(key.value)}: rate limit ({int(delay)}s)"
                continue  # rotate to another key (or wait for the soonest one)
            if s in (401, 403) or "api key not valid" in low or "api_key_invalid" in low:
                key.dead = "invalid key or no permission (HTTP %d)" % s
                continue
            if s == 404 or "is not found" in low or "not supported for generatecontent" in low:
                if self._model_idx + 1 < len(self.models):
                    self._model_idx += 1
                    continue
                raise ConfigurationError(f"Model not available: {msg} / النموذج غير متاح، غيّر اسم النموذج.")
            if s == 400:
                if "thinking" in low and "thinking" not in self._off:
                    self._off.append("thinking"); continue
                if any(w in low for w in ("schema", "response_schema", "responseschema")) and "schema" not in self._off:
                    self._off.append("schema"); continue
                if "thinking" not in self._off:  # unknown 400: drop the optional features once
                    self._off += ["thinking", "schema"]; continue
                raise TranslationError(f"Google rejected the request: {msg}")
            if s >= 500 or s == 408:
                last_error = f"server error {s}"
                self._sleep(min(10.0, 2.0 * attempts))
                continue
            raise TranslationError(f"Google answered HTTP {s}: {msg}")
        raise TranslationError(f"Gemini did not answer ({last_error or 'unknown error'}). "
                               "/ لم يرد Gemini، تحقق من الاتصال والمفاتيح.")

    @staticmethod
    def _answer_text(call: _Call) -> Optional[str]:
        try:
            parts = call.body["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts if not p.get("thought"))
        except (KeyError, IndexError, TypeError):
            return None

    @staticmethod
    def _block_reason(call: _Call) -> str:
        fb = (call.body or {}).get("promptFeedback", {}) if isinstance(call.body, dict) else {}
        reason = fb.get("blockReason")
        fin = None
        try:
            fin = call.body["candidates"][0].get("finishReason")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        return (f"Gemini blocked this page ({reason or fin or 'no answer'}). "
                "/ منع Gemini ترجمة هذه الصفحة، جرّب إعادة المحاولة أو عدّل الكتلة يدويًا.")

    # -- public ---------------------------------------------------------
    def translate_items(self, items: Sequence[TranslateItem]) -> Dict[int, str]:
        """Translate the blocks of one page.

        Args:
            items: Blocks in reading order.

        Returns:
            ``{id: arabic}`` for every item.

        Raises:
            ConfigurationError: Missing/invalid key or model.
            TranslationError: Network/quota failure, or blocks the model never returned.
        """
        todo = [i for i in items if i.text.strip()]
        result: Dict[int, str] = {}
        for start in range(0, len(todo), MAX_ITEMS_PER_REQUEST):
            chunk = todo[start: start + MAX_ITEMS_PER_REQUEST]
            result.update(self._translate_chunk(chunk, todo))
        return result

    def _translate_chunk(self, chunk: Sequence[TranslateItem], page: Sequence[TranslateItem]) -> Dict[int, str]:
        got: Dict[int, str] = {}
        wanted = {i.id for i in chunk}
        for attempt in range(self.retries):
            missing = [i for i in chunk if i.id not in got]
            payload: Dict[str, Any] = {"items": [
                {"id": i.id, "role": i.role, "text": i.text, **({"max_chars": i.max_chars} if i.max_chars else {})}
                for i in missing]}
            if len(page) > len(missing):  # give the model the rest of the page as context only
                payload["page_context"] = [i.text for i in page if i.id not in {m.id for m in missing}][:60]
            raw = self._request(json.dumps(payload, ensure_ascii=False))
            try:
                parsed = parse_translations(extract_json(raw))
            except ValueError:
                parsed = {}
            got.update({k: v for k, v in parsed.items() if k in wanted})
            if wanted <= set(got):
                return got
            logger.info("Gemini answer incomplete (%d/%d), asking again", len(got), len(wanted))
        gone = sorted(wanted - set(got))
        raise TranslationError(f"Gemini did not translate block(s) {gone}. / لم يترجم Gemini بعض الكتل، أعد المحاولة.")

    def translate_texts(self, texts: Sequence[str]) -> List[str]:
        """Convenience wrapper: translate plain strings, keeping order."""
        items = [TranslateItem(i, t, "dialogue", suggest_max_chars(t)) for i, t in enumerate(texts)]
        out = self.translate_items(items)
        return [out.get(i, "") for i in range(len(texts))]

    def check_keys(self) -> List[KeyCheck]:
        """Test every key with a tiny request and explain failures."""
        checks: List[KeyCheck] = []
        for key in list(self.keys):
            single = GeminiTranslator([key.value], self.models[0], 0.0, self.timeout, 1, self.notes,
                                      self._post, self._sleep, self._clock)
            try:
                text = single.translate_texts(["Hello, how are you?"])[0]
                checks.append(KeyCheck(mask_key(key.value), True, f"{text}   [{single.model}]"))
            except TranslationError as exc:
                checks.append(KeyCheck(mask_key(key.value), False, str(exc)))
            except Exception as exc:  # noqa: BLE001
                checks.append(KeyCheck(mask_key(key.value), False, explain_error(exc)))
        if not self.keys:
            checks.append(KeyCheck("-", False, "No API key / لا يوجد مفتاح. " + KEY_URL))
        return checks
