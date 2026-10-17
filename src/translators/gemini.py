"""Gemini translator: one request per page, several API keys, structured JSON answers.

All the speech bubbles of a page are sent together (so pronouns, names and tone stay
consistent) and the model must answer with ``{"translations": [{"id": n, "text": "..."}]}``.
The answer is validated and repaired; anything missing is asked again. Requests are spaced by
``min_interval`` seconds (default 1). Keys rotate when one hits its limit, and when every key
is out of quota for a model the next *backup model* takes over. A small ``SeriesMemory``
(summary, names, last lines) keeps names and tone steady from page to page, and an optional
numbered picture of the page lets the model correct OCR mistakes.
"""

from __future__ import annotations

import base64
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.translators.base import ConfigurationError, TranslationError, explain_error
from src.translators.memory import SeriesMemory

logger = logging.getLogger(__name__)

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-3.5-flash-lite"
FALLBACK_MODELS = ("gemini-flash-latest", "gemini-2.5-flash")
KNOWN_MODELS = (DEFAULT_MODEL, "gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite")
KEY_URL = "https://aistudio.google.com/apikey"
MAX_ITEMS_PER_REQUEST = 40
_NOT_CHAT = re.compile(r"tts|image|embed|aqa|live|audio|robotics|computer|imagen|veo|learnlm|gemma", re.I)

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

MEMORY_RULE = """\
8. "series_memory" (when present) is what is known about this story: keep every name and term spelled
   as in its glossary, follow the summary and continue the tone of "last_bubbles".
9. Also add to the JSON "context": {"summary": "<English story so far including this page, 400 characters
   at most>", "glossary": [{"source": "<name or term>", "arabic": "<Arabic spelling used>"}]}
   listing character names, places and recurring terms (40 entries at most).
"""

VISION_RULE = """\
10. An image of the page is attached; the boxes are numbered with the same "id" values. The text was
   read by OCR and may contain mistakes. When a block's OCR text is wrong, translate what the picture
   says and add the corrected English as "source" in that block's entry; otherwise leave "source" out.
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


_TEXT_KEYS = ("text", "translation", "arabic", "translated")


def parse_answer(data: Any, order: Sequence[int]) -> Tuple[Dict[int, str], Dict[int, str], Any]:
    """Read the whole structured answer, repairing common deviations.

    Accepts ``{"translations": [...]}`` or a bare list; rows may be ``{"id", "text"}`` objects (also
    ``translation``/``arabic`` keys) or plain strings, which map by position when their number is at
    least the number of ids asked for (extra rows are ignored).

    Args:
        data: Decoded JSON.
        order: Ids that were asked for, in order.

    Returns:
        ``(texts, sources, context)``: Arabic by id, corrected English by id, and the ``context`` object.
    """
    rows = data.get("translations") if isinstance(data, dict) else data
    rows = rows if isinstance(rows, list) else []
    texts: Dict[int, str] = {}
    sources: Dict[int, str] = {}
    if rows and all(isinstance(r, str) for r in rows):
        if len(rows) >= len(order):
            texts = {rid: r.strip() for rid, r in zip(order, rows) if r.strip()}
    else:
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                rid = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            text = str(next((row[k] for k in _TEXT_KEYS if row.get(k)), "")).strip()
            if text:
                texts[rid] = text
            if str(row.get("source") or "").strip():
                sources[rid] = str(row["source"]).strip()
    return texts, sources, (data.get("context") if isinstance(data, dict) else None)


def context_parts(context: Any) -> Tuple[str, Dict[str, str]]:
    """Summary and ``{name: arabic}`` glossary from the model's ``context`` answer (tolerant)."""
    if not isinstance(context, dict):
        return "", {}
    glossary: Dict[str, str] = {}
    for row in context.get("glossary") or []:
        if isinstance(row, dict) and str(row.get("source", "")).strip() and str(row.get("arabic", "")).strip():
            glossary[str(row["source"]).strip()[:60]] = str(row["arabic"]).strip()[:60]
    return str(context.get("summary") or ""), glossary


def suggest_max_chars(text: str) -> int:
    """Length budget (characters) for the Arabic translation of ``text``."""
    n = len(text.strip())
    return max(6, int(n * 1.15) + (6 if n <= 14 else 3))


@dataclass
class _Key:
    value: str
    cooldown: Dict[str, float] = field(default_factory=dict)  # model -> time it can be used again
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
        fallback_models: Sequence[str] = FALLBACK_MODELS,
        memory: Optional[SeriesMemory] = None,
        poster: Optional[Callable[..., Any]] = None,
        getter: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create the translator.

        Args:
            keys: API keys (list or text with one key per line).
            model: Main model code name.
            min_interval: Minimum seconds between two requests (whatever the key).
            timeout: Network timeout per request.
            retries: Attempts for temporary failures (5xx, timeouts, bad answers).
            notes: Extra instructions from the user (tone, glossary...).
            fallback_models: Backup models, used when the main one is unknown or out of quota on every key.
            memory: Series memory shared between pages (``None`` = off).
            poster: HTTP POST function (``requests.post`` compatible), injectable for tests.
            getter: HTTP GET function (``requests.get`` compatible), injectable for tests.
            sleep: Sleep function (injectable for tests).
            clock: Monotonic clock (injectable for tests).
        """
        self.keys = [_Key(k) for k in parse_keys(keys)]
        self.models = list(dict.fromkeys(m.strip() for m in [model, *fallback_models] if m and m.strip())) or [DEFAULT_MODEL]
        self.min_interval = max(0.0, float(min_interval))
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.notes = notes.strip()
        self.memory = memory
        self._post, self._get = poster, getter
        self._sleep = sleep
        self._clock = clock
        self._lock = threading.Lock()
        self._last = -1e9
        self._rr = 0
        self._dead_models: List[str] = []  # unknown to the server
        self._off: List[str] = []  # optional request features the server rejected
        self.last_used: Optional[str] = None
        self.last_model: str = self.models[0]
        self.last_sources: Dict[int, str] = {}  # corrected English text (picture mode)
        self.last_missing: List[int] = []  # ids the model never returned
        self.requests_sent = 0

    # -- low level ------------------------------------------------------
    @property
    def model(self) -> str:
        """Model used by the latest request."""
        return self.last_model

    def _throttle(self) -> None:
        """Keep at least ``min_interval`` seconds between requests."""
        with self._lock:
            wait = self.min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = self._clock()

    def _pick(self) -> Tuple[_Key, str]:
        """Next usable ``(key, model)``: the first model with a ready key (keys round robin).

        Waits briefly when everything is cooling down.

        Raises:
            ConfigurationError: No key set, all keys invalid, or no model left.
            TranslationError: Every key is out of quota for every model for too long.
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
        models = [m for m in self.models if m not in self._dead_models]
        if not models:
            raise ConfigurationError("None of the selected models exists. / لا يوجد أي من النماذج المختارة، "
                                     "اختر نموذجًا من القائمة أو اضغط تحديث القائمة.")
        now = self._clock()
        soonest = float("inf")
        for model in models:
            ready = [k for k in alive if k.cooldown.get(model, 0.0) <= now]
            if ready:
                key = ready[self._rr % len(ready)]
                self._rr += 1
                return key, model
            soonest = min(soonest, min(k.cooldown.get(model, 0.0) for k in alive))
        wait = soonest - now
        if wait > 75:
            raise TranslationError(
                f"Every key reached its limit (try again in about {int(wait // 60) + 1} min, add a key "
                "from another Google account, or pick another model). / كل المفاتيح وصلت حدها، حاول بعد قليل "
                "أو أضف مفتاحًا من حساب Google آخر أو اختر نموذجًا آخر.")
        self._sleep(wait + 0.2)
        return self._pick()

    def _body(self, user_json: str, image: Optional[bytes], memory: bool) -> Dict[str, Any]:
        vision = image is not None
        item_props: Dict[str, Any] = {"id": {"type": "INTEGER"}, "text": {"type": "STRING"}}
        if vision:
            item_props["source"] = {"type": "STRING"}
        props: Dict[str, Any] = {"translations": {"type": "ARRAY", "items": {
            "type": "OBJECT", "properties": item_props, "required": ["id", "text"]}}}
        if memory:
            props["context"] = {"type": "OBJECT", "properties": {
                "summary": {"type": "STRING"},
                "glossary": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                    "source": {"type": "STRING"}, "arabic": {"type": "STRING"}}}}}}
        cfg: Dict[str, Any] = {"temperature": 0.3, "responseMimeType": "application/json"}
        if "schema" not in self._off:
            cfg["responseSchema"] = {"type": "OBJECT", "properties": props, "required": ["translations"]}
        if "thinking" not in self._off:
            cfg["thinkingConfig"] = {"thinkingBudget": 0}
        system = SYSTEM_PROMPT + (MEMORY_RULE if memory else "") + (VISION_RULE if vision else "")
        if self.notes:
            system += f"\nExtra instructions from the user:\n{self.notes}\n"
        safety = [{"category": c, "threshold": "BLOCK_NONE"} for c in (
            "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT")]
        parts: List[Dict[str, Any]] = [{"text": user_json}]
        if vision:
            parts.append({"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(image).decode()}})
        return {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": cfg, "safetySettings": safety}

    def _send(self, key: _Key, model: str, body: Dict[str, Any]) -> _Call:
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
        self.last_model = model
        resp = post(f"{API_ROOT}/{model}:generateContent", headers={"x-goog-api-key": key.value,
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

    def _request(self, user_json: str, image: Optional[bytes] = None, memory: bool = False) -> str:
        """Send the page and return the model's raw text, handling keys/models/errors.

        Raises:
            ConfigurationError: Invalid keys / no available model.
            TranslationError: Network or quota problems after all attempts.
        """
        attempts, last_error = 0, ""
        while attempts < self.retries + len(self.keys) + len(self.models) + 3:
            attempts += 1
            key, model = self._pick()
            body = self._body(user_json, None if "vision" in self._off else image, memory)
            try:
                call = self._send(key, model, body)
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
                key.cooldown.pop(model, None)
                self.last_used = f"{self.name} ({model})"
                return text
            msg = self._error_message(call)
            low = msg.lower()
            if s == 429:
                delay = self._retry_delay(call) or (3600.0 if "perday" in low.replace(" ", "") or "per day" in low else 30.0)
                key.cooldown[model] = self._clock() + delay + 0.5
                last_error = f"{mask_key(key.value)}: rate limit on {model} ({int(delay)}s)"
                continue  # another key, then the backup model
            if s in (401, 403) or "api key not valid" in low or "api_key_invalid" in low:
                key.dead = "invalid key or no permission (HTTP %d)" % s
                continue
            if s == 404 or "is not found" in low or "not supported for generatecontent" in low:
                self._dead_models.append(model)
                last_error = f"model {model} not available"
                continue
            if s == 400:
                if "thinking" in low and "thinking" not in self._off:
                    self._off.append("thinking"); continue
                if any(w in low for w in ("schema", "response_schema", "responseschema")) and "schema" not in self._off:
                    self._off.append("schema"); continue
                if image is not None and "vision" not in self._off and any(w in low for w in ("image", "inline", "mime", "media")):
                    self._off.append("vision"); continue
                dropped = [f for f in ("thinking", "schema", "vision") if f not in self._off]
                if dropped:  # unknown 400: drop the optional features once
                    self._off += dropped; continue
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
    def translate_items(self, items: Sequence[TranslateItem], image: Optional[bytes] = None,
                        remember: bool = True) -> Dict[int, str]:
        """Translate the blocks of one page.

        Blocks the model never returns (after the retries) are left out of the result and listed in
        ``last_missing`` so one stubborn block never fails the page. ``last_sources`` holds the
        corrected English text when ``image`` is given.

        Args:
            items: Blocks in reading order.
            image: JPEG of the page with numbered boxes (picture mode), or ``None``.
            remember: Use and update the series memory (off for single-block retries).

        Returns:
            ``{id: arabic}`` for every translated item.

        Raises:
            ConfigurationError: Missing/invalid key or model.
            TranslationError: Network/quota failure, or nothing at all was translated.
        """
        todo = [i for i in items if i.text.strip()]
        result: Dict[int, str] = {}
        self.last_sources, self.last_missing = {}, []
        for start in range(0, len(todo), MAX_ITEMS_PER_REQUEST):
            chunk = todo[start: start + MAX_ITEMS_PER_REQUEST]
            result.update(self._translate_chunk(chunk, todo, image, remember and self.memory is not None))
        self.last_missing = [i.id for i in todo if i.id not in result]
        if todo and not result:
            raise TranslationError("Gemini did not translate any block. / لم يترجم Gemini أي كتلة، أعد المحاولة.")
        if self.last_missing:
            logger.warning("Gemini left %d block(s) untranslated: %s", len(self.last_missing), self.last_missing)
        return result

    def _translate_chunk(self, chunk: Sequence[TranslateItem], page: Sequence[TranslateItem],
                         image: Optional[bytes], remember: bool) -> Dict[int, str]:
        got: Dict[int, str] = {}
        wanted = {i.id for i in chunk}
        for _ in range(self.retries):
            missing = [i for i in chunk if i.id not in got]
            payload: Dict[str, Any] = {"items": [
                {"id": i.id, "role": i.role, "text": i.text, **({"max_chars": i.max_chars} if i.max_chars else {})}
                for i in missing]}
            if len(page) > len(missing):  # give the model the rest of the page as context only
                payload["page_context"] = [i.text for i in page if i.id not in {m.id for m in missing}][:60]
            if remember and self.memory is not None and not self.memory.is_empty():
                payload["series_memory"] = self.memory.payload()
            raw = self._request(json.dumps(payload, ensure_ascii=False), image, remember)
            try:
                texts, sources, context = parse_answer(extract_json(raw), [i.id for i in missing])
            except ValueError:
                texts, sources, context = {}, {}, None
            got.update({k: v for k, v in texts.items() if k in wanted})
            self.last_sources.update({k: v for k, v in sources.items() if k in wanted})
            if remember and self.memory is not None and got:
                by_id = {i.id: i.text for i in chunk}
                summary, glossary = context_parts(context)
                self.memory.update(summary, glossary, [(self.last_sources.get(k, by_id[k]), v) for k, v in got.items()])
            if wanted <= set(got):
                break
            logger.info("Gemini answer incomplete (%d/%d), asking again", len(got), len(wanted))
        return got

    def translate_texts(self, texts: Sequence[str]) -> List[str]:
        """Convenience wrapper: translate plain strings, keeping order."""
        items = [TranslateItem(i, t, "dialogue", suggest_max_chars(t)) for i, t in enumerate(texts)]
        out = self.translate_items(items, remember=False)
        return [out.get(i, "") for i in range(len(texts))]

    def list_models(self) -> List[str]:
        """Models of the account that can generate text, from Google (for the model picker).

        Raises:
            ConfigurationError: No usable key.
            TranslationError: Google could not be reached or refused.
        """
        alive = [k for k in self.keys if not k.dead]
        if not alive:
            raise ConfigurationError("Add a Gemini key first. / أضف مفتاح Gemini أولًا.")
        get = self._get
        if get is None:
            import requests
            get = requests.get
        try:
            resp = get(API_ROOT, headers={"x-goog-api-key": alive[0].value}, params={"pageSize": 200},
                       timeout=self.timeout)
            if int(resp.status_code) != 200:
                raise TranslationError(f"Google answered HTTP {resp.status_code} when listing models.")
            rows = resp.json().get("models", [])
        except TranslationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise TranslationError(explain_error(exc).replace(alive[0].value, mask_key(alive[0].value))) from exc
        names = [str(r.get("name", "")).split("/")[-1] for r in rows
                 if "generateContent" in (r.get("supportedGenerationMethods") or [])]
        return sorted({n for n in names if n.startswith("gemini") and not _NOT_CHAT.search(n)})

    def check_keys(self) -> List[KeyCheck]:
        """Test every key with a tiny request and explain failures."""
        checks: List[KeyCheck] = []
        for key in list(self.keys):
            single = GeminiTranslator([key.value], self.models[0], 0.0, self.timeout, 1, self.notes,
                                      self.models[1:], poster=self._post, sleep=self._sleep, clock=self._clock)
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
