"""
LLM access via the OpenAI API, with a hard guarantee: the pipeline runs with no
API key.

Public interface is IDENTICAL to the Anthropic version - `.live`, `.mode`,
`.text()`, `.json()` - so extractor.py, generator.py, simulator.py and
pipeline.py need no changes at all.

If OPENAI_API_KEY is set, calls go to the real model. If not, MockHeuristics
takes over with deterministic keyword logic. The mock is deliberately good at
Tier 0-2 and deliberately bad at Tier 3-4, which is exactly the real-world
failure this project exists to fix. So the offline demo still tells the truth.

Never let a missing key be the reason your demo dies at 3am.

Self-test:
    python -m agentready.llm
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, List, Optional

MODEL = os.environ.get("AGENTREADY_MODEL", "gpt-4o-mini")
BASE_URL = os.environ.get("OPENAI_BASE_URL")  # set this for a school/hackathon proxy
MAX_TOKENS = 4000


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def safe_json(text: str) -> Optional[Any]:
    """Parse JSON out of a model reply without dying on preamble or fences."""
    if not text:
        return None
    cleaned = _strip_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start, end = cleaned.find(opener), cleaned.rfind(closer)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


# ---------------------------------------------------------------------------
# OpenAI strict structured outputs are fussier than Anthropic's tool schemas.
# Three rules that will silently 400 you if you get them wrong:
#   1. EVERY object needs "additionalProperties": false
#   2. EVERY object needs all of its properties listed in "required"
#   3. Validation keywords like minimum/maximum are NOT supported
# This converts our schema.py output to satisfy all three.
# ---------------------------------------------------------------------------

_UNSUPPORTED = ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                "multipleOf", "minLength", "maxLength", "pattern",
                "minItems", "maxItems", "uniqueItems", "default", "format")


def openaify(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively make a JSON Schema safe for OpenAI strict mode."""
    node = copy.deepcopy(schema)

    def walk(n: Any) -> Any:
        if not isinstance(n, dict):
            return n
        for kw in _UNSUPPORTED:
            n.pop(kw, None)

        if n.get("type") == "object" or "properties" in n:
            props = n.get("properties", {})
            for k, v in props.items():
                props[k] = walk(v)
            n["properties"] = props
            n["additionalProperties"] = False
            n["required"] = list(props.keys())  # strict mode: ALL keys required

        if "items" in n:
            n["items"] = walk(n["items"])

        for key in ("anyOf", "oneOf", "allOf"):
            if key in n:
                n[key] = [walk(x) for x in n[key]]
        return n

    return walk(node)


class LLM:
    """Thin wrapper. `.json()` is the only method the pipeline should call."""

    def __init__(self, api_key: Optional[str] = None, model: str = MODEL,
                 base_url: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or BASE_URL
        self.client = None
        self.live = False
        self.supports_strict = True  # flipped off automatically on first rejection

        if self.api_key:
            try:
                from openai import OpenAI  # noqa
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self.client = OpenAI(**kwargs)
                self.live = True
            except Exception as exc:  # pragma: no cover
                print(f"[llm] openai SDK unavailable ({exc}); falling back to mock")

    @property
    def mode(self) -> str:
        return f"openai:{self.model}" if self.live else "mock"

    # -- raw text ----------------------------------------------------------

    def text(self, system: str, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        if not self.live:
            raise RuntimeError("text() requires a live client; use MockHeuristics paths instead")
        resp = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content or ""

    # -- structured --------------------------------------------------------

    def json(
        self,
        system: str,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        max_tokens: int = MAX_TOKENS,
        retries: int = 2,
        schema_name: str = "extraction",
    ) -> Optional[Any]:
        """
        Ask for JSON. Three levels of degradation, so one schema quirk never
        kills a run:
          1. strict json_schema  (guaranteed shape)
          2. json_object mode    (guaranteed valid JSON, shape not enforced)
          3. plain text + safe_json parsing
        """
        if not self.live:
            return None

        # Level 1: strict structured output
        if schema and self.supports_strict:
            out = self._call(system, prompt, max_tokens, response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": openaify(schema),
                    "strict": True,
                },
            })
            if out is not None:
                return out
            # A rejected schema is a config problem, not a transient one. Stop
            # retrying strict mode for the rest of the run.
            self.supports_strict = False
            print("[llm] strict schema rejected; falling back to json_object mode")

        # Level 2: json_object mode
        sys_prompt = system
        if schema:
            sys_prompt += ("\n\nReturn ONLY a JSON object conforming to this JSON Schema. "
                           "No preamble, no markdown fences.\n" + json.dumps(schema))
        else:
            sys_prompt += "\n\nReturn ONLY valid JSON. No preamble, no markdown fences."

        for attempt in range(retries + 1):
            out = self._call(sys_prompt, prompt, max_tokens,
                             response_format={"type": "json_object"})
            if out is not None:
                return out
            # Level 3: some proxies do not support response_format at all
            out = self._call(sys_prompt, prompt, max_tokens, response_format=None)
            if out is not None:
                return out
            prompt += "\n\nYour previous reply was not parseable JSON. Return the raw JSON object only."
        return None

    def _call(self, system: str, prompt: str, max_tokens: int,
              response_format: Optional[Dict[str, Any]]) -> Optional[Any]:
        kwargs: Dict[str, Any] = dict(
            model=self.model,
            max_tokens=max_tokens,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        if response_format:
            kwargs["response_format"] = response_format
        try:
            resp = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            msg = str(exc)
            # Newer reasoning models renamed the token cap and reject temperature.
            if "max_tokens" in msg and "max_completion_tokens" in msg:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
                kwargs.pop("temperature", None)
                try:
                    resp = self.client.chat.completions.create(**kwargs)
                except Exception as exc2:
                    print(f"[llm] call failed: {str(exc2)[:200]}")
                    return None
            else:
                print(f"[llm] call failed: {msg[:200]}")
                return None

        content = resp.choices[0].message.content
        if getattr(resp.choices[0], "finish_reason", "") == "length":
            print("[llm] warning: response hit the token cap and may be truncated")
        return safe_json(content)


# ---------------------------------------------------------------------------
# Mock heuristics used when there is no API key. Provider-independent.
# ---------------------------------------------------------------------------

class MockHeuristics:
    """
    Keyword extraction that mimics a catalog-era content stack: strong on
    identity and specs, weak on context and constraints. That asymmetry is the
    point of the demo, so the mock is honest rather than flattering.
    """

    MATERIALS = ["mesh", "knit", "foam", "rubber", "leather", "cotton", "nylon",
                 "polyester", "recycled", "carbon", "aluminium", "ceramic", "silicone"]
    COLORS = ["black", "white", "blue", "red", "green", "grey", "gray", "pink",
              "navy", "orange", "beige", "cream", "yellow", "purple"]

    @staticmethod
    def find_all(text: str, needles: List[str]) -> List[str]:
        low = text.lower()
        return sorted({n for n in needles if n in low})

    @staticmethod
    def sentence_with(text: str, needle: str) -> Optional[str]:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if needle.lower() in sent.lower():
                return sent.strip()
        return None

    @staticmethod
    def number_near(text: str, patterns: List[str]) -> Optional[float]:
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, IndexError):
                    continue
        return None


# ---------------------------------------------------------------------------
# Self-test: python -m agentready.llm
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from .schema import fields_for_tiers, json_schema_for

    llm = LLM()
    print(f"mode      : {llm.mode}")
    print(f"base_url  : {llm.base_url or 'default (api.openai.com)'}")

    raw = json_schema_for(fields_for_tiers([3]))
    conv = openaify(raw)

    def check(node, path="root", problems=None):
        problems = problems if problems is not None else []
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                if node.get("additionalProperties") is not False:
                    problems.append(f"{path}: missing additionalProperties=false")
                props = set(node.get("properties", {}))
                if set(node.get("required", [])) != props:
                    problems.append(f"{path}: required != properties")
            for k, v in node.items():
                check(v, f"{path}.{k}", problems)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                check(v, f"{path}[{i}]", problems)
        return problems

    issues = check(conv)
    print(f"schema    : {len(conv['properties'])} fields converted")
    print(f"strict ok : {'YES' if not issues else 'NO -> ' + '; '.join(issues[:3])}")
    print(f"unsupported keywords remaining: "
          f"{'none' if 'minimum' not in json.dumps(conv) else 'FOUND minimum/maximum'}")

    if llm.live:
        out = llm.json("You are a test.", 'Return {"ok": true, "note": "hello"}')
        print(f"live call : {out}")
    else:
        print("live call : skipped (no OPENAI_API_KEY). Pipeline will run in mock mode.")
