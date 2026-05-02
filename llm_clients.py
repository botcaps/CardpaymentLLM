"""
Multi-provider LLM client.

Single env var picks the provider for the entire pipeline:

  LLM_PROVIDER=openai     + OPENAI_API_KEY=sk-...
  LLM_PROVIDER=anthropic  + ANTHROPIC_API_KEY=sk-ant-...
  LLM_PROVIDER=google     + GEMINI_API_KEY=AIza...     (also: GOOGLE_API_KEY)
  LLM_PROVIDER=groq       + GROQ_API_KEY=gsk_...        (Llama 3.3 70B via Groq)
  LLM_PROVIDER=ollama                                    (local Llama, no key)

  LLM_MODE=mock                                          (deterministic, no API)

Three "model tiers" let agents pick a model by job, not by name:

  fast       cheap, low-latency       (auth/fraud agents, reflection)
  smart      strong reasoning         (cost agent, planner)
  judge      cheap eval-mode model    (LLM-as-judge)

Each provider has a default for each tier (see TIER_DEFAULTS). You can
override any of them with env vars: OVERRIDE_FAST_MODEL, OVERRIDE_SMART_MODEL,
OVERRIDE_JUDGE_MODEL.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

# ── provider/tier matrix ─────────────────────────────────────────────────────

ProviderName = Literal["openai", "anthropic", "google", "groq", "ollama", "mock"]
Tier = Literal["fast", "smart", "judge"]

# Defaults are chosen for: (1) capability, (2) cost, (3) availability.
# Update freely - everything below dispatches on (provider, tier).
TIER_DEFAULTS: dict[str, dict[Tier, str]] = {
    "openai": {
        "fast":  "gpt-4o-mini",
        "smart": "gpt-4o",
        "judge": "gpt-4o-mini",
    },
    "anthropic": {
        "fast":  "claude-haiku-4-5-20251001",
        "smart": "claude-sonnet-4-5",
        "judge": "claude-haiku-4-5-20251001",
    },
    "google": {
        "fast":  "gemini-2.5-flash",
        "smart": "gemini-2.5-flash",
        "judge": "gemini-2.5-flash",
    },
    "groq": {
        # Groq runs Llama models cheap + fast on their custom hardware.
        "fast":  "llama-3.3-70b-versatile",
        "smart": "llama-3.3-70b-versatile",
        "judge": "llama-3.1-8b-instant",
    },
    "ollama": {
        # Local Llama. Assumes user has run `ollama pull llama3.1:8b` etc.
        "fast":  "llama3.1:8b",
        "smart": "llama3.1:70b",
        "judge": "llama3.1:8b",
    },
}

# init_chat_model expects "provider:model" - this is the LangChain naming.
INIT_PROVIDER_NAME = {
    "openai":    "openai",
    "anthropic": "anthropic",
    "google":    "google_genai",
    "groq":      "groq",
    "ollama":    "ollama",
}

# Each provider needs an env var. mock and ollama don't.
PROVIDER_KEY_ENV = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google":    "GEMINI_API_KEY",   # we also accept GOOGLE_API_KEY
    "groq":      "GROQ_API_KEY",
    "ollama":    None,
    "mock":      None,
}


# ── config & dispatch ────────────────────────────────────────────────────────

@dataclass
class LLMConfig:
    provider: ProviderName
    mode: Literal["live", "mock"]

    @property
    def use_llm(self) -> bool:
        """True if real LLM calls should be made (vs deterministic mocks)."""
        return self.mode == "live"

    # back-compat alias for existing graph.py / agent code
    @property
    def use_gemini(self) -> bool:
        return self.use_llm

    def describe(self) -> str:
        if self.mode == "mock":
            return "MOCK (deterministic, no API calls)"
        return f"LIVE ({self.provider})"


def get_config() -> LLMConfig:
    """Resolve provider + mode from environment variables."""
    if os.environ.get("LLM_MODE", "").lower() == "mock":
        return LLMConfig(provider="mock", mode="mock")

    raw = os.environ.get("LLM_PROVIDER", "").lower().strip()
    if raw == "":
        # Auto-detect: pick the first provider whose key is present.
        # Order = cost/availability preference. Tweak freely.
        for p in ("google", "openai", "anthropic", "groq"):
            env_var = PROVIDER_KEY_ENV[p]
            if env_var and os.environ.get(env_var):
                return LLMConfig(provider=p, mode="live")  # type: ignore[arg-type]
            # accept GOOGLE_API_KEY as a synonym for GEMINI_API_KEY
            if p == "google" and os.environ.get("GOOGLE_API_KEY"):
                return LLMConfig(provider="google", mode="live")
        # Ollama needs no key — only auto-pick it if explicitly requested.
        return LLMConfig(provider="mock", mode="mock")

    if raw not in PROVIDER_KEY_ENV:
        raise ValueError(
            f"LLM_PROVIDER={raw!r} is not recognised. "
            f"Valid: {sorted(PROVIDER_KEY_ENV)}"
        )

    if raw == "mock":
        return LLMConfig(provider="mock", mode="mock")

    if raw not in ("ollama",):
        env_var = PROVIDER_KEY_ENV[raw]
        has_key = bool(os.environ.get(env_var)) if env_var else True
        if raw == "google" and not has_key:
            has_key = bool(os.environ.get("GOOGLE_API_KEY"))
        if not has_key:
            raise RuntimeError(
                f"LLM_PROVIDER={raw} but {env_var} is not set. "
                f"Either set the key or use LLM_MODE=mock."
            )

    return LLMConfig(provider=raw, mode="live")  # type: ignore[arg-type]


def _resolve_model_name(provider: str, tier: Tier) -> str:
    """Apply env-var overrides (OVERRIDE_FAST_MODEL etc.) on top of defaults."""
    override = os.environ.get(f"OVERRIDE_{tier.upper()}_MODEL", "").strip()
    if override:
        return override
    return TIER_DEFAULTS[provider][tier]


# ── public API: chat models ──────────────────────────────────────────────────

def get_chat_model(tier: Tier = "fast", **kwargs):
    """
    Return a LangChain BaseChatModel for the configured provider and tier.

    Use:
        llm = get_chat_model("smart")
        llm.invoke([HumanMessage(content="...")])

    Pass through extra kwargs to the underlying constructor (e.g. temperature).
    """
    cfg = get_config()
    if cfg.mode == "mock":
        return _MockChatModel()

    model_name = _resolve_model_name(cfg.provider, tier)
    init_provider = INIT_PROVIDER_NAME[cfg.provider]

    # Use LangChain's universal init_chat_model. Available since langchain 0.3.
    from langchain.chat_models import init_chat_model

    # init_chat_model picks up provider env vars automatically. The one
    # exception is google_genai, which expects GOOGLE_API_KEY but we
    # standardise on GEMINI_API_KEY — alias it here if needed.
    if cfg.provider == "google" and not os.environ.get("GOOGLE_API_KEY"):
        if os.environ.get("GEMINI_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

    return init_chat_model(
        model=model_name,
        model_provider=init_provider,
        **kwargs,
    )


# ── back-compat: model-name constants used by old graph.py prompts ───────────
# These are read by the existing graph.py for prompt-template variables.
# They no longer drive model selection (that goes through get_chat_model).

MODEL_ORCHESTRATOR = "smart"  # tier label, not a model id
MODEL_AUTH_AGENT   = "fast"
MODEL_COST_AGENT   = "smart"


# ── back-compat: gemini_client (used by the old async google-genai path) ────
# Kept so legacy code paths in agents/auth_score/agent.py don't break.
# New code should use get_chat_model() instead.

_gemini_client = None

def gemini_client():
    """Legacy direct google-genai client. Prefer get_chat_model('fast')."""
    global _gemini_client
    if _gemini_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# ── mock chat model (for LLM_MODE=mock) ──────────────────────────────────────

class _MockChatModel:
    """
    Minimal stand-in for a chat model. Used when LLM_MODE=mock.

    The mock returns a fixed string; real agent paths should detect
    cfg.use_llm == False and skip the LLM call entirely (which the
    existing graph.py already does). This class exists only so
    `get_chat_model()` always returns *something* usable.
    """
    def invoke(self, messages, **_kwargs):
        from langchain_core.messages import AIMessage
        return AIMessage(content="[mock] No anomalies detected.")

    async def ainvoke(self, messages, **_kwargs):
        return self.invoke(messages)

    def bind_tools(self, tools):
        return self  # no-op; mock path doesn't actually use tools
