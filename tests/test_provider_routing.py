"""OpenRouter provider routing: config → LLMClient → request body.

OpenRouter fronts one model id with many upstreams at different quantizations, so a run
that does not pin them executes different weights call to call. These tests hold the whole
path: what the config serialises, that only OpenRouter models receive it, and that it is
material to the config fingerprint.
"""

from unittest.mock import MagicMock, patch

import pytest

from daedalus.core.config import (
    ConfigKeyError,
    ProviderRoutingConfig,
    config_fingerprint,
    config_from_dict,
)
from daedalus.core.llm.client import LLMClient


class TestToBody:
    def test_an_unset_config_sends_nothing(self):
        assert ProviderRoutingConfig().to_body() is None

    def test_quantizations_only(self):
        assert ProviderRoutingConfig(quantizations=["fp8"]).to_body() == {
            "quantizations": ["fp8"]
        }

    def test_order_with_fallbacks_disabled(self):
        assert ProviderRoutingConfig(
            order=["baidu/fp8"], allow_fallbacks=False
        ).to_body() == {"order": ["baidu/fp8"], "allow_fallbacks": False}

    def test_allow_fallbacks_false_alone_sends_nothing(self):
        # It would pin no upstream while still removing the fallbacks that keep a
        # 168-task run alive through a single provider's 429.
        assert ProviderRoutingConfig(allow_fallbacks=False).to_body() is None

    def test_every_filter_field_is_carried(self):
        body = ProviderRoutingConfig(
            order=["a/fp8"], only=["b"], ignore=["c"], quantizations=["fp8"], sort="price"
        ).to_body()
        assert body == {
            "order": ["a/fp8"],
            "only": ["b"],
            "ignore": ["c"],
            "quantizations": ["fp8"],
            "sort": "price",
        }

    def test_the_body_does_not_alias_the_config(self):
        cfg = ProviderRoutingConfig(quantizations=["fp8"])
        cfg.to_body()["quantizations"].append("fp4")
        assert cfg.quantizations == ["fp8"]


class TestSentToTheProvider:
    def _call(self, model, routing):
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content="ok", tool_calls=None))]
        response.usage = MagicMock(prompt_tokens=1, completion_tokens=1)
        with patch("litellm.completion", return_value=response) as completion:
            LLMClient(model=model, provider_routing=routing).generate(
                [{"role": "user", "content": "hi"}]
            )
        return completion.call_args.kwargs

    def test_openrouter_model_receives_the_provider_block(self):
        kwargs = self._call("openrouter/deepseek/deepseek-v4-flash-0731", {"quantizations": ["fp8"]})
        assert kwargs["extra_body"] == {"provider": {"quantizations": ["fp8"]}}

    def test_a_non_openrouter_model_never_receives_it(self):
        # OpenAI rejects unknown body fields; only OpenRouter has this concept.
        kwargs = self._call("gpt-5.4-mini", {"quantizations": ["fp8"]})
        assert "extra_body" not in kwargs

    def test_no_routing_means_no_extra_body(self):
        kwargs = self._call("openrouter/deepseek/deepseek-v4-flash-0731", None)
        assert "extra_body" not in kwargs

    def test_an_empty_dict_is_treated_as_unset(self):
        kwargs = self._call("openrouter/qwen/qwen3.6-35b-a3b", {})
        assert "extra_body" not in kwargs


class TestConfigWiring:
    def test_yaml_block_is_parsed(self):
        cfg = config_from_dict(
            {"provider_routing": {"quantizations": ["fp8"], "allow_fallbacks": False}}
        )
        assert cfg.provider_routing.quantizations == ["fp8"]
        assert cfg.provider_routing.allow_fallbacks is False

    def test_default_is_inert(self):
        assert config_from_dict({}).provider_routing.to_body() is None

    def test_a_typo_is_refused(self):
        with pytest.raises(ConfigKeyError, match="quantization"):
            config_from_dict({"provider_routing": {"quantization": ["fp8"]}})

    def test_routing_changes_the_fingerprint(self):
        # Two runs that route to different quantizations did not execute the same weights,
        # so a resume must not silently reuse one's traces under the other's config.
        base = config_from_dict({})
        pinned = config_from_dict({"provider_routing": {"quantizations": ["fp8"]}})
        assert config_fingerprint(base) != config_fingerprint(pinned)
