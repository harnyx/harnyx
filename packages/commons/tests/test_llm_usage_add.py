from caster_commons.llm.schema import LlmUsage


def test_add_sums_fields_and_cost() -> None:
    first = LlmUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        prompt_cached_tokens=2,
        reasoning_tokens=3,
    )

    second = LlmUsage(
        prompt_tokens=4,
        completion_tokens=1,
        total_tokens=5,
        prompt_cached_tokens=None,
        reasoning_tokens=2,
    )

    combined = first + second

    assert combined.prompt_tokens == 14
    assert combined.completion_tokens == 6
    assert combined.total_tokens == 20
    assert combined.prompt_cached_tokens == 2
    assert combined.reasoning_tokens == 5


def test_add_treats_none_as_zero_and_preserves_none_when_both_missing() -> None:
    first = LlmUsage(completion_tokens=1)
    second = LlmUsage(prompt_tokens=2)

    combined = first + second

    assert combined.prompt_tokens == 2
    assert combined.completion_tokens == 1
    # total_tokens missing on both => remains None
    assert combined.total_tokens is None
    # prompt_cached_tokens missing on both => None
    assert combined.prompt_cached_tokens is None


def test_radd_supports_sum_with_zero_start() -> None:
    usages = [LlmUsage(prompt_tokens=1), LlmUsage(prompt_tokens=2)]

    combined = sum(usages, start=0)

    assert isinstance(combined, LlmUsage)
    assert combined.prompt_tokens == 3


def test_iadd_returns_new_instance() -> None:
    usage = LlmUsage(prompt_tokens=1)
    usage += LlmUsage(completion_tokens=2)

    assert usage.prompt_tokens == 1
    assert usage.completion_tokens == 2
