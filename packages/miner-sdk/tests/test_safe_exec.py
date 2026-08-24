from __future__ import annotations

from dataclasses import dataclass

import pytest

from harnyx_miner_sdk.safe_exec import SafeExecError, safe_exec


def test_safe_exec_runs_multistatement_python_with_functions_imports_and_control_flow() -> None:
    code = """
import statistics

ordered = sorted(values)

def describe(items):
    if not items:
        return {"count": 0, "mean": None}
    return {"count": len(items), "mean": statistics.mean(items)}

result = describe(ordered)
"""

    assert safe_exec(code, {"values": [6, 2, 4]}) == {"count": 3, "mean": 4}


def test_safe_exec_uses_one_fresh_namespace_for_variables_and_defined_functions() -> None:
    code = """
offset = 2

def adjusted():
    return [value + offset for value in values]

result = adjusted()
"""

    assert safe_exec(code, {"values": [1, 3]}) == [3, 5]


def test_safe_exec_does_not_expose_sdk_module_globals() -> None:
    with pytest.raises(NameError):
        safe_exec("result = JsonValue")


def test_safe_exec_detaches_input_variables_before_execution() -> None:
    variables = {"values": [1, {"label": "original"}]}

    result = safe_exec(
        "values[1]['label'] = 'changed'\nvalues.append(2)\nresult = values",
        variables,
    )

    assert result == [1, {"label": "changed"}, 2]
    assert variables == {"values": [1, {"label": "original"}]}


def test_safe_exec_returns_a_detached_json_result() -> None:
    result = safe_exec("value = [1, {'label': 'ok'}]\nresult = value")

    assert result == [1, {"label": "ok"}]


def test_safe_exec_requires_result_assignment() -> None:
    with pytest.raises(SafeExecError, match="assign result"):
        safe_exec("value = 1")


@pytest.mark.parametrize(
    "variables",
    [
        {"result": 1},
        {"__builtins__": {}},
        {"value": (1, 2)},
        {"value": float("inf")},
    ],
)
def test_safe_exec_rejects_reserved_names_and_non_json_variables(variables: object) -> None:
    with pytest.raises(SafeExecError):
        safe_exec("result = 1", variables)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "code",
    [
        "result = {1, 2}",
        "result = object()",
    ],
)
def test_safe_exec_rejects_non_json_results(code: str) -> None:
    with pytest.raises(SafeExecError, match="JSON-compatible"):
        safe_exec(code)


def test_safe_exec_propagates_generated_code_errors() -> None:
    with pytest.raises(ZeroDivisionError):
        safe_exec("result = 1 / 0")


def test_safe_exec_uses_a_private_copy_of_the_builtin_namespace() -> None:
    """Future failure: generated code must not mutate process-global builtins."""
    assert safe_exec("__builtins__['len'] = lambda value: 99\nresult = len([])") == 99
    assert len([]) == 0


class _DictSubclass(dict[str, object]):
    pass


@dataclass
class _Record:
    value: int


@pytest.mark.parametrize("variables", [_DictSubclass(value=1), {"value": _Record(1)}])
def test_safe_exec_requires_an_exact_json_variable_boundary(variables: object) -> None:
    with pytest.raises(SafeExecError):
        safe_exec("result = value", variables)  # type: ignore[arg-type]
