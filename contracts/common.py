"""Small, dependency-free helpers for strict JSON dataclass contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import MISSING, asdict, fields, is_dataclass
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

type JSONValue = (
    None | bool | int | float | str | list["JSONValue"] | dict[str, "JSONValue"]
)
type JSONMapping = dict[str, JSONValue]


class ContractError(ValueError):
    """Raised when a persisted payload violates a domain contract."""


class MappingView(Mapping[str, object]):
    """Read-only mapping compatibility view backed by ``to_mapping``."""

    def to_mapping(self) -> JSONMapping:
        raise NotImplementedError

    def __getitem__(self, key: str) -> object:
        return self.to_mapping()[key]

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.to_mapping()

    def get(self, key: str, default: object = None) -> object:
        return self.to_mapping().get(key, default)

    def __iter__(self):
        return iter(self.to_mapping())

    def __len__(self) -> int:
        return len(self.to_mapping())


def require_object(payload: object, *, context: str) -> Mapping[str, object]:
    if not isinstance(payload, Mapping):
        raise ContractError(f"{context} must be a JSON object")
    if any(not isinstance(key, str) for key in payload):
        raise ContractError(f"{context} keys must be strings")
    return payload


def as_json_value(value: object, *, field: str) -> JSONValue:
    """Validate and copy a JSON-compatible value."""
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ContractError(f"{field} must contain only finite numbers")
        return value
    if isinstance(value, (list, tuple)):
        return [as_json_value(item, field=f"{field}[]") for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ContractError(f"{field} keys must be strings")
        return {
            key: as_json_value(item, field=f"{field}.{key}")
            for key, item in value.items()
        }
    raise ContractError(f"{field} contains non-JSON value {type(value).__name__}")


def _decode(value: object, hint: object, field: str) -> object:
    origin, args = get_origin(hint), get_args(hint)
    if hint in (Any, object, JSONValue):
        return as_json_value(value, field=field)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        options = tuple(item for item in args if item is not type(None))
        if len(options) == 1:
            return _decode(value, options[0], field)
        for option in options:
            try:
                return _decode(value, option, field)
            except ContractError:
                pass
        raise ContractError(f"{field} has invalid type {type(value).__name__}")
    if origin is Literal:
        if value not in args:
            raise ContractError(f"{field} has unsupported value {value!r}")
        return value
    if isinstance(hint, type) and is_dataclass(hint):
        parser = getattr(hint, "from_mapping", None)
        return (
            parser(value, context=field)
            if parser is not None
            else from_dict(hint, value, context=field)
        )
    if origin in (dict, Mapping):
        data = require_object(value, context=field)
        key_type, value_type = args or (str, JSONValue)
        return {
            _decode(key, key_type, f"{field}.key"): _decode(
                item, value_type, f"{field}.{key}"
            )
            for key, item in data.items()
        }
    if origin in (list, tuple):
        if not isinstance(value, list):
            raise ContractError(f"{field} must be a list")
        item_type = args[0] if args else Any
        converted = [_decode(item, item_type, f"{field}[]") for item in value]
        return tuple(converted) if origin is tuple else converted
    if hint is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ContractError(f"{field} must be a number")
        if not math.isfinite(result := float(value)):
            raise ContractError(f"{field} must be finite")
        return result
    if hint is int and (not isinstance(value, int) or isinstance(value, bool)):
        raise ContractError(f"{field} must be an integer")
    if hint is bool and not isinstance(value, bool):
        raise ContractError(f"{field} must be a boolean")
    if hint is str and not isinstance(value, str):
        raise ContractError(f"{field} must be a string")
    return value


def from_dict[T](
    cls: type[T],
    payload: object,
    *,
    context: str,
    headers: Mapping[str, object] | None = None,
    ignore_unknown: bool = False,
) -> T:
    """Strictly construct a dataclass using its fields and type annotations."""
    data = require_object(payload, context=context)
    definitions = fields(cls)
    header_values = dict(headers or {})
    allowed = {item.name for item in definitions} | set(header_values)
    if not ignore_unknown and (unknown := sorted(set(data) - allowed)):
        raise ContractError(f"{context} contains unknown fields: {', '.join(unknown)}")
    for name, expected in header_values.items():
        if data.get(name) != expected:
            raise ContractError(f"Unsupported {context} {name}: {data.get(name)!r}")
    hints = get_type_hints(cls)
    values: dict[str, object] = {}
    for definition in definitions:
        if definition.name not in data:
            if type(None) in get_args(hints[definition.name]):
                values[definition.name] = None
            elif (
                definition.default is MISSING and definition.default_factory is MISSING
            ):
                raise ContractError(
                    f"{context} is missing required field '{definition.name}'"
                )
            continue
        value = _decode(
            data[definition.name],
            hints[definition.name],
            f"{context}.{definition.name}",
        )
        if definition.metadata.get("non_empty") and not value:
            raise ContractError(f"{context}.{definition.name} must not be empty")
        minimum = definition.metadata.get("minimum")
        if minimum is not None and value is not None and value < minimum:
            raise ContractError(f"{context}.{definition.name} must be >= {minimum}")
        values[definition.name] = value
    try:
        return cls(**values)
    except (TypeError, ValueError) as exc:
        raise ContractError(str(exc)) from exc


def to_dict(
    instance: object,
    *,
    headers: Mapping[str, JSONValue] | None = None,
    compact: bool = False,
    omit_empty: frozenset[str] = frozenset(),
    late_optional: bool = False,
) -> JSONMapping:
    """Serialize a dataclass from ``asdict`` while preserving wire ordering."""
    values = asdict(instance)  # type: ignore[arg-type]
    definitions = list(fields(instance))  # type: ignore[arg-type]
    if late_optional:
        definitions.sort(key=lambda item: item.default is None)
    payload: JSONMapping = dict(headers or {})
    for definition in definitions:
        value = values[definition.name]
        if (compact and value is None) or (definition.name in omit_empty and not value):
            continue
        payload[definition.name] = as_json_value(value, field=definition.name)
    return payload
