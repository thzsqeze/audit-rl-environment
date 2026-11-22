"""Minimal offline JSON Schema (draft 2020-12) validator for the schemas in `schemas/`.

The runtime stays stdlib-only, so instead of the `jsonschema` package this module implements exactly the
subset of keywords the published schemas use:

  type (incl. lists of types), enum, const, properties, required, additionalProperties (bool or schema),
  items, minItems, maxItems, uniqueItems, minLength, maxLength (Unicode code points), pattern (re.search),
  minimum, maximum, anyOf, oneOf, $ref (local "#/$defs/x" and "<file>.schema.json[#/$defs/x]"), $defs.

Any other keyword in a schema is rejected when the schema is loaded, so a schema can never silently
use a keyword this validator ignores. JSON types follow JSON semantics: `true` is a boolean, not an
integer; `1.0` parsed by `json` is a number, not an integer.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
SCHEMA_NAMES = ("manifest", "policy", "ledger_row", "invoice", "payment", "fx_row", "findings", "action",
                "observation_reset", "observation_step", "truth", "trajectory_line")

_ANNOTATIONS = {"$schema", "$id", "title", "description", "$comment", "examples", "default"}
_KEYWORDS = _ANNOTATIONS | {
    "type", "enum", "const", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
    "uniqueItems", "minLength", "maxLength", "pattern", "minimum", "maximum", "anyOf", "oneOf", "$ref", "$defs",
}


def _json_type_ok(value: Any, name: str) -> bool:
    if name == "null":
        return value is None
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":  # NaN/Infinity are not JSON numbers (json.loads accepts them as an extension)
        return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    if name == "string":
        return isinstance(value, str)
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    raise ValueError(f"unsupported JSON type {name!r}")


class SchemaStore:
    """Loads all schemas of a directory and validates instances against them."""

    def __init__(self, directory: str | Path = SCHEMA_DIR):
        self.directory = Path(directory)
        self._cache: dict[str, dict] = {}

    def load(self, name: str) -> dict:
        file = name if name.endswith(".schema.json") else f"{name}.schema.json"
        if file not in self._cache:
            with open(self.directory / file, encoding="utf-8") as fh:
                schema = json.load(fh)
            _check_keywords(schema, file)
            self._cache[file] = schema
        return self._cache[file]

    def errors(self, instance: Any, name: str) -> list[str]:
        """All validation errors (empty list = valid). Each error starts with a JSON pointer."""
        root = self.load(name)
        out: list[str] = []
        self._validate(instance, root, root, "", out)
        return out

    def validate(self, instance: Any, name: str) -> None:
        errs = self.errors(instance, name)
        if errs:
            raise ValueError(f"{name}: " + "; ".join(errs[:5]))

    # ------------------------------------------------------------------ internals
    def _resolve(self, ref: str, root: dict) -> tuple[dict, dict]:
        file, _, pointer = ref.partition("#")
        target_root = self.load(file) if file else root
        node: Any = target_root
        for part in [p for p in pointer.split("/") if p]:
            node = node[part]
        return node, target_root

    def _validate(self, value: Any, schema: dict | bool, root: dict, path: str, out: list[str]) -> None:
        if schema is True:
            return
        if schema is False:
            out.append(f"{path or '/'}: no value is allowed here")
            return
        where = path or "/"
        if "$ref" in schema:
            target, target_root = self._resolve(schema["$ref"], root)
            self._validate(value, target, target_root, path, out)
        if "type" in schema:
            types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
            if not any(_json_type_ok(value, t) for t in types):
                out.append(f"{where}: expected {'|'.join(types)}, got {type(value).__name__}")
                return
        if "enum" in schema and not any(_same(value, e) for e in schema["enum"]):
            out.append(f"{where}: {value!r} is not one of {schema['enum']}")
        if "const" in schema and not _same(value, schema["const"]):
            out.append(f"{where}: must equal {schema['const']!r}")
        if isinstance(value, str):
            if "minLength" in schema and len(value) < schema["minLength"]:
                out.append(f"{where}: shorter than {schema['minLength']}")
            if "maxLength" in schema and len(value) > schema["maxLength"]:
                out.append(f"{where}: longer than {schema['maxLength']}")
            if "pattern" in schema and not _pattern_ok(schema["pattern"], value):
                out.append(f"{where}: {value!r} does not match {schema['pattern']}")
        if _json_type_ok(value, "number"):
            if "minimum" in schema and value < schema["minimum"]:
                out.append(f"{where}: below minimum {schema['minimum']}")
            if "maximum" in schema and value > schema["maximum"]:
                out.append(f"{where}: above maximum {schema['maximum']}")
        if isinstance(value, list):
            if "minItems" in schema and len(value) < schema["minItems"]:
                out.append(f"{where}: fewer than {schema['minItems']} items")
            if "maxItems" in schema and len(value) > schema["maxItems"]:
                out.append(f"{where}: more than {schema['maxItems']} items")
            if schema.get("uniqueItems"):
                seen = [json.dumps(v, sort_keys=True) for v in value]
                if len(seen) != len(set(seen)):
                    out.append(f"{where}: items are not unique")
            if "items" in schema:
                for i, item in enumerate(value):
                    self._validate(item, schema["items"], root, f"{path}/{i}", out)
        if isinstance(value, dict):
            for key in schema.get("required", []):
                if key not in value:
                    out.append(f"{where}: missing required property {key!r}")
            props = schema.get("properties", {})
            for key, item in value.items():
                if key in props:
                    self._validate(item, props[key], root, f"{path}/{key}", out)
                elif "additionalProperties" in schema:
                    self._validate(item, schema["additionalProperties"], root, f"{path}/{key}", out)
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                passing = 0
                for option in schema[keyword]:
                    sub: list[str] = []
                    self._validate(value, option, root, path, sub)
                    passing += not sub
                if keyword == "anyOf" and passing == 0:
                    out.append(f"{where}: matches none of anyOf")
                if keyword == "oneOf" and passing != 1:
                    out.append(f"{where}: matches {passing} of oneOf (exactly 1 required)")


def _pattern_ok(pattern: str, value: str) -> bool:
    """ECMA-262 semantics for the anchors used here: '$' matches only at the very end (Python's '$' would
    also match before a trailing newline, so 'KZT\\n' would pass '^[A-Z]{3}$')."""
    ecma = re.sub(r"(?<!\\)\$", r"\\Z", pattern)
    return re.search(ecma, value) is not None


def _same(a: Any, b: Any) -> bool:
    """JSON equality: booleans never equal numbers."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    return a == b


def _check_keywords(schema: Any, where: str) -> None:
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise ValueError(f"{where}: schema must be an object or boolean")
    unknown = set(schema) - _KEYWORDS
    if unknown:
        raise ValueError(f"{where}: unsupported schema keywords {sorted(unknown)}")
    for key in ("properties", "$defs"):
        for name, sub in schema.get(key, {}).items():
            _check_keywords(sub, f"{where}/{key}/{name}")
    for key in ("items", "additionalProperties"):
        if key in schema:
            _check_keywords(schema[key], f"{where}/{key}")
    for key in ("anyOf", "oneOf"):
        for i, sub in enumerate(schema.get(key, [])):
            _check_keywords(sub, f"{where}/{key}/{i}")
