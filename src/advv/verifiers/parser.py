import json

from jsonschema import Draft7Validator

from ..errors import ResponseError


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ResponseError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value):
    raise ResponseError(f"Non-JSON numeric constant: {value}")


def parse_response(text: str, schema: dict) -> dict:
    try:
        value = json.loads(text.strip(), object_pairs_hook=_pairs, parse_constant=_constant)
    except (ValueError, TypeError) as exc:
        raise ResponseError(f"Invalid JSON completion: {exc}") from exc
    errors = sorted(Draft7Validator(schema).iter_errors(value), key=lambda e: str(e.path))
    if errors:
        raise ResponseError("; ".join(error.message for error in errors))
    return value
