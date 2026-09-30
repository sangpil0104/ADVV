def decide(checks: dict) -> str:
    completed = [checks.get(name, {}) for name in ("physical", "semantic")]
    answers = [
        c.get("response", {}).get("answer") if c.get("status") == "completed" else None for c in completed
    ]
    if "NO" in answers:
        return "rejected"
    if answers == ["YES", "YES"]:
        return "accepted"
    if any(c.get("status") == "error" for c in completed):
        return "verification_error"
    if "UNCERTAIN" in answers:
        return "uncertain"
    return "pending"
