"""Safety layer: disclaimers, refusal/guardrail helpers, output framing.

Every model-facing path in this project routes through here so that the
uncertainty + clinician-review framing is applied consistently and cannot
be silently dropped.
"""
