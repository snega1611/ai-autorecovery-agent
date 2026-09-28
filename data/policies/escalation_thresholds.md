# Escalation Thresholds

Escalate to a human reviewer whenever any of the following occur, regardless
of request type:

- Two or more different candidate values are found for the same field
  (conflicting evidence).
- No reliable evidence is found for a required field after searching the
  full transcript.
- The customer's tone or wording indicates distress, a complaint, or a
  threat to escalate to a regulator or legal action — these always go to a
  human regardless of whether the requested data change itself is clear.
- The system's confidence in a recovered value is below the threshold used
  by the deterministic validator (see validate_field tool).

Never auto-apply a recovery solely because the agent's language model
"sounds confident." Confidence must be established by the deterministic
validator based on evidence, not by the model's own self-assessment.
