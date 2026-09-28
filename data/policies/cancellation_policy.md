# Policy Cancellation Rules

A cancellation request may be auto-processed only when ALL of the following
are true:

1. A valid policy number is identified (format: one letter followed by 4-6
   digits, e.g. `P8921`), matched either directly from the customer's speech
   or via a phone-number lookup against the customer database.
2. An effective date is stated, and it is today or in the future.
3. The customer stated the policy number and/or effective date exactly once,
   with no correction or contradictory value elsewhere in the call.

If the customer corrects themselves (e.g. states one policy number, then
later states a different one), the request MUST be escalated to a human
agent for manual verification. Do not guess which value is correct.

If the customer explicitly states they do not have or do not remember their
policy number, and no phone-number match is available, the request cannot be
auto-processed and should be escalated for a manual follow-up contact.

Cancellations effective more than 30 days in the future should still be
auto-processed if the evidence is otherwise clear; there is no policy reason
to delay these.
