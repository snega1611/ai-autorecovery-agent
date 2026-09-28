# Identity Verification Rules

Before any recovery is auto-applied, the customer's identity must be
resolvable to exactly one customer record, either by:

1. A valid policy number, or
2. A phone number that matches exactly one customer in the customer
   database.

If a phone-number lookup matches zero or more than one customer, identity is
not resolved and the request must be escalated. Never auto-apply a recovery
against an unresolved identity, even if the requested change itself seems
unambiguous.
