##  Phase 2 — Black-Box Attack

### Overview

The black-box phase removes all access to model internals.  
Interaction is limited to a prediction API that returns only:

- A classification label
- A confidence score

This reflects a **realistic deployed machine learning system**, where the model itself is not exposed.

---

### Execution

The black-box attack script was executed:

```bash
python3 faze2.py
```

Output:

```text
[*] Starting black-box phase...
```

## Review-by-Review Evasion

Without access to model parameters, word effectiveness was estimated empirically through repeated API queries:

- A curated list of positive sentiment words was used

- Each word was appended individually to the review text

- The change in predicted probability was observed

- Words were ranked based on their impact

- The most effective words were added greedily until the classification flipped

## Output:

```text
Attacking review bb_0... Done (22 words, 73 queries total)
Attacking review bb_1... Done (40 words, 164 queries total)
Attacking review bb_2... Done (20 words, 235 queries total)
Attacking review bb_3... Done (17 words, 303 queries total)
Attacking review bb_4... Done (4 words, 358 queries total)
Attacking review bb_5... Done (38 words, 447 queries total)
Attacking review bb_6... Done (40 words, 538 queries total)
Attacking review bb_7... Done (4 words, 593 queries total)
Attacking review bb_8... Done (13 words, 657 queries total)
Attacking review bb_9... Done (6 words, 714 queries total)
```
