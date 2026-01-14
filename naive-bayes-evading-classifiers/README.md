
#  Naive Bayes Inference-Time Evasion (GoodWords Attack) (HTB)

This repository demonstrates **inference-time evasion attacks** against a deployed **Naive Bayes text classifier** using the classic **GoodWords attack** technique.

The core idea is simple:

> We manipulate only the input text at prediction time by adding carefully chosen words, causing the model to misclassify the input **without modifying the model or training data**.

This project explores the attack under two realistic threat models:
- **White-box access** (full model visibility)
- **Black-box access** (API-only interaction)

---

##  Goal of the Exercise

Given a text classifier (spam/ham or sentiment analysis):

- Keep the **original message intact**
- Add a **limited number of words**
- Flip the model’s prediction
- Stay within a strict word budget

This simulates how adversaries evade deployed ML systems in the real world.

---

##  Why This Works (Conceptual Overview)

Naive Bayes classifiers make several simplifying assumptions:

- Each word contributes **independently**
- Evidence is **summed additively** (in log space)
- No semantic or contextual consistency is enforced

Because of this, attackers can exploit the math:

> If enough words statistically favor a target class, they can overpower strong opposing signals — even if the original message still clearly contains spam or negative sentiment.

This weakness is the foundation of the **GoodWords attack**.

---

##  White-Box Attack

### Threat Model
- Full access to the trained model
- Vocabulary, feature probabilities, and class labels are visible

This setting is useful for understanding **optimal evasion**, even though it is less realistic for production systems.

### Attack Strategy
1. Download the trained Naive Bayes model
2. Extract per-word conditional probabilities
3. Score words by how strongly they favor the target class
4. Greedily append the highest-scoring words to the input
5. Stop once the prediction flips

Because we have direct access to the model internals, word selection is **precise and efficient**, requiring very few additions.

 **White-box implementation shown below**

---

##  Black-Box Attack

### Threat Model
- No access to model internals
- Interaction limited to a prediction API
- Only labels and confidence scores are observable

This reflects **realistic deployed ML systems**.

### Attack Strategy
1. Start with a curated vocabulary of candidate words
2. Add words one at a time and observe probability changes
3. Measure each word’s impact on the target class
4. Rank words by effectiveness
5. Greedily append the strongest words until the label flips

Even without internals, **probability outputs leak enough information** to guide a successful attack.

 **Black-box implementation shown below**

---

##  White-Box vs Black-Box Summary

| Aspect | White-Box | Black-Box |
|------|----------|----------|
| Model access | Full | None |
| Word scoring | Exact | Estimated via probing |
| Query count | Low | Higher |
| Realism | Lower | Higher |
| Core weakness exploited | Same | Same |

Both attacks succeed for the same underlying reason:
**Naive Bayes cannot reason about intent or context — only statistics.**

---

##  Defensive Takeaways

This exercise highlights why simple probabilistic classifiers are vulnerable in adversarial settings:

- Conditional independence assumptions are exploitable
- Static word probabilities leak attack surface
- Confidence outputs enable black-box probing
- No semantic coherence checks exist

Modern defenses include:
- Context-aware embeddings
- Feature interaction modeling
- Ensemble classifiers
- Adversarial training
- Output confidence obfuscation

---

##  Key Takeaway

> If a model relies on independent feature contributions without understanding meaning, attackers can always manipulate the math at inference time.

---

##  Repository Structure

```text
whitebox/
  └── whitebox_attack.py

blackbox/
  └── blackbox_attack.py
```

