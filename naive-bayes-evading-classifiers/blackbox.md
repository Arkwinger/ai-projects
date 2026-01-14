##  Phase 2 — Black-Box Attack

### Overview

The black-box phase removes all access to model internals.  
Interaction is limited to a prediction API that returns only:

- A classification label
- A confidence score

This reflects a **realistic deployed machine learning system**, where the model itself is not exposed.

---

##  Script — Black-Box GoodWords Attack (`faze2.py`)

The following script implements the **black-box GoodWords attack**.  
Unlike the white-box phase, this attack has **no access to model internals** and relies entirely on observing prediction outputs from the API.

Word effectiveness is estimated empirically by measuring how individual word additions change the model’s confidence score.

```python
#!/usr/bin/env python3
import os
import requests
import numpy as np
from typing import List, Dict, Tuple

SEED = 1337
np.random.seed(SEED)

BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")


class BlackBoxAttacker:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.query_count = 0

    def predict(self, text: str) -> Dict:
        self.query_count += 1
        r = requests.post(
            f"{self.base_url}/predict",
            json={"text": text},
            timeout=15
        )
        r.raise_for_status()
        return r.json()

    def estimate_word_impacts(
        self, base_text: str, vocabulary: List[str]
    ) -> List[Tuple[str, float]]:
        base_pred = self.predict(base_text)
        base_prob = base_pred["positive_probability"]

        impacts = []
        for word in vocabulary:
            augmented = base_text + " " + word
            pred = self.predict(augmented)
            impact = pred["positive_probability"] - base_prob
            impacts.append((word, impact))

        impacts.sort(key=lambda x: x[1], reverse=True)
        return impacts

    def attack_review(self, text: str, max_words: int) -> Tuple[str, int]:
        positive_vocabulary = [
            "excellent", "amazing", "wonderful", "fantastic", "brilliant",
            "outstanding", "superb", "magnificent", "perfect", "exceptional",
            "masterpiece", "genius", "beautiful", "stunning", "remarkable",
            "awesome", "incredible", "phenomenal", "spectacular", "marvelous",
            "great", "good", "love", "loved", "best", "favorite", "enjoyed",
            "recommend", "highly", "definitely", "must", "liked", "appreciate",
            "admire", "adore", "enjoy", "compelling", "engaging", "captivating",
            "mesmerizing", "powerful", "touching", "moving", "inspiring",
            "uplifting", "heartwarming", "clever", "witty", "funny", "hilarious",
            "entertaining"
        ]

        impacts = self.estimate_word_impacts(text, positive_vocabulary[:50])

        augmented = text
        for i, (word, _) in enumerate(impacts, 1):
            if i > max_words:
                break

            augmented = augmented + " " + word
            pred = self.predict(augmented)

            if pred["label"] == "positive":
                return augmented, i

        # Fallback: repeat strongest words if flip did not occur
        if impacts:
            top_words = [w for w, _ in impacts[:10]]
            repeated = []
            while len(repeated) < max_words:
                repeated.extend(top_words)
            augmented = text + " " + " ".join(repeated[:max_words])

        return augmented, max_words

    def solve_blackbox(self) -> Dict:
        print("\n[*] Starting black-box phase...")

        r = requests.get(f"{self.base_url}/challenge/blackbox", timeout=15)
        r.raise_for_status()
        challenge = r.json()

        reviews = challenge["reviews"]
        max_words = challenge["max_added_words"]

        solutions = []
        for review in reviews:
            print(f"  Attacking review {review['id']}...", end=" ")
            augmented, words_used = self.attack_review(
                review["text"], max_words
            )
            solutions.append(
                {
                    "id": review["id"],
                    "augmented_text": augmented,
                }
            )
            print(
                f"Done ({words_used} words, {self.query_count} queries total)"
            )

        r = requests.post(
            f"{self.base_url}/submit/blackbox",
            json={"solutions": solutions},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()

        if "results" in result:
            successes = sum(
                1 for r in result["results"] if r.get("success", False)
            )
            print(
                f"[+] Black-box phase: {successes}/{len(reviews)} completed"
            )

        return result


def main():
    attacker = BlackBoxAttacker(BASE_URL)
    result = attacker.solve_blackbox()

    if result.get("flag"):
        print("\n" + "=" * 60)
        print("[+] SUCCESS! All challenges completed!")
        print(f"[+] Flag: {result['flag']}")
        print("=" * 60)
    else:
        print("[-] Black-box phase failed")


if __name__ == "__main__":
    main()
```

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
