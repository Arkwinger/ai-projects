#  Whitebox GoodWords Attack — Naive Bayes Inference-Time Evasion 

This repository documents a complete **inference-time evasion attack** against a Naive Bayes text classifier using the classic **GoodWords attack** technique.

Both **white-box** and **black-box** threat models are demonstrated, following the exact execution flow used during the assessment, from environment setup to successful flag retrieval.

---

## Environment Setup

A dedicated project directory and Python virtual environment were created to isolate dependencies.

```bash
mkdir goodwords_attack
cd goodwords_attack
python3 -m venv venv
source venv/bin/activate
```

Required dependencies were installed:

```bash
pip3 install scikit-learn numpy requests
```

The target service URL was exported as an environment variable:

```bash
export BASE_URL="http://94.237.120.112:51238"
```

Service availability was verified:

```bash
curl -s "$BASE_URL/health" | jq
```

```json
{
  "service": "skills_assessment_lab",
  "status": "healthy"
}
```

## Overview

The white-box phase assumes full access to the trained model, including:

- Naive Bayes classifier

- CountVectorizer

- Feature names (vocabulary)

- Class labels

This allows direct inspection of word-level probabilities and enables optimal evasion with minimal trial and error.

## Script — White-Box GoodWords Attack (`faze1.py`)

The following script implements the **white-box GoodWords attack** described above.  
It downloads the trained Naive Bayes model, analyzes word-level probabilities, and greedily appends statistically favorable words until each review flips to the target sentiment.

```python
#!/usr/bin/env python3
import os
import requests
import pickle
import numpy as np
from typing import List, Dict, Tuple

SEED = 1337
np.random.seed(SEED)

BASE_URL = os.environ.get("BASE_URL")


class WhiteBoxAttacker:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.classes = None

    def download_model(self):
        print("[*] Downloading model...")
        r = requests.get(f"{self.base_url}/model/download", timeout=15)
        r.raise_for_status()

        with open("/tmp/model.pkl", "wb") as f:
            f.write(r.content)

        with open("/tmp/model.pkl", "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["classifier"]
        self.vectorizer = bundle["vectorizer"]
        self.feature_names = bundle["feature_names"]
        self.classes = bundle["classes"]

        print(f"[+] Model loaded: {len(self.feature_names)} features")
        print(f"[+] Classes: {self.classes}")

    def calculate_word_scores(self, target_class: str) -> List[Tuple[str, float]]:
        target_idx = self.classes.index(target_class)
        other_idx = 1 - target_idx

        scores = []
        for i, feature in enumerate(self.feature_names):
            # Skip multi-word features (ngrams)
            if " " in feature:
                continue

            target_prob = np.exp(self.model.feature_log_prob_[target_idx][i])
            other_prob = np.exp(self.model.feature_log_prob_[other_idx][i])

            score = target_prob / (other_prob + 1e-10)
            scores.append((feature, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def attack_review(self, text: str, target: str, max_words: int) -> Tuple[str, int]:
        word_scores = self.calculate_word_scores(target)

        augmented = text
        for num_words in range(1, max_words + 1):
            words_to_add = [w for w, _ in word_scores[:num_words]]
            augmented = text + " " + " ".join(words_to_add)

            vec = self.vectorizer.transform([augmented])
            prediction = self.model.predict(vec)[0]

            if prediction == target:
                return augmented, num_words

        return augmented, max_words

    def solve_whitebox(self) -> Dict:
        print("\n[*] Starting white-box phase...")

        r = requests.get(f"{self.base_url}/challenge/whitebox", timeout=15)
        r.raise_for_status()
        challenge = r.json()

        reviews = challenge["reviews"]
        max_words = challenge["max_added_words"]

        self.download_model()

        solutions = []
        for review in reviews:
            print(f"  Attacking review {review['id']}...", end=" ")
            augmented, words_used = self.attack_review(
                review["text"],
                review["target_sentiment"],
                max_words,
            )
            solutions.append(
                {
                    "id": review["id"],
                    "augmented_text": augmented,
                }
            )
            print(f"Done ({words_used} words)")

        r = requests.post(
            f"{self.base_url}/submit/whitebox",
            json={"solutions": solutions},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json()

        if "results" in result:
            successes = sum(1 for r in result["results"] if r.get("success", False))
            print(f"[+] White-box phase: {successes}/{len(reviews)} completed")

        return result


def main():
    attacker = WhiteBoxAttacker(BASE_URL)
    result = attacker.solve_whitebox()

    if result.get("phase_complete", False):
        print("[+] White-box phase completed successfully!")
    else:
        print("[-] White-box phase failed")


if __name__ == "__main__":
    main()
```

## Execution

```bash
python3 faze1.py
````

Output: 

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

## Conceptual Explanation

Naive Bayes assumes conditional independence between words and sums their contributions to reach a final decision.

Because the model does not evaluate semantic consistency or intent, a small number of statistically strong words can overwhelm the original sentiment signal.

With full model access, evasion becomes deterministic, fast, and highly reliable.




















