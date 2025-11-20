#  Red Teaming ML — Practical Attacks Writeup  
*(HTB Academy)*

This writeup documents three hands-on attacks performed against machine-learning systems:

- **ML01 — Input Manipulation**
- **ML02 — Data Poisoning**
- **ML05 — Model Theft**

All attacks were executed against a vulnerable ML web application provided by HackTheBox Academy.

---

#  Overview

Machine learning models are not only vulnerable to traditional software exploits —  
they also fail in predictable ways when an attacker manipulates:

- **Their inputs**  
- **Their training data**  
- **Their output**  
- **Their architecture or API**

In this lab, we exploited all three categories.

---

#  ML01 — Input Manipulation Attack  
### *“Take a spam message and trick the classifier into saying it’s ham.”*

The application provided a fixed spam message:

<img width="1465" height="846" alt="image" src="https://github.com/user-attachments/assets/c4e89fdc-e4f2-4e6c-89ca-6353ef01dcd2" />


**Goal:** append text that shifts the classification toward **HAM (legitimate)**.

I added a long block of neutral, harmless Lorem Ipsum-style text:

````bash
Congratulations! You've won a $1000 Walmart gift card. Go to https://bit.ly/3YCN7PF to claim now. One ring to rule them all, one ring to find them, one ring to bring them all and in the darkness, bind them! This is the one ring. Forged by the Dark Lord Sauron.I dont really know what to write in here, so I am just doing random things now.
````

The classifier now saw far more “ham-like” vocabulary than spam indicators, and the final output flipped to **HAM**.

This demonstrates how fragile bag-of-words / TF-IDF models are when an attacker injects enough neutral or benign tokens.

###  Why This Works
- The classifier counts word frequency.
- “Normal” words overpower the spam-related terms.
- No semantic understanding — just statistics.
- Result: **By appending harmless text, attackers can bypass spam filters entirely.**

<img width="1488" height="595" alt="image" src="https://github.com/user-attachments/assets/492c6618-dc7b-49d5-9828-28ad4854fe35" />


# ML02 — Data Poisoning Attack

In this scenario, we downloaded the training dataset from the vulnerable web app and created a poisoned subset.

We extracted the first 100 rows:

```bash
head -n 101 train.csv > poison-student.csv
````

Then we manipulated the labels inside poison-student.csv:

- Changed many legitimate ham → spam (using 'Find/Replace'

- Added misleading spam labels to benign messages

- Kept only a few normal ham samples

- This intentionally skewed the distribution.

After modifying main.py to train on our poisoned data:
```python
model = train("./poison-student.csv")
acc = evaluate(model, "./train.csv")
```

The classifier became almost useless — a direct result of our poisoned labels.

 Ham: 62.32%
Spam: 37.68%


<img width="1479" height="715" alt="image" src="https://github.com/user-attachments/assets/9510ce83-6519-44d5-8912-79a1e7071c16" />

# Why This Works

- ML models depend heavily on clean, representative training data.

- Poisoned labels cause:

- incorrect decision boundaries

- biased behavior

- huge performance drops

- potential backdoors or targeted misclassifications

- Even a small poisoned subset can destabilize the entire model.

---

# ML05 — Model Theft Attack

In the final scenario, the vulnerable ML web app accidentally exposed an undocumented endpoint.

While inspecting the page source, we found a commented-out reference to:

````bash
model/
````

<img width="1472" height="448" alt="image" src="https://github.com/user-attachments/assets/65ea890d-fcb6-4f6d-8775-6ebbdd566e3d" />

Using a simple wget request, we pulled the model from the server:

```bash
─╼ [★]$ wget http://83.136.254.84:52446/model
--2025-11-19 21:54:55--  http://83.136.254.84:52446/model
Connecting to 83.136.254.84:52446... connected.
HTTP request sent, awaiting response... 200 OK
Length: 1757818 (1.7M) [application/octet-stream]
Saving to: ‘model’

model               100%[===================>]   1.68M  3.59MB/s    in 0.5s    

2025-11-19 21:54:55 (3.59 MB/s) - ‘model’ saved [1757818/1757818]

```
```bash
[★]$ md5sum model
8007cd6c209a40399cf3ca82dd7db02c  model
````
---

# Why This Works

This attack is a real-world example of ML05 — Model Theft, one of the OWASP ML Top 10.

Machine-learning models are valuable intellectual property.
If an attacker can download them:

- They can reverse-engineer training behavior

- Perform membership inference or model inversion

- Replicate the model and bypass API usage quotas

- Analyze the model for weaknesses

- Inject backdoored versions if upload endpoints exist

- A single forgotten debug endpoint is enough to leak the entire ML system.

---

# Conclusion

These labs showed how easy it is to break ML systems without attacking the server — just the model itself.

- Input Manipulation: Adding harmless text was enough to flip a spam message into “ham.”

- Data Poisoning: Corrupting a small slice of the training data tanked the model’s accuracy.

- Model Theft: A hidden endpoint exposed the entire trained model for download.

Overall: ML models are fragile, easy to influence, and must be protected just like any other part of an application


















 
