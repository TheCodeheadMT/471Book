# L28: Naive Bayes Classification

In the previous lesson, we used physical sensor data (numbers) to predict outcomes. But what happens when the intelligence we gather isn't numbers, but unstructured text?

Whether it's an intercepted enemy radio transmission, a secure network email, or a translated intelligence brief, the Air Force processes millions of words a day. To classify this text rapidly, we rely on **Bayes' Theorem**. The **Naive Bayes** classifier is incredibly fast, highly scalable, and forms the historical foundation for all modern natural language processing (NLP).

## The Core Concept: Bayes' Theorem

Bayes' Theorem allows us to update our beliefs based on new evidence. In classification, we want to find the probability of a specific class (like "Urgent") given the text evidence we just observed.

$$P(\text{Class} | \text{Evidence}) = \frac{P(\text{Evidence} | \text{Class}) \times P(\text{Class})}{P(\text{Evidence})}$$

* **$P(\text{Class})$ [The Prior]:** What is the baseline probability of this class before seeing any evidence? (e.g., 90% of all transmissions are Routine).

* **$P(\text{Evidence} | \text{Class})$ [The Likelihood]:** If the transmission *is* Urgent, how likely is it to contain the word "contact"?

* **$P(\text{Class} | \text{Evidence})$ [The Posterior]:** The final probability that the transmission is Urgent, given that we saw the word "contact".

   
> #### Check out the [**Bayes Simulator**](https://thecodeheadmt.github.io/CS471/BayesSim/index.html) got gain a better intution of how Bayes classification works.


---

## Text Classification

**Scenario:** Tactical Communications Triage. You are receiving thousands of short, intercepted radio transmissions. You need to automatically flag messages as "Routine" or "Urgent" so human analysts know what to read first.

### The Math: Naive Bayes for Text

To classify a full sentence, we look at the words $w_1, w_2, \dots, w_n$. The formula becomes:


$$P(\text{Class} | w_1, w_2, \dots, w_n) \propto P(\text{Class}) \prod_{i=1}^{n} P(w_i | \text{Class})$$

The $\propto$ symbol means "proportional to." We calculate this score for both "Urgent" and "Routine" and simply pick the class with the higher score.

> **Example 28.1 - Calculating the Posterior**
> Historically, 10% of transmissions are Urgent ($P(\text{U}) = 0.1$) and 90% are Routine ($P(\text{R}) = 0.9$).
> We intercept a new transmission: **"Enemy spotted"**.
> Let's look at our historical dictionary (Likelihoods):
> * $P(\text{``enemy''} | \text{U}) = 0.4$ (40% of Urgent messages use this word)
>
> * $P(\text{``spotted''} | \text{U}) = 0.3$
>
> * $P(\text{``enemy''} | \text{R}) = 0.01$ (1% of Routine messages use this word)
>
> * $P(\text{``spotted''} | \text{R}) = 0.05$
> 
> 
> **Calculate Urgent Score:** $0.1 \times 0.4 \times 0.3 = 0.012$
>
> **Calculate Routine Score:** $0.9 \times 0.01 \times 0.05 = 0.00045$
>
> **Result:** Since $0.012 > 0.00045$, the model confidently classifies the message as **Urgent**. Even though Routine messages are 9 times more common, the specific words provided overwhelming evidence to the contrary.

---

## Cyber Threats & The Independence Assumption

**Scenario:** Secure Network Phishing. You are filtering incoming emails on a secure Air Force network to detect malicious phishing attempts designed to steal credentials.

### The Math: The "Naive" Assumption

Why is it called *Naive* Bayes? Because the algorithm makes a massive, mathematically "naive" assumption: **It assumes every word in a sentence is completely independent of every other word.**

$$P(\text{"password"}, \text{"reset"} | \text{Malicious}) = P(\text{"password"} | \text{Malicious}) \times P(\text{"reset"} | \text{Malicious})$$

> **Example 28.2 - The Independence Failure**
> Let's look at the words "password" and "reset". In reality, if a phishing email contains the word "password", there is a massive probability the next word is "reset". They are highly correlated.
> However, Naive Bayes ignores this. It calculates the threat level of "password" (say, 80% malicious) and the threat level of "reset" (say, 70% malicious), and multiplies them together as if they were two entirely separate pieces of evidence.
>
> **Result:** This causes Naive Bayes to "double-count" the evidence, making its final probability scores overly confident (e.g., predicting a 99.99% probability of an attack). Yet, despite being mathematically overconfident, the *decision boundary* rarely changes—an email marked 99.99% malicious and an email marked 85% malicious are both correctly sent to the spam folder!

---

## Complex Problem: Zero-Frequency Events & Laplace Smoothing

**Scenario:** Adversary Communications. You are classifying intercepted text. What happens if the adversary uses a brand new code word (like "thunderbird") that has *never* appeared in your historical training data?

### The Math: The Zero Probability Problem

If a word has never been seen in a class, its historical probability is $0$. Because Naive Bayes uses multiplication, a single $0$ destroys the entire calculation:


$$P(\text{Urgent} | w_1, w_2, w_{new}) \propto 0.5 \times 0.3 \times \mathbf{0} = \mathbf{0}$$


Even if the rest of the message screams "Urgent", an unseen word acts as an absolute veto.

We fix this with **Laplace Smoothing** (also called Add-One smoothing). We mathematically pretend we have seen every possible word at least once.

$$P(w | C) = \frac{\text{count}(w, C) + \alpha}{\text{total words in } C + \alpha \times |V|}$$

* **$\alpha$ (Alpha):** The smoothing parameter (usually 1).
* **$|V|$:** The total size of our vocabulary.

> **Example 28.3 - Laplace Smoothing**
> Assume class "Urgent" has 100 total words. Our vocabulary $|V|$ contains 500 unique words.
> We encounter the unseen word "thunderbird".
> * **Without smoothing ($\alpha = 0$):** $P(\text{"thunderbird"} | U) = \frac{0}{100} = 0$
>
> * **With smoothing ($\alpha = 1$):** $P(\text{"thunderbird"} | U) = \frac{0 + 1}{100 + (1 \times 500)} = \frac{1}{600} \approx 0.0016$
> 
> 
> **Result:** The probability is tiny ($0.16\%$), representing that it is rare, but crucially, it is *not zero*. The math survives, and the rest of the sentence can still drive the classification.

---

## Knowledge Check

1. Why is the Naive Bayes algorithm considered "Naive"? Give an example of how this assumption might be violated in an intelligence report.
2. If an intercepted message contains the word "encryption", and this word has never appeared in your training dataset, what mathematical error will occur if you do not use smoothing?
3. Explain how Laplace (Add-One) smoothing prevents the mathematical failure identified in question 2.