# Stage 12 — Final presentation and system demonstration

Weight: 20%. Group presentation, marked individually. Dates in the brief: 7, 10, and 11 October 2026.

The audience is a non-technical credit committee. The slides should sound like a lending recommendation, not a model tutorial. Algorithm names appear once, when the committee asks what sits behind the score. This document is the slide outline and the speaker script. Build the slide file from it; do not read the technical report aloud.

## What each person must be ready to do

- Explain the business problem and why unpaid loans matter.
- Explain the tool in everyday language.
- Show the patterns: who defaults more often, and by how much.
- Show how the tool changes the approval process.
- Run the screen live.
- Tie every recommendation to a number from the analysis.
- Answer a follow-up about their own contribution.

Avoid, unless asked: cross-validation, one-hot encoding, Gini, hyperparameters, pipeline objects. If asked “how does it work?”, answer with the inputs, the risk score, and the review rule, then offer the model name.

## Slide outline

### 1. Title

Loan application review: spotting likely defaults before we release funds.

Names, course, date. No model name on this slide.

### 2. The problem in the business

About 12 in every 100 loans in this book default (11.6%, 29,653 of 255,347).

A rule that approves everyone looks accurate on paper, because 88 in 100 loans repay. That rule also misses every default.

We need a first look, while the application is still open, that sends the riskier files to a person.

### 3. Who uses it, and what they decide

The user is the credit officer.

The tool does three jobs:

- Supports an approval when risk looks low.
- Puts a file in a review queue when risk looks high.
- Gives pricing a probability they can see.

It does not decline a customer on its own.

### 4. What we looked at

Public loan-application data, 255,347 applications, free to use (CC0), from the Loan Default Prediction Dataset on Kaggle.

For each application we know age, income, job situation, credit score, existing debts, and the loan they asked for (amount, rate, term, purpose), plus whether that loan later defaulted.

We do not use the application id. A new customer does not have one that would help.

### 5. What the data shows (the insight slides)

Use three pictures or three big numbers. Suggested charts, computed from the file:

**Age.** Default rate is 21% for ages 18–25 and 6% for ages 56–69.

**Price of the loan.** Default rate is 7% when the rate is 2–8%, and 18% when the rate is 20–25%.

**Size of the loan versus income.** In the quarter where the loan is largest relative to income, 19% default. In the quarter where it is smallest, 8% default.

One more line if there is room: unemployed applicants default at 14%, full-time applicants at 9%. How long the loan runs (1 year versus 5 years) does not change the default rate. We still ask for the term, because it changes the monthly payment.

Say the sensitive point in one sentence: age and marital status predict default here. A live lending policy would need a fairness check before those facts were allowed to drive a decline.

### 6. What we recommend the committee do

1. Use the score to **build a review queue**, not to auto-decline.
2. At the chosen setting, the queue catches about **half of the loans that will default** (recall 49%).
3. About **30%** of the files in that queue really default. The rest repay. That is much higher than the 12% base rate, so the queue is enriched, and it is still a human review.
4. Prioritise review for younger applicants, higher interest rates, and loans that are large relative to income. Those three patterns show up no matter which method we used.
5. Keep a person on the decision. The screen says the result supports the approval and does not replace it.

Do not promise that the model “prevents defaults.” It orders the book so staff time goes to riskier files first.

### 7. How good is the ordering

Plain version:

- If we ranked applications at random, a list of the riskiest names would be only 12% actual defaults.
- Our ranking lifts that to about 33% average precision. (If the phrase “average precision” will not land, say: “the high-risk end of the list contains far more true defaults than a random list.”)
- The ordering score on unseen applications is 0.76 on a scale where 0.50 is a coin toss (ROC-AUC). Use this only if someone asks for a single quality number.

We tried four methods. A simple scored checklist (logistic regression) was already strong. A more flexible method (gradient boosting) was a little better and is the one in the tool. A single decision tree was weaker. We did not pick the method that memorised old files; we picked the one that held up on applications it had not been shown during the choice.

### 8. Live demonstration

Script, about three minutes:

1. Open the review screen. Read the first sentence: this is a check before approval.
2. Submit the example already filled in (age 35, income 72,000, loan 15,000, score 680). Read the heading and the probability. Point at the line that says this supports the decision.
3. Change the example to a higher-risk case: age 22, income 30,000, loan amount 180,000, interest rate 22%, unemployed, no co-signer. Submit again. The heading should move to “Send this application for a closer review,” with a higher probability.
4. Set age to 10 and income to 0. Show that the screen refuses the application and names the fields. Say we would rather block a bad form than invent a score.
5. Stop. Do not open the API docs unless a marker asks.

If the service is down, the screen itself tells you the start command. Start the API before the presentation:

```bash
uvicorn backend.main:app --reload
streamlit run frontend/app.py
```

### 9. What this changes in operations

- Same inputs the officer already types.
- One button, one probability, one action: review or lower risk.
- Invalid forms do not get a score.
- The preparation of the data is the same preparation used when the model was trained, so the live score matches the tested score.

### 10. Limits, said as business limits

- This book looks complete and tidy. A live book will have blanks and unusual cases. Recheck the score on our own recent loans before relying on it.
- We predict whether a default happens, not how much money is lost.
- Half of defaults are not in the review queue at today’s setting. If a missed default is much more expensive than an extra review, lower the bar and accept a bigger queue.
- Age and marital status need a policy decision before go-live.

### 11. Close

We can rank new applications so reviewers spend time where defaults are more common: younger borrowers, expensive rates, and loans that are large relative to pay. The tool is ready to demo. The decision stays with the officer.

### 12. Backup slide (only if asked)

Four methods compared. Winner: histogram gradient boosting, review if the probability is at least 18%.

| | Average precision on unseen loans | Share of defaults caught | Share of the queue that truly defaults |
| --- | ---: | ---: | ---: |
| Selected tool | 0.33 | 49% | 30% |
| Guess “no-one defaults” | 0.12 | 0% | — |

Training used 80% of the 255,347 applications. The 20% hold-out was scored only after the method was chosen.

## Likely questions and answers

| Question | Answer |
| --- | --- |
| Will this reject customers automatically? | No. High risk means “review this file.” |
| Why not just use credit score? | Credit score is a weak signal here compared with age, interest rate, and loan size versus income. |
| How many defaults will we still miss? | About half, at the current review bar. We can lower the bar if the committee wants a wider net. |
| Can we trust the percentage? | It is the model’s estimated chance. It is good for ordering files. We have not published a separate calibration chart. |
| Where did the data come from? | A public Kaggle loan-default table, originally from a Coursera challenge, licence CC0. It is not our bank’s own book. |
| What did you personally do? | *[Each speaker: one decision, one file, one number they can defend.]* |

## Slide style

- One claim per slide, then the number that supports it.
- Large type. Three charts at most in the whole deck.
- No code, no confusion matrices, no parameter lists on the main path.
- The backup slide can hold the comparison table for a technical question.
