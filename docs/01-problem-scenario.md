# Stage 1 — Understand the problem scenario

This stage fixes the business problem before any dataset is treated as “the project.” The scenario was chosen first. The dataset in Stage 2 was then checked against that scenario.

## Problem

A lender loses money when an approved applicant stops repaying. Reviewing every application by hand is slow, and a simple rule such as “reject low credit scores” misses applicants who default for other reasons (younger age, short employment, a large loan relative to income, a high interest rate) and also rejects people who would have repaid.

The problem to solve is: **before funds are released, estimate whether this applicant will default**, so a credit officer can approve the loan, send it for a closer review, or price the risk.

The system supports that decision. It does not replace the officer. The screen states this explicitly: the result supports the approval decision.

## Prediction objective and target

| Item | Decision |
| --- | --- |
| Objective | Estimate the chance that a new loan application will default |
| Target variable | `Default` |
| Meaning of 1 | The applicant defaulted |
| Meaning of 0 | The applicant did not default |
| When the prediction is used | Before approval, using only information known on the application |

`LoanID` is an application key. It is not the target and it is not a predictor. A new applicant does not have a historical loan identifier that the model could use.

## Users, stakeholders, and decisions

| Who | What they need from the system |
| --- | --- |
| Credit officer (primary user) | A clear default / no-default result and a probability, while the application is still open |
| Lending operations | Fewer defaults that were approved without review, and a consistent way to flag higher-risk files |
| Risk / pricing staff | A ranked risk score so higher-risk loans can be reviewed or priced differently |
| Applicant | Indirect stakeholder. A wrong “default” flag can delay a fair approval, so the officer still makes the decision |

Decisions the score can support:

- Approve when predicted risk is low.
- Send the file for manual review when predicted risk is high.
- Use the probability as one input when setting the interest rate or asking for a co-signer.

## User requirements, inputs, and outputs

**Requirements**

- The officer enters the same facts that exist on a loan application. No field that would only be known after default is requested.
- Invalid or missing values are rejected with a plain message, not a stack trace.
- The result is a label the officer can act on, plus the probability of default.
- The same preparation used in training is applied to the live application, so the score is comparable to the evaluation.

**Inputs (one application)**

Age, annual income, loan amount, credit score, months employed, number of credit lines, interest rate, loan term, debt-to-income ratio, education, employment type, marital status, mortgage flag, dependents flag, loan purpose, and co-signer flag.

**Outputs**

- `prediction`: `Default` or `No default`
- `will_default`: true or false, using the selected decision cutoff (0.18)
- `default_probability`: model probability of default
- A short sentence naming the model and the outcome, for example that the applicant will default or will repay

## Task type

This is **binary classification**.

The target has two outcomes. The organisation needs a yes/no review flag and a probability, which classification provides. Regression would be appropriate if the target were a continuous amount, such as loss given default or months until delinquency. Those quantities are not in the problem or the data. Clustering or association mining would describe groups of applicants, but they would not answer “will this application default?”

## Why the dataset has to follow this scenario

A dataset is suitable only if each row is one loan decision, the columns are knowable at application time, and a default label exists. A generic credit-card or marketing table would not match a credit officer reviewing a new loan. Stage 2 records how the chosen file meets that test.
