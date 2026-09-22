# Stage 10 — Frontend and user experience

The screen is a Streamlit app in `frontend/app.py`. It is the credit-officer view from Stage 1: enter the application, check it, read whether the loan looks likely to default. It does not retrain a model and it does not reimplement the preprocessor. It sends JSON to `POST /predict`.

Start the API first, then:

```bash
streamlit run frontend/app.py
```

The screen opens at http://localhost:8501. The API address defaults to `http://127.0.0.1:8000` and can be overridden with the `API_URL` environment variable.

## Layout and instructions

The page title is “Loan application review.” The introduction says the check estimates whether the loan is likely to default before it is approved.

Fields are grouped the way an officer works:

- **Applicant:** age, annual income, credit score, months employed, education, employment type, marital status, credit lines.
- **Loan:** amount, interest rate, term, debt-to-income ratio, purpose, mortgage, dependents, co-signer.

Each numeric field has a help line with the same limits as the API (age 18–100, score 300–850, rate between 0 and 100, DTI 0–1.5, and so on). Categories are select boxes and yes/no radios, so the officer cannot invent a label that training never saw. Loan term is a select box of 12, 24, 36, 48, and 60 months. The form opens with a worked example (age 35, income 72,000, score 680, amount 15,000, rate 12.5%, term 36, DTI 0.32, bachelor’s degree, full-time, married, auto loan, no mortgage, dependents, no co-signer) so a demo can be submitted immediately and then edited.

The submit button is “Check default risk.”

## Validation

`validate()` in `frontend/app.py` repeats the API limits before the request is sent:

- Age 18–100, credit score 300–850, months employed ≥ 0, credit lines ≥ 1.
- Income and loan amount > 0.
- Interest rate in (0, 100).
- Term in the five allowed lengths.
- DTI in [0, 1.5].

If any rule fails, the screen lists the problems and does not call the API. If the API still returns 422 (for example a limit tightened on the server), those field messages are shown as well. If the service is not running, the screen says so and prints the uvicorn command, instead of a connection traceback.

Number inputs allow a slightly wider range than the business rule (age up to 120, DTI up to 2) so an out-of-policy value can be typed and then rejected with an explanation. The select boxes cannot submit an unknown category.

## How the result is shown

`show_result()` reads the API JSON.

- If `will_default` is true, the message is shown as an error and the heading is “Send this application for a closer review.”
- If it is false, the message is shown as a success and the heading is “Lower default risk.”
- A metric shows the probability of default as a percent, with a progress bar.
- A caption names the model and states that the result supports the approval decision and does not replace it.

The officer sees an action (review or lower risk), the probability, and the limit of the tool. They do not see coefficients, folds, or feature names.

## End-to-end path

1. The officer submits the form.
2. The screen validates locally.
3. `score_application()` POSTs the 16 raw fields to `{API_URL}/predict`.
4. The API validates again, runs `preprocessor.transform`, and scores with the selected booster at cutoff 0.18.
5. The screen renders the returned sentence, label, and probability.

Both sides use the training category lists and the training domain rules. The screen does not scale or one-hot anything itself, so it cannot drift from the training pipeline.

## How this meets the brief

| Requirement | What the screen does |
| --- | --- |
| Clear input fields and instructions | Two sections, help text on every numeric field, constrained choices for categories |
| Validate user inputs | Local checks plus display of API 422 errors |
| Present prediction results clearly | Review vs lower risk, probability, progress bar, plain sentence |
| Follow the scenario | Written for a credit officer before approval, not as a generic model playground |
| Simple experience | One form, one button, one result |
| Frontend and backend together | The screen only scores by calling `/predict` |

Browser check for a demo: with both processes running, submit the default example, then change age to 10 and income to 0 and confirm the screen blocks the request and names those fields.
