# Decisions

- readmission: The positive class is patients readmitted within 30 days of discharge. I would prioritize reducing false negatives because missing a high-risk patient could prevent them from receiving additional follow-up care or discharge support. False positives also matter because unnecessary interventions consume hospital resources, but I consider missed readmissions the more concerning error.

- diabetes: The positive class is adults with a doctor's diagnosis of diabetes or an HbA1c of at least 6.5%. I would prioritize minimizing false negatives because missing a patient with diabetes could delay additional testing and treatment. False positives may lead to unnecessary testing, but follow-up testing can help clarify the result.

## Before results
For each dataset: positive class; which error matters more for the job (missed positive or false alarm) and why.
- readmission:
- diabetes:

## Recommendations
- readmission: <model and threshold, or "none is adequate for this job"> because ...
- diabetes: ...

## Design choice and AI use
A few sentences each. The design choice must be your own idea, not an AI tool's.
- design choice: <one thing you chose to show, or leave out, for your clinical lead> because ...
- AI suggestion I turned down: <what an AI tool suggested, and why you didn't use it, or "no AI tools used">
