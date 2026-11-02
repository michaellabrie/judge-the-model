# The data

Two public health datasets. One row is one patient or adult. Every file has the same layout:

- `row_id`: a row number.
- `split`: `train` (80%: the models learn from these rows) or `test` (20%: used only to evaluate the models). Never reshuffle it.
- the predictors, then `target` (1 = the positive class).

Blank cells are missing values. **When is it known?** says when each value exists, relative to the prediction point. A column known only at or after the outcome would leak the answer, so those columns are left out (listed under **Excluded**).

`manifest.json` records sources, licences, versions, row counts and SHA-256 fingerprints. `datasets.json` holds this dictionary as data.

## `readmission.csv`: 30-day readmission (20,000 discharges, 1,796 positive, 9.0%)

- **Cohort:** UCI Diabetes 130-US Hospitals 1999–2008: inpatient stays of patients with diabetes. First discharge per patient, excluding discharge to hospice or death, then a stratified random sample of 20,000 (random_state=0).
- **Job:** Predict readmission within 30 days, at discharge.
- **Prediction point:** At discharge
- **Positive class (`target` = 1):** Readmitted within 30 days
- **Framing:** Results apply to this historical inpatient cohort involving diabetes, not to all discharges.
- **Split:** 16,000 training rows (1,437 positive); 4,000 test rows (359 positive).
- **Source:** UCI Machine Learning Repository: Diabetes 130-US Hospitals for Years 1999–2008 (CC BY 4.0). Strack B, DeShazo JP, Gennings C, et al. Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records. BioMed Research International 2014;2014:781670.

| Column | Meaning | Units | Categories / missing | When is it known? |
|---|---|---|---|---|
| `row_id` | Row number in this file |  |  | Assigned by the course |
| `split` | `train` (the models learn from these rows) or `test` (used only to evaluate them) |  |  | Assigned by the course |
| `age` | Age band midpoint (the file gives 10-year bands, e.g. [70-80) → 75) | years |  | At admission |
| `female` | 1 = female, 0 = male | 0/1 |  | At admission |
| `time_in_hospital` | Length of stay | days |  | At discharge |
| `num_lab_procedures` | Lab tests during the stay | count |  | At discharge (during the stay) |
| `num_procedures` | Procedures other than lab tests during the stay | count |  | At discharge (during the stay) |
| `num_medications` | Distinct medications given during the stay | count |  | At discharge (during the stay) |
| `number_outpatient` | Outpatient visits in the year before this stay | count |  | At admission (past year) |
| `number_emergency` | Emergency visits in the year before this stay | count |  | At admission (past year) |
| `number_inpatient` | Inpatient admissions in the year before this stay | count |  | At admission (past year) |
| `number_diagnoses` | Diagnoses recorded for this stay | count |  | At discharge |
| `race` | Race as recorded by the hospital |  | `African American`; `Asian`; `Hispanic`; `Other`; `Unknown`; `White` | At admission |
| `admission_type` | Admission type |  | `Elective`; `Emergency`; `Newborn`; `Not recorded`; `Trauma Center`; `Urgent` | At admission |
| `admission_source` | Where the patient was admitted from |  | `Clinic Referral`; `Court/Law Enforcement`; `Emergency Room`; `Extramural Birth`; `HMO Referral`; `Not recorded`; `Physician Referral`; `Transfer from a Skilled Nursing Facility (SNF)`; `Transfer from a hospital`; `Transfer from another health care facility`; `Transfer from critial access hospital`; `Transfer from hospital inpt/same fac reslt in a sep claim` | At admission |
| `discharge_disposition` | Where the patient was discharged to (hospice and death excluded) |  | `Admitted as an inpatient to this hospital`; `Discharged to home`; `Discharged/transferred to ICF`; `Discharged/transferred to SNF`; `Discharged/transferred to a federal health care facility.`; `Discharged/transferred to a long term care hospital.`; `Discharged/transferred to a nursing facility certified under Medicaid but not certified under Medicare.`; `Discharged/transferred to another rehab fac including rehab units of a hospital .`; `Discharged/transferred to another short term hospital`; `Discharged/transferred to another type of inpatient care institution`; `Discharged/transferred to home under care of Home IV provider`; `Discharged/transferred to home with home health service`; `Discharged/transferred within this institution to Medicare approved swing bed`; `Discharged/transferred/referred another institution for outpatient services`; `Discharged/transferred/referred to a psychiatric hospital of psychiatric distinct part unit of a hospital`; `Discharged/transferred/referred to this institution for outpatient services`; `Left AMA`; `Neonate discharged to another hospital for neonatal aftercare`; `Not recorded` | At discharge |
| `a1c_result` | HbA1c test result during the stay (`None` = not measured) |  | `>7`; `>8`; `None`; `Normal` | At discharge (during the stay) |
| `max_glu_serum` | Highest serum glucose result during the stay (`None` = not measured) | mg/dL band | `>200`; `>300`; `None`; `Normal` | At discharge (during the stay) |
| `insulin` | Insulin during the stay: not prescribed, or dose steady, up or down |  | `Down`; `Not prescribed`; `Steady`; `Up` | At discharge (during the stay) |
| `med_change` | 1 = any diabetes medication changed during the stay | 0/1 |  | At discharge |
| `diabetes_med` | 1 = any diabetes medication prescribed | 0/1 |  | At discharge |
| `diag1_group` | Primary diagnosis, in Strack et al.'s nine groups of ICD-9 codes |  | `Circulatory`; `Diabetes`; `Digestive`; `Genitourinary`; `Injury`; `Musculoskeletal`; `Neoplasms`; `Other`; `Respiratory` | At discharge |
| `target` | **Target:** 1 = readmitted within 30 days of discharge, 0 = not (readmitted after 30 days, or not at all) | 0/1 |  | 30 days after discharge (the outcome) |

One row per person, so each person's records are already kept together.

## `diabetes.csv`: diabetes screening (5,257 adults, 997 positive, 19.0%)

- **Cohort:** NHANES 2017–2018: adults (18+) with an HbA1c measurement who answered “Has a doctor ever told you that you have diabetes?” with yes, no or borderline.
- **Job:** Flag adults for follow-up testing; finding positives matters most.
- **Prediction point:** At the survey examination, before the blood test results
- **Positive class (`target` = 1):** Diabetes (reported doctor diagnosis or HbA1c ≥ 6.5%)
- **Framing:** The label is “reported doctor diagnosis or HbA1c ≥ 6.5%”. It includes adults who were already diagnosed. It identifies the label at one time point and does not predict future diabetes. Results describe adults in this sample.
- **Split:** 4,205 training rows (797 positive); 1,052 test rows (200 positive).
- **Source:** CDC/NCHS National Health and Nutrition Examination Survey (NHANES) 2017–2018, public-use files (Public domain). Centers for Disease Control and Prevention, National Center for Health Statistics. National Health and Nutrition Examination Survey 2017–2018.

| Column | Meaning | Units | Categories / missing | When is it known? |
|---|---|---|---|---|
| `row_id` | Row number in this file |  |  | Assigned by the course |
| `split` | `train` (the models learn from these rows) or `test` (used only to evaluate them) |  |  | Assigned by the course |
| `age` | Age (80 means 80 or older) | years |  | At the interview |
| `female` | 1 = female, 0 = male | 0/1 |  | At the interview |
| `bmi` | Body mass index | kg/m² | blank = not recorded (86 rows) | At the examination |
| `waist` | Waist circumference | cm | blank = not recorded (280 rows) | At the examination |
| `sbp` | Systolic blood pressure, mean of up to 3 readings | mmHg | blank = not recorded (254 rows) | At the examination |
| `income_ratio` | Family income ÷ the poverty line (5 means 5 or more) | ratio | blank = not recorded (683 rows) | At the interview |
| `race` | Race/ethnicity (self-reported) |  | `Mexican American`; `Non-Hispanic Asian`; `Non-Hispanic Black`; `Non-Hispanic White`; `Other / multiracial`; `Other Hispanic` | At the interview |
| `hdl` | HDL cholesterol | mg/dL | blank = not recorded (89 rows) | From the blood sample taken at the examination |
| `total_chol` | Total cholesterol | mg/dL | blank = not recorded (89 rows) | From the blood sample taken at the examination |
| `target` | **Target:** 1 = told by a doctor they have diabetes, or HbA1c ≥ 6.5%; 0 = neither | 0/1 |  | From the interview and the HbA1c result (the outcome) |

**Excluded (leaked) columns, not in the file:**

| Column | Meaning | When is it known? |
|---|---|---|
| `hba1c` | HbA1c (%) | From the same blood test that defines the label (HbA1c ≥ 6.5% is positive) |
| `told_diabetes` | “Has a doctor ever told you that you have diabetes?” (DIQ010) | At the interview; a yes defines the label |
| `insulin` | Now taking insulin (DIQ050) | At the interview; only people already treated for diabetes answer yes |
| `pills` | Now taking diabetes pills (DIQ070) | At the interview; only people already treated for diabetes answer yes |

One row per person, so each person's records are already kept together.
