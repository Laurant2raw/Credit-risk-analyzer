# Credit Risk Analyzer

This project started from a simple question: when a lender approves or declines an application, what's actually happening underneath? The answer turns out to be less about the model and more about where you set the threshold.

Predicting loan default on the UCI German Credit dataset, with attention to metric choice, decision thresholds, and regulatory constraints on feature use.

## Results

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Majority-class baseline | 0.700 | 0.500 |
| Logistic Regression | 0.700 | 0.747 |
| Gradient Boosting | 0.740 | **0.772** |

Gradient boosting outperformed logistic regression by 0.025 AUC — a real but
modest gain, discussed under *Model choice* below.

## Data

UCI German Credit (via OpenML `credit-g`): 1,000 labeled records, 20 features,
no missing values. Class distribution is 700 good / 300 bad.

That 70/30 split is the reason accuracy is not used as the headline metric. A
model that approves every applicant scores 70% accuracy and catches zero
defaults. Logistic regression here scores exactly 0.700 — identical to the
trivial baseline — while achieving 0.747 AUC. Its confusion matrix shows why:
it catches 30 of 60 defaulters (the baseline catches none) but rejects 30 good
customers, and the two errors cancel out in the accuracy figure. Accuracy
cannot distinguish a working model from a useless one on imbalanced data.
ROC-AUC can.

## Protected features

`personal_status` (sex and marital status) and `foreign_worker` (national
origin) were removed. Under the Equal Credit Opportunity Act these attributes
cannot be used in a US credit decision, so a model relying on them is not
deployable regardless of its performance.

The cost of that exclusion was measured directly by retraining on all 20
columns with an identical split and pipeline:

| Feature set | Columns | AUC |
|---|---|---|
| All features | 20 | 0.764 |
| Legally compliant | 18 | 0.772 |

The compliant model scored marginally higher. On a 200-row test set a 0.008
difference is within noise, so the honest conclusion is that exclusion carried
**no measurable cost** — not that removing protected attributes improves
performance.

## Model choice

Logistic regression is the baseline because lenders must issue adverse action
notices explaining why an applicant was declined. Coefficients map directly to
that explanation; a boosted ensemble does not.

Gradient boosting won by 0.025 AUC. For a regulated lender that gain is
unlikely to justify losing per-decision explainability, so the interpretable
model is the more defensible production choice here. The comparison is the
point — the tradeoff was measured rather than assumed.

## Threshold selection

The default 0.50 cutoff implicitly treats both errors as equally costly. They
are not: approving a defaulter forfeits principal, while declining a
creditworthy applicant forfeits only expected interest. Assuming $5,000 and
$500 respectively, every threshold from 0.05 to 0.95 was evaluated on total
expected cost.

| Threshold | Bad approved | Good rejected | Cost | Approval rate |
|---|---|---|---|---|
| 0.50 (default) | 29 | 23 | $156,500 | 74% |
| 0.94 (cost-optimal) | 5 | 83 | $66,500 | 31% |
| 0.54 (constrained) | 25 | 23 | $136,500 | 71% |

Unconstrained cost minimization selected 0.94, near the boundary of the search
range, and approved only 31% of applicants. That is a limitation of the cost
model rather than a usable policy — no lender survives rejecting 59% of
creditworthy applicants, and the absence of an interior optimum indicates the
objective is missing a term.

Adding a minimum 70% approval-rate constraint yields 0.54. Notably, staying
commercially viable roughly doubles expected credit losses ($136,500 vs
$66,500). That tension is real and is the substance of the exercise.

The $5,000 / $500 figures are assumptions, not measurements. A production
system would use realized loss data and a customer-lifetime-value estimate for
rejected good applicants.

## Feature importance

Logistic regression coefficients (positive = predicts repayment):

| Feature | Coefficient |
|---|---|
| purpose: used car | +1.35 |
| checking status: none | +0.94 |
| other parties: guarantor | +0.87 |
| credit history: critical/other existing | +0.78 |
| purpose: education | −0.76 |
| checking status: < 0 | −0.72 |
| purpose: new car | −0.69 |

Used-car loans are the strongest positive signal while new-car loans are among
the strongest negative ones. The likely driver is the borrower rather than the
collateral — purchasing used is a proxy for spending below means.

Two of the strongest signals are probably artifacts rather than causal
relationships. "No checking account" reads as safe most likely because the
overdrawn accounts visible to this lender are what the feature actually
captures; customers banking elsewhere are recorded identically to customers
with no banking relationship. Likewise, "critical credit history" predicting
repayment reflects a labeling quirk of this dataset — borrowers who have been
through difficulty and survived it — rather than a rule that would generalize.

## Limitations

- 1,000 records is small; a 200-row test set makes differences under roughly
  0.02 AUC unreliable.
- Cost parameters are assumed, not measured.
- Data is 1990s German consumer credit and does not reflect current US lending
  populations or products.
- No hyperparameter tuning; both models use library defaults.

## Reproducing

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/01_exploration.ipynb
```

Data downloads automatically via `fetch_openml` on first run.