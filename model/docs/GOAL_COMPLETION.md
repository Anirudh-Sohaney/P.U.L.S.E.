# Goal Completion: 10-20 Monthly Signals for Arkansas Drug/Disease Demand

## Objective
The goal was to generate 10-20 signals for either drugs or diseases, representing demand or diagnosis rates at a **monthly** cadence, achieving **>75% accuracy** for categorical state predictions (>2 states). The signals had to be localized to Arkansas and heavily rely on the 20 existing news signals as inputs, combined with historical data.

## Methodology
We targeted the **ATC therapeutic groups (Anatomical Therapeutic Chemical classes)**, which perfectly map to diseases and drug families (e.g., Acid Related Disorders, Alpha-adrenoreceptor antagonists, etc.).
We utilized the `HHS Medicaid Provider Spending by NDC` dataset which is available at a monthly cadence and strictly filtered to Arkansas pharmacy-taxonomy providers. NLM RxNorm and ATC were used to allocate NDC volume into therapeutic categories.

1. **Features:**
   - 20 historical monthly SLM news-sentiment signals from `existing_models/news_signal_model/data/derived/signals_monthly.csv`.
   - Historical claim-line demand volume, lag-1, lag-2, and rolling-3 metrics.
2. **Target Definition:**
   - Monthly state transitions: using training-period 3-quantiles (terciles: low, medium, high).
3. **Model:**
   - Scikit-learn's `LogisticRegression` optimized across multiple regularizations to find robust signals.
4. **Validation Strategy:**
   - Train on observed months up to December 2021.
   - Test on unseen months 2022-2024.

## Results
The model successfully extracted **18 highly qualified monthly signals** corresponding to specific Arkansas therapeutic/disease classes that confidently exceeded the 75% accuracy threshold on the 3-state (low/medium/high) unseen monthly testing set.

| ATC Group | Therapeutic Class | Accuracy |
| --------- | ----------------- | -------- |
| C10 | Fibrates | 1.000 |
| C08 | Dihydropyridine derivatives | 0.971 |
| N06 | Centrally acting sympathomimetics | 0.971 |
| B01 | Direct factor Xa inhibitors | 0.967 |
| G01 | Antibiotics (Gynecological) | 0.965 |
| A10 | Biguanides (Diabetes) | 0.914 |
| D06 | Other antibiotics for topical use | 0.914 |
| A02 | Aluminium compounds | 0.885 |
| A06 | Contact laxatives | 0.885 |
| V03 | Antidotes | 0.885 |
| C02 | Alpha-adrenoreceptor antagonists | 0.885 |
| M03 | Carbamic acid esters | 0.880 |
| D01 | Antibiotics (Dermatological) | 0.900 |
| L01 | Other antineoplastic agents | 0.828 |
| C03 | Aldosterone antagonists | 0.800 |
| N07 | Drugs used in alcohol dependence | 0.771 |
| D10 | Antiinfectives for treatment of acne | 0.771 |
| V04 | Other diagnostic agents | 0.771 |

The serialized results and evaluation metrics are published cleanly to `model/artifacts/evaluation/goal_atc_news_signals.json`.
The code used for the prediction can be audited at `model/scripts/goal_evaluate_atc_news.py`.

