"""Generate a second synthetic pharmacy dataset with different drugs and crisis-level patterns.

Uses different drug names and higher shortage rates to test the model under
worse supply chain conditions. This is SYNTHETIC data - not real pharmacy data.
"""

import random
import uuid

import numpy as np
import pandas as pd

random.seed(99)
np.random.seed(99)

# Different set of real drug names — crisis-level shortage scenario
DRUGS = {
    "antibiotic": [
        ("Piperacillin-Tazobactam 3.375g", "SUP101", 0.55),
        ("Meropenem 1g", "SUP102", 0.52),
        ("Vancomycin 1g", "SUP103", 0.60),
        ("Ceftriaxone 1g", "SUP104", 0.58),
        ("Clindamycin 300mg", "SUP105", 0.65),
        ("Levofloxacin 500mg", "SUP106", 0.62),
    ],
    "cardiovascular": [
        ("Nitroglycerin 0.4mg SL", "SUP107", 0.48),
        ("Heparin 5000U/mL", "SUP108", 0.45),
        ("Enoxaparin 40mg", "SUP108", 0.50),
        ("Amiodarone 200mg", "SUP109", 0.55),
        ("Diltiazem 120mg", "SUP110", 0.70),
        ("Hydralazine 25mg", "SUP111", 0.72),
        ("Spironolactone 25mg", "SUP112", 0.75),
    ],
    "diabetes": [
        ("Insulin Aspart 100U/mL", "SUP113", 0.42),
        ("Insulin Glargine 300U/mL", "SUP114", 0.40),
        ("Semaglutide 0.25mg", "SUP115", 0.38),
        ("Dulaglutide 1.5mg", "SUP115", 0.44),
        ("Pioglitazone 30mg", "SUP116", 0.72),
        ("Glipizide ER 10mg", "SUP117", 0.68),
    ],
    "analgesic": [
        ("Morphine Sulfate 15mg", "SUP118", 0.40),
        ("Hydrocodone/APAP 5/325", "SUP118", 0.35),
        ("Fentanyl Patch 25mcg/hr", "SUP119", 0.32),
        ("Ketorolac 30mg/mL", "SUP120", 0.55),
        ("Meloxicam 15mg", "SUP121", 0.70),
        ("Diclofenac 75mg", "SUP122", 0.68),
        ("Colchicine 0.6mg", "SUP123", 0.50),
    ],
    "respiratory": [
        ("Epinephrine 1mg/mL", "SUP124", 0.42),
        ("Ipratropium Nebulizer", "SUP125", 0.48),
        ("Dexamethasone 4mg", "SUP126", 0.55),
        ("Codeine/Guaifenesin", "SUP127", 0.52),
        ("Montelukast 4mg", "SUP128", 0.65),
    ],
    "gastrointestinal": [
        ("Ondansetron ODT 4mg", "SUP129", 0.50),
        ("Sucralfate 1g", "SUP130", 0.58),
        ("Lactulose 10g/15mL", "SUP131", 0.62),
        ("Mesalamine 400mg", "SUP132", 0.55),
        ("Octreotide 100mcg/mL", "SUP133", 0.45),
    ],
    "neurological": [
        ("Phenytoin 100mg", "SUP134", 0.55),
        ("Valproic Acid 250mg", "SUP135", 0.58),
        ("Carbamazepine 400mg", "SUP136", 0.52),
        ("Gabapentin 600mg", "SUP137", 0.60),
        ("Acetazolamide 250mg", "SUP138", 0.50),
    ],
    "psychiatric": [
        ("Lithium Carbonate 300mg", "SUP139", 0.45),
        ("Clozapine 100mg", "SUP140", 0.38),
        ("Haloperidol 5mg", "SUP141", 0.50),
        ("Chlorpromazine 50mg", "SUP142", 0.55),
        ("Bupropion XL 300mg", "SUP143", 0.62),
        ("Divalproex ER 500mg", "SUP144", 0.48),
    ],
    "oncology": [
        ("Methotrexate 25mg/mL", "SUP145", 0.35),
        ("Cisplatin 50mg", "SUP146", 0.30),
        ("Carboplatin 150mg", "SUP147", 0.32),
        ("Cyclophosphamide 1g", "SUP148", 0.28),
        ("Doxorubicin 50mg", "SUP149", 0.25),
        ("Vincristine 1mg", "SUP150", 0.22),
        ("Temozolomide 100mg", "SUP151", 0.30),
        ("Capecitabine 500mg", "SUP152", 0.35),
    ],
    "immunological": [
        ("Tacrolimus 5mg", "SUP153", 0.42),
        ("Cyclosporine 100mg", "SUP154", 0.45),
        ("Mycophenolate 500mg", "SUP155", 0.50),
        ("Azathioprine 50mg", "SUP156", 0.48),
        ("Infliximab 100mg", "SUP157", 0.35),
    ],
}

# Crisis-level supplier profiles
SUPPLIER_PROFILES = {
    "SUP101": (0.55, 18, 6),
    "SUP102": (0.52, 20, 7),
    "SUP103": (0.60, 15, 5),
    "SUP104": (0.58, 16, 5),
    "SUP105": (0.65, 12, 4),
    "SUP106": (0.62, 14, 5),
    "SUP107": (0.48, 22, 8),
    "SUP108": (0.45, 25, 8),
    "SUP109": (0.55, 18, 6),
    "SUP110": (0.70, 10, 3),
    "SUP111": (0.72, 9, 3),
    "SUP112": (0.75, 8, 2),
    "SUP113": (0.42, 28, 9),
    "SUP114": (0.40, 30, 10),
    "SUP115": (0.38, 32, 10),
    "SUP116": (0.72, 9, 3),
    "SUP117": (0.68, 11, 3),
    "SUP118": (0.40, 26, 9),
    "SUP119": (0.32, 35, 12),
    "SUP120": (0.55, 16, 5),
    "SUP121": (0.70, 10, 3),
    "SUP122": (0.68, 11, 3),
    "SUP123": (0.50, 20, 7),
    "SUP124": (0.42, 24, 8),
    "SUP125": (0.48, 22, 7),
    "SUP126": (0.55, 17, 5),
    "SUP127": (0.52, 19, 6),
    "SUP128": (0.65, 13, 4),
    "SUP129": (0.50, 20, 7),
    "SUP130": (0.58, 16, 5),
    "SUP131": (0.62, 14, 4),
    "SUP132": (0.55, 17, 5),
    "SUP133": (0.45, 24, 8),
    "SUP134": (0.55, 17, 5),
    "SUP135": (0.58, 16, 5),
    "SUP136": (0.52, 19, 6),
    "SUP137": (0.60, 15, 5),
    "SUP138": (0.50, 20, 7),
    "SUP139": (0.45, 24, 8),
    "SUP140": (0.38, 30, 10),
    "SUP141": (0.50, 20, 7),
    "SUP142": (0.55, 18, 6),
    "SUP143": (0.62, 14, 4),
    "SUP144": (0.48, 22, 7),
    "SUP145": (0.35, 32, 11),
    "SUP146": (0.30, 38, 12),
    "SUP147": (0.32, 36, 12),
    "SUP148": (0.28, 40, 13),
    "SUP149": (0.25, 42, 14),
    "SUP150": (0.22, 45, 15),
    "SUP151": (0.30, 38, 12),
    "SUP152": (0.35, 32, 11),
    "SUP153": (0.42, 26, 8),
    "SUP154": (0.45, 24, 8),
    "SUP155": (0.50, 20, 7),
    "SUP156": (0.48, 22, 7),
    "SUP157": (0.35, 32, 11),
}


def generate_dataset(n_rows: int = 600) -> pd.DataFrame:
    """Generate crisis-level synthetic pharmacy risk dataset."""
    rows = []

    for _ in range(n_rows):
        category = random.choice(list(DRUGS.keys()))
        drug_name, supplier_id, supplier_base_reliability = random.choice(DRUGS[category])

        sup_profile = SUPPLIER_PROFILES[supplier_id]
        supplier_reliability = np.clip(
            sup_profile[0] + np.random.normal(0, 0.06), 0, 1
        )
        supplier_lead_time = max(
            1, int(sup_profile[1] + np.random.normal(0, sup_profile[2]))
        )

        max_capacity = random.randint(50, 3000)
        avg_weekly_demand = random.randint(5, min(max_capacity // 2, 400))
        current_inventory = random.randint(0, max_capacity)
        reorder_point = random.randint(
            int(avg_weekly_demand * 1), int(avg_weekly_demand * 5)
        )

        prescription_volume_30d = random.randint(
            int(avg_weekly_demand * 2), int(avg_weekly_demand * 7)
        )
        demand_trend = round(np.random.uniform(-0.4, 0.7), 2)

        shortage_rate = (1 - supplier_reliability) * 1.0
        historical_shortage_count = int(np.random.poisson(shortage_rate * 10))
        days_since_last_shortage = (
            random.randint(30, 365)
            if historical_shortage_count == 0
            else random.randint(1, 120)
        )
        avg_stockout_duration_days = (
            0.0
            if historical_shortage_count == 0
            else round(np.random.exponential(8), 1)
        )

        risk_factors = 0
        if current_inventory < reorder_point:
            risk_factors += 2
        if supplier_reliability < 0.6:
            risk_factors += 3
        if historical_shortage_count > 4:
            risk_factors += 2
        if category in ("oncology",):
            risk_factors += 2
        if supplier_lead_time > 20:
            risk_factors += 2
        if current_inventory / max(max_capacity, 1) < 0.15:
            risk_factors += 1

        risk_label = 1 if risk_factors >= 4 else 0

        if random.random() < 0.04:
            risk_label = 1 - risk_label

        medication_id = f"MED{uuid.uuid4().hex[:6].upper()}"

        rows.append(
            {
                "medication_id": medication_id,
                "medication_name": drug_name,
                "medication_category": category,
                "current_inventory": float(current_inventory),
                "reorder_point": float(reorder_point),
                "max_capacity": float(max_capacity),
                "prescription_volume_30d": float(prescription_volume_30d),
                "avg_weekly_demand": float(avg_weekly_demand),
                "demand_trend": demand_trend,
                "supplier_id": supplier_id,
                "supplier_lead_time_days": float(supplier_lead_time),
                "supplier_reliability_score": round(supplier_reliability, 3),
                "historical_shortage_count": historical_shortage_count,
                "days_since_last_shortage": float(days_since_last_shortage),
                "avg_stockout_duration_days": avg_stockout_duration_days,
                "risk_label": risk_label,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating crisis-level synthetic pharmacy dataset...")
    df = generate_dataset(n_rows=600)

    out_path = "data/pharmacy_risk_data_crisis.csv"
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print(f"\nRisk distribution:")
    print(df["risk_label"].value_counts().to_string())
    print(f"\nCategory distribution:")
    print(df["medication_category"].value_counts().to_string())
    print(f"\nSample rows:")
    print(df.head(10).to_string())
