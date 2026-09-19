"""Generate realistic synthetic pharmacy data for testing the risk model.

Uses real drug names and realistic inventory/supplier parameters based on
ASHP drug shortage patterns. This is SYNTHETIC data - not real pharmacy data.
"""

import random
import uuid

import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

# Real drug names grouped by category, based on actual ASHP shortage patterns
DRUGS = {
    "antibiotic": [
        ("Amoxicillin 500mg", "SUP001", 0.85),
        ("Amoxicillin 875mg", "SUP001", 0.87),
        ("Azithromycin 250mg", "SUP002", 0.90),
        ("Ciprofloxacin 500mg", "SUP003", 0.82),
        ("Doxycycline 100mg", "SUP004", 0.88),
        ("Metronidazole 500mg", "SUP002", 0.84),
        ("Cephalexin 500mg", "SUP001", 0.86),
        ("Trimethoprim-Sulfamethoxazole", "SUP005", 0.79),
    ],
    "cardiovascular": [
        ("Lisinopril 10mg", "SUP006", 0.92),
        ("Lisinopril 20mg", "SUP006", 0.91),
        ("Metoprolol Tartrate 50mg", "SUP007", 0.88),
        ("Metoprolol Succinate 100mg", "SUP007", 0.86),
        ("Amlodipine 5mg", "SUP008", 0.93),
        ("Losartan 50mg", "SUP009", 0.84),
        ("Atorvastatin 20mg", "SUP010", 0.90),
        ("Warfarin 5mg", "SUP011", 0.78),
        ("Clopidogrel 75mg", "SUP008", 0.87),
        ("Furosemide 40mg", "SUP006", 0.82),
    ],
    "diabetes": [
        ("Metformin 500mg", "SUP012", 0.91),
        ("Metformin 1000mg", "SUP012", 0.89),
        ("Glipizide 5mg", "SUP013", 0.85),
        ("Insulin Lispro 100U/mL", "SUP014", 0.62),
        ("Insulin Glargine 100U/mL", "SUP014", 0.58),
        ("Empagliflozin 10mg", "SUP015", 0.88),
        ("Sitagliptin 100mg", "SUP015", 0.86),
    ],
    "analgesic": [
        ("Acetaminophen 500mg", "SUP016", 0.94),
        ("Ibuprofen 400mg", "SUP016", 0.95),
        ("Ibuprofen 600mg", "SUP016", 0.93),
        ("Naproxen 500mg", "SUP017", 0.91),
        ("Aspirin 81mg", "SUP016", 0.96),
        ("Tramadol 50mg", "SUP018", 0.74),
        ("Acetaminophen/Codeine #3", "SUP018", 0.68),
        ("Gabapentin 300mg", "SUP019", 0.83),
        ("Pregabalin 75mg", "SUP019", 0.80),
    ],
    "respiratory": [
        ("Albuterol Inhaler", "SUP020", 0.72),
        ("Fluticasone Nasal Spray", "SUP021", 0.88),
        ("Montelukast 10mg", "SUP022", 0.86),
        ("Prednisone 10mg", "SUP023", 0.90),
        ("Budesonide Inhaler", "SUP021", 0.84),
        ("Theophylline 200mg", "SUP020", 0.76),
    ],
    "gastrointestinal": [
        ("Omeprazole 20mg", "SUP024", 0.91),
        ("Pantoprazole 40mg", "SUP024", 0.89),
        ("Famotidine 20mg", "SUP025", 0.90),
        ("Ondansetron 4mg", "SUP026", 0.82),
        ("Ondansetron 8mg", "SUP026", 0.80),
        ("Docusate 100mg", "SUP025", 0.93),
        ("Polyethylene Glycol 17g", "SUP027", 0.91),
    ],
    "neurological": [
        ("Levetiracetam 500mg", "SUP028", 0.85),
        ("Lamotrigine 100mg", "SUP029", 0.82),
        ("Carbamazepine 200mg", "SUP028", 0.80),
        ("Topiramate 25mg", "SUP030", 0.84),
        ("Sumatriptan 50mg", "SUP031", 0.79),
        ("Baclofen 10mg", "SUP032", 0.83),
    ],
    "psychiatric": [
        ("Sertraline 50mg", "SUP033", 0.90),
        ("Sertraline 100mg", "SUP033", 0.88),
        ("Fluoxetine 20mg", "SUP034", 0.91),
        ("Escitalopram 10mg", "SUP035", 0.89),
        ("Venlafaxine 75mg", "SUP036", 0.85),
        ("Quetiapine 50mg", "SUP037", 0.81),
        ("Lithium 300mg", "SUP038", 0.77),
        ("Methylphenidate 20mg", "SUP039", 0.55),
        ("Amphetamine Salt Combo 20mg", "SUP039", 0.52),
    ],
    "oncology": [
        ("Methotrexate 2.5mg", "SUP040", 0.70),
        ("Cyclophosphamide 500mg", "SUP041", 0.65),
        ("Ifosfamide 1g", "SUP041", 0.48),
        ("Fluorouracil 500mg", "SUP042", 0.68),
        ("Doxorubicin 50mg", "SUP043", 0.63),
        ("Paclitaxel 100mg", "SUP044", 0.60),
        ("Carboplatin 150mg", "SUP042", 0.58),
        ("Cisplatin 50mg", "SUP044", 0.56),
    ],
    "immunological": [
        ("Adalimumab 40mg", "SUP045", 0.66),
        ("Methotrexate 15mg", "SUP040", 0.72),
        ("Prednisone 5mg", "SUP023", 0.91),
        ("Tacrolimus 1mg", "SUP046", 0.74),
        ("Mycophenolate 500mg", "SUP046", 0.78),
    ],
}

# Supplier characteristics: (reliability_base, lead_time_mean, lead_time_std)
SUPPLIER_PROFILES = {
    "SUP001": (0.85, 7, 2),
    "SUP002": (0.90, 5, 1),
    "SUP003": (0.82, 10, 3),
    "SUP004": (0.88, 6, 2),
    "SUP005": (0.79, 12, 4),
    "SUP006": (0.92, 4, 1),
    "SUP007": (0.87, 8, 2),
    "SUP008": (0.90, 5, 1),
    "SUP009": (0.84, 9, 3),
    "SUP010": (0.90, 6, 2),
    "SUP011": (0.78, 14, 5),
    "SUP012": (0.91, 5, 1),
    "SUP013": (0.85, 7, 2),
    "SUP014": (0.60, 18, 6),  # Insulin - problematic supply chain
    "SUP015": (0.87, 7, 2),
    "SUP016": (0.94, 3, 1),
    "SUP017": (0.91, 5, 1),
    "SUP018": (0.72, 15, 5),  # Controlled substances - harder to source
    "SUP019": (0.82, 9, 3),
    "SUP020": (0.74, 12, 4),
    "SUP021": (0.86, 7, 2),
    "SUP022": (0.86, 6, 2),
    "SUP023": (0.90, 5, 1),
    "SUP024": (0.90, 5, 1),
    "SUP025": (0.91, 4, 1),
    "SUP026": (0.81, 8, 3),
    "SUP027": (0.91, 5, 1),
    "SUP028": (0.84, 8, 2),
    "SUP029": (0.82, 9, 3),
    "SUP030": (0.84, 7, 2),
    "SUP031": (0.79, 10, 3),
    "SUP032": (0.83, 8, 2),
    "SUP033": (0.89, 6, 2),
    "SUP034": (0.91, 5, 1),
    "SUP035": (0.89, 6, 2),
    "SUP036": (0.85, 7, 2),
    "SUP037": (0.81, 9, 3),
    "SUP038": (0.77, 11, 4),
    "SUP039": (0.54, 20, 7),  # Controlled stimulants - chronic shortages
    "SUP040": (0.70, 14, 5),
    "SUP041": (0.65, 16, 5),
    "SUP042": (0.67, 15, 5),
    "SUP043": (0.63, 17, 6),
    "SUP044": (0.61, 18, 6),
    "SUP045": (0.66, 16, 5),
    "SUP046": (0.76, 11, 4),
}


def generate_dataset(n_rows: int = 500) -> pd.DataFrame:
    """Generate synthetic pharmacy risk dataset."""
    rows = []

    for _ in range(n_rows):
        # Pick a random category and drug
        category = random.choice(list(DRUGS.keys()))
        drug_name, supplier_id, supplier_base_reliability = random.choice(DRUGS[category])

        # Supplier characteristics with noise
        sup_profile = SUPPLIER_PROFILES[supplier_id]
        supplier_reliability = np.clip(
            sup_profile[0] + np.random.normal(0, 0.05), 0, 1
        )
        supplier_lead_time = max(
            1, int(sup_profile[1] + np.random.normal(0, sup_profile[2]))
        )

        # Inventory characteristics
        max_capacity = random.randint(100, 5000)
        avg_weekly_demand = random.randint(10, min(max_capacity // 2, 500))
        current_inventory = random.randint(0, max_capacity)
        reorder_point = random.randint(
            int(avg_weekly_demand * 1), int(avg_weekly_demand * 4)
        )

        # Demand patterns
        prescription_volume_30d = random.randint(
            int(avg_weekly_demand * 2), int(avg_weekly_demand * 6)
        )
        demand_trend = round(np.random.uniform(-0.3, 0.5), 2)

        # Historical shortage patterns - correlate with supplier reliability
        shortage_rate = (1 - supplier_reliability) * 0.8
        historical_shortage_count = int(
            np.random.poisson(shortage_rate * 8)
        )
        days_since_last_shortage = (
            random.randint(30, 365)
            if historical_shortage_count == 0
            else random.randint(1, 180)
        )
        avg_stockout_duration_days = (
            0.0
            if historical_shortage_count == 0
            else round(np.random.exponential(5), 1)
        )

        # Risk label: binary (1 = high risk of shortage)
        # Based on realistic factors: low inventory, unreliable supplier,
        # high shortage history, controlled substance, oncology drug
        risk_factors = 0
        if current_inventory < reorder_point:
            risk_factors += 2
        if supplier_reliability < 0.7:
            risk_factors += 2
        if historical_shortage_count > 3:
            risk_factors += 2
        if category in ("oncology",):
            risk_factors += 1
        if supplier_lead_time > 14:
            risk_factors += 1
        if current_inventory / max(max_capacity, 1) < 0.2:
            risk_factors += 1

        risk_label = 1 if risk_factors >= 4 else 0

        # Add some noise to labels (flip ~5% to make it realistic)
        if random.random() < 0.05:
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
    print("Generating synthetic pharmacy dataset...")
    df = generate_dataset(n_rows=500)

    out_path = "data/pharmacy_risk_data.csv"
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
