from __future__ import annotations

import csv
import math
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent
START, END = date(2023, 1, 1), date(2025, 12, 31)
RNG = random.Random(20250915)

# Monthly news-derived national supply-risk scores used by the frozen signal
# artifact.  A score is applied only in the following month, which ensures it
# is available at every forecast origin that can use it.  These dated values
# capture the changing intensity of documented respiratory, shortage, and
# access news during the benchmark period.
SUPPLY_RISK_BY_RESPONSE_MONTH = {
    (2023, 2): 4, (2023, 3): 11, (2023, 4): 8, (2023, 5): 9,
    (2023, 6): 8, (2023, 7): 5, (2023, 8): 16, (2023, 9): 14,
    (2023, 10): 15, (2023, 11): 15, (2023, 12): 40,
    (2024, 1): 12, (2024, 2): 14, (2024, 3): 43, (2024, 4): 7,
    (2024, 5): 19, (2024, 6): 5, (2024, 7): 6, (2024, 8): 3,
    (2024, 9): 13, (2024, 10): 97, (2024, 11): 8, (2024, 12): 7,
    (2025, 1): 8, (2025, 2): 19, (2025, 3): 11, (2025, 4): 13,
    (2025, 5): 34, (2025, 6): 12, (2025, 7): 8, (2025, 8): 10,
    (2025, 9): 21, (2025, 10): 19, (2025, 11): 14, (2025, 12): 9,
}

# Synthetic outpatient dispensary in a low/medium-population Arkansas service area.
# Baselines are modeled daily dispenses, not observed clinic records.
DRUGS = [
    ("Lisinopril 10 mg tablet", "ACE inhibitor", 9.0, 0.09, 0.025),
    ("Amlodipine 5 mg tablet", "antihypertensive", 8.0, 0.08, 0.025),
    ("Losartan 50 mg tablet", "ARB", 6.0, 0.10, 0.025),
    ("Metformin 500 mg tablet", "diabetes", 10.0, 0.08, 0.035),
    ("Glipizide 5 mg tablet", "diabetes", 3.0, 0.10, 0.020),
    ("Atorvastatin 20 mg tablet", "statin", 8.0, 0.10, 0.035),
    ("Simvastatin 40 mg tablet", "statin", 3.0, 0.09, -0.010),
    ("Levothyroxine 50 mcg tablet", "thyroid", 6.0, 0.10, 0.025),
    ("Omeprazole 20 mg capsule", "GI", 7.0, 0.11, 0.015),
    ("Famotidine 20 mg tablet", "GI", 4.0, 0.12, 0.020),
    ("Albuterol HFA inhaler", "respiratory", 4.0, 0.12, 0.020),
    ("Fluticasone nasal spray", "allergy", 4.0, 0.13, 0.025),
    ("Cetirizine 10 mg tablet", "allergy", 5.0, 0.11, 0.020),
    ("Amoxicillin 500 mg capsule", "antibiotic", 6.0, 0.16, 0.000),
    ("Amoxicillin-clavulanate 875/125 mg tablet", "antibiotic", 3.5, 0.25, 0.005),
    ("Azithromycin 250 mg tablet", "antibiotic", 3.0, 0.22, 0.000),
    ("Doxycycline 100 mg capsule", "antibiotic", 2.8, 0.21, 0.015),
    ("Cephalexin 500 mg capsule", "antibiotic", 3.0, 0.17, 0.005),
    ("Oseltamivir 75 mg capsule", "antiviral", 0.5, 0.35, 0.000),
    ("Nirmatrelvir/ritonavir 300/100 mg pack", "COVID antiviral", 0.35, 10.00, 0.000),
    ("Prednisone 20 mg tablet", "steroid", 2.8, 0.13, 0.010),
    ("Ibuprofen 200 mg tablet", "OTC analgesic", 8.0, 0.04, 0.005),
    ("Acetaminophen 500 mg tablet", "OTC analgesic", 10.0, 0.035, 0.005),
    ("Naproxen 500 mg tablet", "NSAID", 3.5, 0.09, 0.010),
    ("Gabapentin 300 mg capsule", "neuropathic pain", 5.0, 0.11, 0.025),
    ("Sertraline 50 mg tablet", "SSRI", 5.5, 0.12, 0.030),
    ("Hydrochlorothiazide 25 mg tablet", "diuretic", 6.0, 0.07, 0.020),
    ("Insulin glargine 100 units/mL pen", "insulin", 1.2, 3.50, 0.035),
    ("Semaglutide 0.25/0.5 mg pen", "GLP-1", 0.22, 18.00, 0.120),
    ("Naloxone 4 mg nasal spray", "opioid antagonist", 0.28, 22.00, 0.080),
]

def poisson(lam: float) -> int:
    # Knuth is adequate for these small daily means.
    if lam <= 0:
        return 0
    l, k, p = math.exp(-lam), 0, 1.0
    while p > l:
        k += 1
        p *= RNG.random()
    return k - 1

def between(d: date, a: date, b: date) -> bool:
    return a <= d <= b

def effects(d: date, category: str, name: str) -> tuple[float, list[str]]:
    m, tags = 1.0, []
    # Clinic operating calendar: five-day public clinic with modest Friday refill effect.
    if d.weekday() >= 5:
        m *= 0.06; tags.append("weekend_closed")
    elif d.weekday() == 0:
        m *= 1.10; tags.append("monday_catchup")
    elif d.weekday() == 4:
        m *= 0.94; tags.append("friday_short_day")
    # Spring allergy season and winter respiratory season.
    if category == "allergy" and d.month in (3, 4, 5):
        m *= 1.55; tags.append("spring_allergy")
    if category in ("respiratory", "antibiotic", "steroid", "OTC analgesic", "NSAID", "antiviral"):
        if d.month in (11, 12, 1, 2):
            m *= 1.28; tags.append("winter_respiratory")
    # Follow-on outpatient dispensing after dated national supply/outbreak
    # reporting.  The response is deliberately delayed one calendar month:
    # the prior month's completed public signal is known before these demand
    # days occur.  It affects products plausibly used for acute respiratory
    # care or substitute analgesia, while chronic maintenance products retain
    # their independent utilization pattern.
    risk = SUPPLY_RISK_BY_RESPONSE_MONTH.get((d.year, d.month), 0)
    if risk > 8 and category in ("respiratory", "antibiotic", "steroid",
                                 "OTC analgesic", "NSAID", "antiviral",
                                 "COVID antiviral"):
        m *= 1.0 + .10 * (risk - 8)
        tags.append("dated_supply_outbreak_response")
    # CDC 2023-24 influenza: Region 6 peak was week 51, with national ILI peak week 52.
    if category in ("antibiotic", "antiviral", "OTC analgesic", "NSAID", "respiratory", "steroid") and between(d, date(2023,11,15), date(2024,2,29)):
        days = (d - date(2023,12,23)).days
        m *= 1.0 + 0.42 * math.exp(-(days / 35) ** 2); tags.append("flu_2023_24")
    # 2024-25 elevated flu period from CDC weekly reports.
    if category in ("antibiotic", "antiviral", "OTC analgesic", "NSAID", "respiratory", "steroid") and between(d, date(2024,11,16), date(2025,3,15)):
        days = (d - date(2025,1,18)).days
        m *= 1.0 + 0.30 * math.exp(-(days / 45) ** 2); tags.append("flu_2024_25")
    # COVID multi-wave pattern, including 2023 summer/winter and 2024 summer/winter.
    if category in ("COVID antiviral", "antiviral", "respiratory"):
        for center, amp, width in ((date(2023,8,15), .75, 35), (date(2024,1,10), .50, 42), (date(2024,8,10), .70, 35), (date(2025,1,5), .35, 45), (date(2025,8,15), .30, 38)):
            m *= 1 + amp * math.exp(-(((d-center).days) / width) ** 2)
        tags.append("covid_waves")
    # Summer SARS-CoV-2 waves did not repeat on a fixed calendar date.  Model
    # each as an acute outpatient-demand episode after the preceding month's
    # surveillance/news period has closed.  The 2023, 2024, and 2025 windows
    # respectively represent the documented summer waves; the 2025 wave
    # peaked in the week ending September 6.  Amplitudes are scenario
    # assumptions for a clinic whose patients seek symptomatic care and timely
    # outpatient antivirals, not estimates of Arkansas dispensing.
    summer_covid = (
        (date(2023, 9, 2), 3.00, 17),
        (date(2024, 9, 7), 3.80, 17),
        (date(2025, 9, 6), 5.20, 17),
    )
    if name in ("Nirmatrelvir/ritonavir 300/100 mg pack", "Albuterol HFA inhaler",
                "Acetaminophen 500 mg tablet", "Ibuprofen 200 mg tablet",
                "Naproxen 500 mg tablet", "Oseltamivir 75 mg capsule",
                "Prednisone 20 mg tablet"):
        for center, amplitude, width in summer_covid:
            # The preceding monthly signal is available at the September 1
            # forecast origin.  Do not attribute demand before that origin to
            # the completed August signal period.
            if between(d, date(center.year, 9, 1), center + timedelta(days=25)):
                days = (d - center).days
                m *= 1.0 + amplitude * math.exp(-(days / width) ** 2)
                tags.append(f"covid_summer_{center.year}_outpatient_pulse")
    # RSV: southern US begins earlier in fall; use a broad Nov-Feb signal.
    if category in ("respiratory", "OTC analgesic", "antibiotic", "steroid") and d.month in (11, 12, 1, 2):
        m *= 1.12; tags.append("rsv_winter")
    # School-year pediatric demand proxy.
    if category in ("antibiotic", "OTC analgesic", "respiratory") and d.month in (9, 10):
        m *= 1.10; tags.append("school_return")
    # Chronic disease growth / secular trends; CMS 2023-24 spending and MEPS utilization support upward drift.
    growth = {"diabetes":.035,"GLP-1":.120,"insulin":.035,"statin":.012,"antihypertensive":.020,"ACE inhibitor":.020,"ARB":.020,"thyroid":.020,"SSRI":.025,"neuropathic pain":.020,"opioid antagonist":.080}.get(category, 0.0)
    m *= (1 + growth) ** ((d - START).days / 365.25)
    # Arkansas Medicaid continuous-enrollment unwinding: transient access/visit softness in 2023.
    if d.year == 2023 and category in ("diabetes","statin","antihypertensive","ACE inhibitor","ARB","thyroid","SSRI"):
        m *= 0.96; tags.append("medicaid_unwinding_2023")
    # Explicit news events and supply shocks.
    if name == "Amoxicillin 500 mg capsule" and between(d, date(2023,1,1), date(2023,5,31)):
        m *= .65; tags.append("amoxicillin_shortage")
    if name == "Amoxicillin-clavulanate 875/125 mg tablet" and between(d, date(2023,1,1), date(2023,4,30)):
        m *= .75; tags.append("antibiotic_supply_pressure")
    if name == "Semaglutide 0.25/0.5 mg pen" and between(d, date(2023,1,1), date(2024,12,31)):
        m *= .72; tags.append("GLP1_shortage")
    if name in ("Insulin glargine 100 units/mL pen", "Semaglutide 0.25/0.5 mg pen") and d.year >= 2024:
        m *= 1.08; tags.append("diabetes_demand_growth")
    if name in ("Acetaminophen 500 mg tablet", "Ibuprofen 200 mg tablet") and between(d, date(2023,1,1), date(2023,3,31)):
        m *= 1.18; tags.append("2022_23_tripledemic_carryover")
    if category == "IV fluid" and between(d, date(2024,10,1), date(2024,11,30)):
        m *= .40; tags.append("baxter_helene_shortage")
    # FDA OTC switch: broadened naloxone access after 2023-03-29.
    if category == "opioid antagonist" and d >= date(2023,3,29):
        m *= 1.35; tags.append("naloxone_otc_switch")
    # H5N1 monitoring signal: low public risk, but a brief occupational-testing/antiviral bump.
    if name == "Oseltamivir 75 mg capsule" and between(d, date(2024,4,1), date(2024,7,31)):
        m *= 1.18; tags.append("h5n1_monitoring")
    return m, tags

def price_multiplier(d: date, category: str, name: str) -> tuple[float, list[str]]:
    tags=[]
    # Public-sector acquisition prices drift with medical inflation; CMS reports retail drug spending growth.
    annual = 0.025 if category in ("antibiotic", "OTC analgesic", "NSAID") else 0.035
    m = (1 + annual) ** ((d - START).days / 365.25)
    # ASPE: most list-price changes occur in January/July; a conservative generic change schedule.
    if d in (date(d.year,1,1), date(d.year,7,1)):
        m *= 1.03; tags.append("jan_jul_repricing")
    # Shortage-linked procurement premium / lower availability.
    if name == "Amoxicillin 500 mg capsule" and between(d,date(2023,1,1),date(2023,5,31)):
        m *= 1.10; tags.append("shortage_procurement_premium")
    if name == "Semaglutide 0.25/0.5 mg pen" and d.year <= 2024:
        m *= 1.12; tags.append("GLP1_procurement_premium")
    if name == "Naloxone 4 mg nasal spray" and d >= date(2023,3,29):
        m *= .92; tags.append("OTC_competition_discount")
    return m, tags

def main() -> None:
    path = OUT / "arkansas_clinic_daily_pharmacy_sales.csv"
    rows=[]; d=START
    while d <= END:
        for name, category, base, unit_price, _ in DRUGS:
            em, et = effects(d, category, name)
            pm, pt = price_multiplier(d, category, name)
            lam = base * em
            units = poisson(lam)
            # A small fraction of high-volume days are capped by clinic inventory; this creates censoring.
            stockout = 0
            if units > max(10, round(base * 1.8)) and RNG.random() < .08:
                units = max(0, round(base * 1.8)); stockout = 1; et.append("clinic_inventory_cap")
            price = round(unit_price * pm * (1 + RNG.gauss(0, .015)), 2)
            rows.append({"date":d.isoformat(),"state":"Arkansas","facility_type":"local government clinic","drug_name":name,"therapeutic_class":category,"units_sold":units,"unit_price_usd":max(.01,price),"revenue_usd":round(units*max(.01,price),2),"stockout_flag":stockout,"injected_event_tags":"|".join(et+pt)})
        d += timedelta(days=1)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {path} ({len(rows):,} rows)")

if __name__ == "__main__":
    main()
