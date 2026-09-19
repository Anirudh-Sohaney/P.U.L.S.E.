# NLM RxNorm/RxClass NDC-to-ATC Crosswalk

This derived crosswalk maps the historical NDC keys in the local Arkansas HHS
Medicaid/CHIP pharmacy panel to RxNorm concepts and ATC therapeutic classes.
NLM documents both historical NDC lookup and RxClass relations through free
RxNav APIs. The mapping is a training-time identity/classification artifact;
the live application can refresh it from the same public services.

## Coverage

- 2,540 unique HHS NDC keys screened.
- 2,052 mapped to an RxNorm concept.
- 1,961 had at least one ATC relation.
- The mapped NDC subset represents 42.41% of observed HHS claim lines in the
  local panel; this is not a complete all-payer or all-NDC market measure.
- Claims are fractionally allocated across unique three-character ATC groups
  for an NDC with multiple relations, preserving total mapped demand without
  an arbitrary primary-class choice.
- Unmapped NDCs are excluded, never assigned to a class or treated as zero.

## Sources

- RxNorm API: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxNormAPIs.html
- RxClass API: https://lhncbc.nlm.nih.gov/RxNav/APIs/RxClassAPIs.html
- HHS source panel: https://opendata.hhs.gov/datasets/medicaid-provider-spending-ndc/

The crosswalk does not establish clinical indication, patient diagnosis, or
pharmacy inventory. ATC group is a therapeutic classification proxy.
