# Historical signal test fixture

`signals_2023_2025.csv.gz` is an exact copy of the project's existing local
`test/test_Signals/signals_2023_2025.csv.gz` benchmark artifact. Its SHA-256 is
`64d6c0f5f9a2efe36a76e8a2c89795f5a13c8270246eff4989f387ad0ac8cbed`.

The fixture lets a clean checkout test all 1,312 stable identities, preserved
historical conflicts, and unusable model output rules. It is test input only:
deployment startup imports the definitions manifest and does not publish these
historical rows unless an operator supplies the history file explicitly.
