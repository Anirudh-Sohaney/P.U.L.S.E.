# Global Economics Downloads

This folder now contains official bulk dataset archives for the global economics source set.

## Retrieved files

- `cache/WDI_CSV.zip` from the World Bank WDI bulk download page
- `cache/WEOApr2026all.xlsx` from the IMF WEO dataset page
- `cache/WS_LBS_D_PUB_csv_flat.zip` from the BIS bulk download portal
- `cache/pwt110.xlsx` from the Penn World Table 11 Dataverse record
- `cache/mpd2023_web.xlsx` from the Maddison Project Database 2023 Dataverse record

## Blocked source

- V-Dem Country-Year Full+Others v16 was attempted from the official download form, but the site returned HTTP 500 for programmatic submission in this environment. The source is documented in `download_manifest.json` and was not replaced with a synthetic or third-party file.

## Notes

- These are raw source archives, not yet normalized into a single tabular schema.
- The focus window for downstream work remains `2012-2022`.
- `sources_clean.json` and `sources_clean.md` document the source catalog; `download_manifest.json` documents the retrieval run.
