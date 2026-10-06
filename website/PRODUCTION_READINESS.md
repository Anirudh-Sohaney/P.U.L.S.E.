# PULSE service and data readiness

This document records the current implementation and its remaining gates. It is
not a claim that every catalog value is current or that the platform is ready
for a public launch.

The root `.github/workflows/ci.yml` is configured to run the complete backend test suite,
frontend typecheck/build and runtime dependency audit, and a Compose smoke
test for the API, frontend proxy, built Chromium in-browser training,
encrypted plan reload without private CSV uploads, evaluated CMS baseline,
read-only signal date/provenance audit, verified backup and restore, worker
imports, cloud secret mounting, and production Caddy routing over a local
HTTP test listener. The smoke job fetches only
the committed CMS source artifact from Git LFS. The 55 KB historical catalog
is bundled under `website/catalog/`; signal API tests compare it with the
original fixture and do not depend on an ignored laptop directory. This replaces the former
`website/.github/workflows/ci.yml`, which GitHub did not discover from the
repository root and which checked only two files. Local tests can pass before
the workflow has run on GitHub; inspect the actual run before using it as a
release gate.

On 2026-10-05, the pinned production Python dependencies resolved with a
Python 3.14 wheel-only pip dry run, and `npm audit --omit=dev
--audit-level=high` reported zero vulnerabilities. These checks do not replace
the container build or the GitHub workflow run; neither has run from this
uncommitted worktree.

On 2026-10-05, an isolated local API database and Vite frontend passed
`scripts/browser_smoke.py --signal-years 3 --simulate-rate-limit`. The browser
requested three years of public history through the API, recovered from an
injected HTTP 429, trained locally, reloaded the encrypted plan, enforced idle
locking and all-session revocation, and sent no private CSV contents to the
API. The temporary account and database were removed after the run. This
checks the browser workflow against a development server; the Compose and
cloud images remain untested locally.

## Running locally

From `website/`, install `requirements.txt`, then run:

```bash
python -m scripts.import_signal_catalog
python -m backend.worker --once
python -m backend.main
```

Run `python -m backend.worker` as a separate long-running process for a daily
refresh at **16:00 UTC**. On startup, it catches up any source that has no
successful run since the most recent scheduled time. FDA Drugs, MedWatch,
Recalls, and Press RSS feeds provide official drug-event context. The Recalls
and Press adapters keep drug-related items from broader FDA feeds. Each news feed has its own
schedule and failure status, so a failed GDELT retry does not delay the FDA
feeds. `--once` forces every adapter to run.
The GDELT adapter retries transient connection failures and HTTP 5xx responses
up to three times with short backoff. It does not immediately retry HTTP 429;
the worker's six-hour failed-source cooldown applies instead. FDA successes
remain visible even when GDELT is rate limited.
It checks for failed sources hourly and retries them after six hours; the
unregistered BLS adapter waits until the next daily schedule so a partial
failure cannot exhaust its 25-query daily allowance.
Its hourly wait checks wall-clock time at least once a minute, so a laptop
resume triggers a missed-schedule check promptly. Sleep still interrupts
refreshes while the laptop is off.
The refresh worker holds an exclusive OS lock at
`website/data/.refresh_worker.lock` for its whole process lifetime. A second
worker pointed at the same data directory exits before calling any public
source, including when a systemd and Compose worker are started together.
The lock is released by the OS if the process crashes. Run workers only on a
shared mount with reliable file-lock semantics.
If the CMS source check succeeds after an earlier baseline run in the same
daily window, the worker schedules the baseline again at its next hourly check
so the ranking uses the newly committed source generation.
Run the frontend from `website/frontend/` with
`npm install` and `npm run dev`. The API is at `http://localhost:8000`; the
frontend is normally at `http://localhost:5173`. After importing or refreshing, run `python -m scripts.audit_signal_store` to
check all 1,312 identities, SQLite integrity and foreign keys, canonical signal
period dates, finite values, timezone-aware ingestion/source timestamps, and
the exact stored fields of all 4,097 rows from the SHA-256-pinned historical
catalog. Existing blank historical source URLs are filled from that source
file; changed nonblank provenance or numeric values fail the audit.
Blank source times are permitted only for historical catalog and news-bridge
rows whose publication time is unknown. `website/catalog/` includes the
1,312-ID manifest and an exact, SHA-256-pinned copy of the 6,484-row
2023–2025 historical research catalog. The source contains 2,387 byte-identical
duplicate rows; import stores 4,097 distinct records and stays idempotent.
Startup fails if the bundled file is
missing or changed, instead of silently booting a definitions-only service.
The import preserves each actual period and source timestamp and marks legacy
model outputs unusable for current publication. An explicit `--source PATH`
can replace the bundled file when the operator has separately verified it.
Docker Compose starts separate refresh and backup workers and mounts the official CMS Part D
normalized source and its manifest; the refresh worker stages new annual releases in
its writable data volume after checking CMS's live catalog.
The worker also checks the official HHS Medicaid Provider Spending by NDC
dataset page against the pinned July 24, 2026 ZIP version and SHA-256 in
`catalog/hhs_ndc_source.json`. A changed release is recorded as a failed source
check and appears in `/api/v1/signals/freshness`; it requires a new verified
extraction and model evaluation before the training-only ATC data can be used.
The check does not download the 321 MB source on every run or publish ATC values.

On this WSL laptop, systemd is enabled. The local units in
`website/deploy/systemd/` have been installed and enabled as `pulse-api`,
`pulse-worker`, and `pulse-frontend`. They restart crashed processes and start
when the WSL distribution boots. The API binds to `127.0.0.1:8000`; the built
frontend preview is at `http://localhost:3141` and proxies `/api` to the API.
Use `systemctl status pulse-api pulse-worker pulse-frontend` and
`journalctl -u pulse-worker -n 100` to inspect them. These units contain this
laptop's absolute paths and run under its current root-owned WSL checkout;
adapt the working directory, service user, TLS ingress, and secret management
for the eventual cloud host. Laptop sleep, shutdown, or a stopped WSL instance
interrupts the service, so this is not a claim of cloud-style 24/7 uptime.
The local units use an owner-only file umask, isolated temporary directories,
and `NoNewPrivileges`. Move them to a dedicated unprivileged service account
when the cloud filesystem and secret mounts are prepared.

The local `pulse-backup.timer` runs at **04:00 UTC** with catch-up after a missed
timer event. It calls `python -m scripts.backup_databases`, which uses SQLite's
online backup API for both databases, includes the committed CMS Part D source
and its matching manifest, and verifies SHA-256 and SQLite integrity,
and retains the latest 14 committed snapshots under `website/data/backups/`.
When a published drug ranking is present, the backup verifier also requires
its source SHA-256 and source year to match the included CMS source generation.
A backup taken across a source/ranking refresh boundary fails before commit;
the local systemd service retries failed backups after 30 minutes, while the
Compose backup worker retries at its next hourly check.
Snapshot creation and retention hold an exclusive OS lock at
`website/data/backups/.backup.lock`, so the laptop timer and a Compose backup
worker cannot modify the same snapshot directory concurrently.
Backup directories are mode `0700`; database files are `0600`. To inspect a
snapshot, run `python -m scripts.backup_databases --verify PATH_TO_SNAPSHOT`.
Run `python -m scripts.rehearse_restore PATH_TO_SNAPSHOT` to boot the API
against a temporary restored copy and check the 1,312-ID catalog, at least
4,097 bundled historical catalog records in the backup itself, signal date and
provenance audit, drug ranking, and CMS source pair without changing the live
databases. It also checks the restored account table and exercises registration,
login, authenticated access, and session revocation using a disposable account
only in the temporary copy. The historical count is checked before API startup can refill it.
Older snapshots with blank historical source URLs receive only that narrow
source-backed migration in the temporary copy before the exact-row audit;
missing rows or changed values still fail rehearsal.
To prepare an encrypted copy for separate storage, create a distinct 32-byte
backup key and keep it outside the data directory and the copy destination:

```bash
umask 077
python -c "import pathlib,secrets; pathlib.Path('backup-key.hex').open('x').write(secrets.token_hex(32) + '\n')"
python -m scripts.encrypted_backup export PATH_TO_SNAPSHOT PATH_TO_COPY.enc --key-file backup-key.hex
python -m scripts.encrypted_backup restore PATH_TO_COPY.enc PATH_TO_EMPTY_RESTORE_DIR --key-file backup-key.hex
python -m scripts.rehearse_restore PATH_TO_EMPTY_RESTORE_DIR
```

The export verifies the source snapshot, encrypts the complete database and
CMS source pair with AES-256-GCM, then decrypts and verifies the bundle before
publishing it. Restore rejects a wrong key, modified ciphertext, unexpected
archive members, and an existing destination. Transfer the encrypted bundle to
separate storage and retain the key in a separate secret store; neither transfer
nor key escrow is configured on this laptop. The cloud Compose backup worker
automates export to its configured mount and retries failed exports at the next
hourly check. It authenticates existing bundles before counting them as done.
Losing the key makes the exported
copy unusable. The key and bundle must not be committed to Git.
For recovery, stop the API and worker, verify the snapshot, copy its two
database files and CMS source/manifest pair into `website/data/` with owner-only
permissions, then restart the services. Preserve the server signing key
separately in a secret manager; the backup intentionally contains no private
sales or inventory CSVs or signing key. These local
snapshots protect against database corruption or accidental changes, but a
cloud deployment still needs a provisioned independent storage mount
and a tested restore procedure for other source artifacts as well. Compose runs its own
backup worker at 04:00 UTC, catches up after downtime, verifies existing
snapshots, and retries failed backups hourly. The WSL timer is specific to this
laptop and should not run alongside the Compose backup worker against the same
data directory.

The Compose frontend uses a multi-stage Node 22 build and Nginx to serve static
assets and proxy same-origin API and documentation routes. The local Compose
ports bind to `127.0.0.1`; a later cloud deployment needs an HTTPS ingress and
explicit public network policy. The frontend dependency lockfile is included
for reproducible builds. Interactive `/docs` serves a pinned Swagger UI bundle,
stylesheet, and favicon from the API's own `/api-doc-assets/` path. Its
initializer is a separate local script, so the docs CSP permits only
same-origin scripts and styles. The vendored package license and SHA-256 are
in `backend/docs_assets/`; `/redoc` redirects to these local docs. The site's
own `/api-guide` is also served locally. Nginx and the
cloud Caddy API route each reject request bodies larger than 256 KiB;
public API calls are small JSON requests, while private sales and inventory
CSV files stay in the browser and never reach this proxy. On 2026-10-03,
`npm audit --omit=dev` reported zero
findings, while the full build-dependency audit reported five high findings
through Tailwind 3's `braces` dependency. The [upstream advisory](https://github.com/advisories/ghsa-vfj7-8cjw-p6xm)
lists no patched `braces` release. Node dependencies stay in the disposable
build stage; the final Nginx image copies only static assets. Track this
build-time finding and reassess the Tailwind upgrade when a compatible path is
available rather than claiming the full audit is clean.
The backend image uses the official Python 3.14 slim image and a 35-package
exact-version runtime lock. Legacy server-side XGBoost and scikit-learn are
absent. A clean Python 3.14 environment installed that lock with wheels only,
loaded the API and fourteen worker tasks, registered and authenticated an
Argon2id account. The previous lock also evaluated the four CMS baseline folds,
served the 1,312-ID catalog, and passed a restored snapshot rehearsal; the
new lock adds only Argon2id bindings. Docker is unavailable on this laptop, so the container
image itself has not yet been built here.
The Compose API, refresh worker, and backup worker run as the unprivileged
`pulse` user (UID 10001).
Before they start, a short-lived `data-permissions` service changes ownership
of the bind-mounted `website/data/` tree to that UID, including existing
databases and backups. This changes host file ownership when Compose is first
started; the root-run laptop systemd units can still access these files.
Avoid starting the systemd and Compose workers together against the same
database. The setup service uses root only for this ownership step.
The long-running application containers drop Linux capabilities and prohibit privilege
escalation. The local systemd units remain root-run because this checkout is
under `/root`; the cloud deployment should move the checkout and data directory
to paths accessible by a dedicated service account.

For a later cloud host, `docker-compose.cloud.yml` adds Caddy as the public
HTTPS ingress. Set a real hostname in `PULSE_DOMAIN`, point its DNS at the
server, and allow inbound TCP 80 and 443. Caddy routes `/api/*`, `/docs`, and
`/openapi.json` directly to the backend, and other paths to the frontend.
The API also limits individual signal IDs to 64 characters and search terms
to 120 characters for direct backend requests.
The public signal API has a per-client token bucket: 300 request credits at
startup, refilling at five per second. History/latest calls cost five credits,
catalog/coverage cost three, and other signal calls cost one. Exhaustion
returns HTTP 429 with `Retry-After`; authentication and readiness checks are
outside this budget. The API trusts `X-Real-IP` only from the configured
frontend or Caddy peer, and Uvicorn does not process client-supplied forwarded
headers. This limiter is process-local and assumes the current single API
process; a multi-process or multi-host deployment needs a shared ingress
limit. Network-level abuse protection remains the cloud operator's task.
It sends HSTS, frame-denial, MIME-sniffing, referrer, and permissions
headers on HTTPS responses, including routes proxied directly to FastAPI.
The direct `/docs` and `/redoc` routes also receive the docs-specific CSP used
by the local Nginx proxy. A local Caddy 2.11.7 reverse-proxy rehearsal verified
these headers and HTTP 413 for a request body above 256 KiB.
It overwrites `X-Real-IP` with the connecting client address; the backend
trusts only the Caddy container as a forwarding peer for login limits. Do not
put an additional CDN or proxy in front without configuring its trusted IPs
in Caddy. The cloud overlay sets secure cookies, same-origin CORS, and the
exact `https://` public origin. The base Compose file still binds its API and
frontend host ports only to loopback.

Create `website/.env` from `.env.cloud.example` and replace the example
hostname and `PULSE_OFFSITE_BACKUP_DIR` with the absolute host path to a
storage mount that survives loss of the application host. Provision that mount
before starting Compose and make it writable by UID 10001. A directory on the
same host disk does not provide host-loss recovery. Generate distinct signing
and backup encryption keys into ignored files under `website/secrets/`. Keep
`.env` owner-only (`0600`), and set both secret files to owner-readable and
group-readable (`0640`) with
group ID 10001. The host Compose user remains its owner; the container's
`pulse` user belongs to group 10001. File-backed Compose secrets preserve
host file permissions on Linux, so a root-owned `0600` file would be unreadable
to the unprivileged processes. The API, refresh worker, and backup worker read
the signing key through `/run/secrets/pulse_signing_key`; only the backup
worker reads `/run/secrets/pulse_backup_key`. The cloud overlay does not pass
either value in a container environment variable. Start from `website/` with:

```bash
umask 077
cp .env.cloud.example .env
# Edit .env to set the real hostname and independent storage mount path.
mkdir -p secrets
python -c "import pathlib,secrets; pathlib.Path('secrets/pulse_signing_key').open('x').write(secrets.token_urlsafe(48))"
python -c "import pathlib,secrets; pathlib.Path('secrets/pulse_backup_key').open('x').write(secrets.token_hex(32) + '\n')"
sudo chown "$(id -u):10001" secrets/pulse_signing_key secrets/pulse_backup_key
chmod 600 .env
chmod 640 secrets/pulse_signing_key secrets/pulse_backup_key
git -C .. lfs pull --include="data/targeted_additions/cms_partd_geography_drug/data/arkansas_partd_geography_drug_by_year.csv.gz" --exclude=""
python -m scripts.cloud_preflight
docker compose -f docker-compose.yml -f docker-compose.cloud.yml config --quiet
docker compose -f docker-compose.yml -f docker-compose.cloud.yml up -d --build
```

The preflight rejects an unhydrated Git LFS pointer, a CMS source that fails
its manifest checksum, placeholder domains, missing or insecure secret files,
and an absent or unwritable backup directory. It cannot prove that the backup
directory is an independent mount or that DNS and TLS are ready; verify those
on the chosen server before exposing the service.

Keep the Caddy `/data` volume across deployments because it holds issued TLS
certificates. The Compose backup worker creates verified local snapshots and
exports encrypted bundles to the configured mount. Confirm that mount is
actually remote or independently replicated, and rehearse recovery using the
chosen storage service before relying on it for host-loss recovery. The overlay is a deployment
template: no cloud domain, DNS, firewall, signing key, or off-host mount has
been configured here. CI is configured to validate its Compose merge,
Caddyfile syntax, production API/worker/backup startup configuration, and a
backup restore once the workflow runs. A verified standalone
Docker Compose 5.6.0 binary merged the base and cloud files successfully, and
Caddy 2.11.7 validated the formatted Caddyfile locally. Docker Engine is
unavailable on this laptop, so the containers and TLS ingress have not been
run here.
Compose probes the refresh worker's database heartbeat every five minutes and
requires a verified backup after the most recent 04:00 UTC schedule. In the
cloud overlay, the backup probe also authenticates the encrypted export and
checks that it matches the verified local snapshot. Run
`python -m scripts.operational_status worker` or `backup` from `website/` to
inspect the same checks as JSON; an unhealthy component exits nonzero. Run
`python -m scripts.operational_status sources` to make the command fail when
an expected source was not checked since the latest daily schedule or its
most recent refresh failed. The worker report lists those sources without
conflating an external outage with a stalled worker. These probes expose
failures to the container runtime; route unhealthy-container and failed-source
events to the chosen host's alert service before a public launch.
When `ENVIRONMENT=production`, startup requires a nontrivial signing key of at
least 32 bytes, secure cookies, a bounded session lifetime, disabled legacy
server training, `PUBLIC_ORIGIN` set to the exact external HTTPS origin, and
either no cross-origin access (`CORS_ORIGINS=[]`) or explicit HTTPS origins.
The laptop's localhost CORS defaults are rejected in production. Configure the
actual domain and TLS ingress before switching modes. Both the API and worker
need `PUBLIC_ORIGIN` and `CORS_ORIGINS` because both load the same production
configuration.
Production registration and cookie-session login require an `Origin` header
matching `PUBLIC_ORIGIN`; this closes the headerless form path for login CSRF.
The separate bearer-token endpoint accepts headerless API clients and sets no
session cookie.

The API process is a reader and account server; the worker owns network
refreshes. This prevents each API replica from running the same scheduled job.
The `refresh_runs` table records success and failure with UTC timestamps.
The worker records a heartbeat before and after each hourly source check.
`/api/v1/signals/freshness` reports whether a heartbeat was recorded within
two hours and shows each source's last attempt, last success, and latest
status. It also returns `failed_sources` so a fallback source failure is
visible even while the worker is healthy; the signals page displays that
partial-failure warning. The signals page warns when the worker is stale. `/health` checks API
process liveness; `/ready` requires the 1,312-ID catalog and all 4,097 bundled
historical catalog records. Both local systemd and Compose run the read-only
signal-store audit after import and before starting the API. Neither
claims that every upstream source has published a current observation.

## Public signal API

All routes are also described at `/docs` and in the frontend API guide.

| Route | Meaning |
| --- | --- |
| `GET /api/v1/signals/catalog` | All 1,312 stable IDs, identity fields, units, source, and last period. Search and pagination are supported. |
| `POST /api/v1/signals/latest` | Latest usable recorded observations in `values` for up to 1,500 IDs, including the full 1,312-ID catalog in one call. `latest_recorded_values` contains the absolute latest stored row for every requested ID with a record, including unvalidated rows flagged `usable: false`. For IDs without a usable value, `unusable_recorded_values` highlights that archived row; `missing_ids` and `missing_details` report the usable-value gap. Unknown IDs have no recorded row. Date filters are rejected; unresolved same-revision conflicts carry `ambiguous: true`. |
| `POST /api/v1/signals/history` | Recorded values for up to 100 IDs on an exact `date` or any `start_date`/`end_date` range, including the full available 2013–present history. The newest source revision for each period is returned by default; `include_revisions: true` returns each recorded revision for point-in-time training. Missing dates are absent. A request exceeding 10,000 raw source rows returns HTTP 413 with no truncated result; split IDs or dates. Browser training retries smaller ID batches automatically. |
| `GET /api/v1/signals/freshness` | Catalog counts, latest periods, and last worker run. |
| `GET /api/v1/signals/coverage` | Per-ID gap status and age of the latest observation. |
| `GET /api/v1/signals/recent` | Newly recorded official external observations in the last 1–7 days, with both source period and recording timestamp. Unresolved same-revision conflicts are omitted; inspect `/history` for them. |
| `GET /api/v1/signals/demand/drugs` | Searchable, paged ranking of the evaluated Arkansas CMS Part D claims baseline. Until a baseline run succeeds, the endpoint returns an explicit unavailable status instead of promoting unvalidated legacy outputs. |
| `GET /api/v1/signals/news/recent` | Recorded news or FDA feed updates from the requested recent window. Each row includes `timestamp_kind`: `gdelt_first_seen` or `rss_pub_date`. The legacy `published_at` field holds that source timestamp; it is not always the publisher's publication time. `source_checks` reports each feed's latest attempt and success separately, including failures when fallback items are present. |
| `GET /api/v1/signals/sources/shortages/recent` | Recent changes in the latest complete openFDA shortage fetch, with its UTC snapshot date and retrieval time. Supply context, not demand. |
| `GET /api/v1/signals/sources/nadac/latest` | Latest official NADAC rate snapshot grouped by rate classification and pricing unit. Price context, not demand. |
| `GET /api/v1/signals/sources/sdud/latest` | Latest Arkansas Medicaid quarterly prescription counts by product name. Suppressed NDC rows remain unknown, so reported totals are lower bounds. Search and pagination are supported. |

The stable ID hashes signal origin, measure, cadence, geography, and entity.
Each row retains its period end, signal date, source timestamp when supplied,
ingestion time, source URL, source kind, forecast horizon, and quality fields. A source
timestamp of `""` means the historical artifact did not supply one; it is not
silently replaced with the import time. Conflicting source rows are preserved,
and callers should not select one without resolving the source discrepancy.
Browser training requests all revisions and uses the newest period and revision
available on each historical feature day. A later revision to an older period
cannot replace a newer published period. Values with unknown source
availability remain excluded from point-in-time training.
An early schema used a separate `signal_observations` table. Startup retains
that table if found: matching total row counts cannot prove its records were
migrated. Remove it only after a row-level audit of that database.

## Current coverage, as of 2026-10-03

The imported historical file has 1,312 distinct signal identities and 4,097
distinct retained source rows. It ends in 2025 for weekly/monthly observations
and in 2024 for the annual drug output's observation period. The legacy catalog
drug states target 2025. **Those rows are not current 2026 medication demand rankings.** The
catalog also contains 120 same-ID/same-period groups with conflicting numeric
values; these are flagged by the history endpoint.

A separate checked-in news bridge has 20 signal columns and 97 monthly periods
from January 2018 through January 2026. Its bundled copy is SHA-256 pinned;
all 720 values overlapping the 2023–2025 historical catalog match exactly.
Startup backfills only absent periods (1,220 rows on the laptop), retaining the
catalog's original rows. The bridge has no verified per-row publication time,
and its upstream extraction runner is absent. Its rows therefore have an empty
source timestamp, are marked unusable in `/history`, cannot be selected for
point-in-time browser training, and remain excluded from `/latest`. The January
2026 period is a historical artifact value, not a current news-model refresh.

The gap report now marks 1,212 legacy drug IDs with a separate evaluated
two-year baseline projection for 2026 and 18 ATC demand outputs that still
require current source data and a validated model rerun. The 20 news-model
outputs have no verified live refresh path; 62 external variables have
separate source-specific status.
The Medicaid state-performance adapter refreshes eight Arkansas monthly
identities. Its latest published reporting period is June 2026; all eight
values are flagged preliminary. The official API provides period and revision
status but no row publication timestamp, so the first retrieval time is stored
as the known availability time. It prefers final updated rows for months where
both preliminary and final versions are present. The live refresh recorded 650
nonmissing values across 110 monthly periods and wrote no duplicates on a
repeat run.
The BLS adapter now refreshes three raw monthly series, 30 lag, difference,
and rolling identities, and six seasonal identities. Its lag and rolling
formulas were checked against 852 historical catalog values without a mismatch;
the seasonal baseline and anomaly were checked against all 210 historical
catalog values without a mismatch. The seasonal baseline uses all prior years
from the same calendar month, starting with 2000 observations. The adapter
requests official BLS v1 history in ten-year windows and retains gaps instead
of filling missing months. It published 2026-08 observations for all 39 IDs.
The CMS geographic-variation adapter checks nine national annual identities
against the official dataset. The source still ends at 2024, matching all
2023 and 2024 values already in the catalog. The worker records when it checked
the published rows without relabeling them as 2026 observations. A repeat run
wrote no duplicate values. The gap report now marks 56 external IDs as having
live source observations and six as having collapsed identity dimensions. A live-source status describes
the check, not the age of the upstream observation; read each row's period.
The two event-derived FDA IDs need identity repair before live publication:
`shortage_active` has one date with conflicting 0/1 values, and
`recall_active` has 92 such dates. Their current national IDs collapse
different event rows into one identity, so a single current value is not
well-defined. The API preserves and flags those conflicts.
The four NADAC catalog IDs also collapse their original rate-classification
and pricing-unit dimensions. The January 2023 rows mix prices per each and
per millilitre. The daily worker now records the official 2026 NADAC snapshot
separately, preserving those dimensions. The first live snapshot is dated
2026-09-30 and contains 30,079 source rows in 11 groups. It is available at
`/api/v1/signals/sources/nadac/latest`; it does not replace the ambiguous
legacy catalog values with a made-up aggregate. All six collapsed IDs are
history-only; `/latest` reports `identity_dimensions_collapsed` for them.
Daily source capture updates matched external-variable IDs. The separate
annual claims baseline appends 2026 projections for 1,212 matching drug IDs
while retaining their 2025 forecast horizon in history.
The drug ranking endpoint selects only evaluated v2 runs; an older v1 run
may remain in the database for audit but cannot become the published ranking.
The worker does not
write current values for the 18 ATC or 20 news-model IDs. The historical ATC
rows came from the legacy pipeline's zero-filled live feature vector;
`/latest` omits them and `/history` marks them unusable. The 20 historical
news-model features remain in `/history`, marked as historical-only;
`/latest` omits them until a verified live refresh path exists. This prevents an FDA
shortage record or an article timestamp from being misrepresented as a drug
demand prediction.
The `/latest` API returns `missing_details` for IDs without a usable value,
with a coverage reason and last recorded period; unknown IDs receive
`unknown_id`. Its separate `unusable_recorded_values` array lets callers
retrieve the absolute latest archived row without presenting it as a current
observation or valid demand estimate.
`latest_recorded_values` exposes the newest stored row for every requested ID,
even when an older validated row remains in `values`; callers must inspect
`usable` and `unusable_reason` before interpreting it.
The publication check is fail-closed: a newly inserted ATC or news-model row
with a different `source_kind` still remains unusable, and a Part D drug-model
row is eligible for `/latest` only with the evaluated
`cms_partd_two_year_persistence_v2` source kind. A future validated pipeline
must add an explicit reviewed promotion rule before its rows can be published.
Historical drug model states remain outside `/latest`'s `values` array until
the evaluated claims baseline has a corresponding row. They appear only in
`unusable_recorded_values`, so a partial or failed baseline run cannot
silently fall back to an old state.

The worker captures the official 2026 Arkansas State Drug Utilization Data as
a separate observed source. The first live snapshot contains only 2026 Q1. Of
22,703 NDC/utilization rows, 12,264 have suppressed prescription counts. The
API exposes the remaining reported count as a lower bound with the source
period and suppression count. The public signals page includes a collapsible,
searchable table of the latest quarter's reported product names, sorted by
that lower bound, with suppressed row counts visible. Names may be abbreviated
in the source. These quarterly statewide prescriptions cannot
be spliced into the monthly pharmacy-provider ATC target without a validated
cross-source model, so the 18 ATC outputs remain unpublished.
The checked-in NDC-to-ATC crosswalk covers only 2,236 of the 22,703 current
SDUD rows (9.8%), representing 44.9% of the reported, unsuppressed
prescriptions. It was built for the historical pharmacy-provider NDC set;
extending it requires a current, source-backed NLM mapping and an evaluated
bridge across the different target definitions.
A reproducible 24-month persistence stress test of the existing HHS/ATC panel
(`PYTHONPATH=model python model/scripts/audit_atc_24_month_baseline.py`)
compared 797 observed class-month pairs across all 18 IDs. It scored 40.5%
five-state accuracy, 35.2% balanced accuracy, and 122.4% weighted absolute
percentage error. Its source still ends in December 2024; this exploratory
proxy is not eligible for live publication. The separate research evaluator's
next-month score does not validate a nearly two-year gap to the present.

CMS currently labels **2024** as the latest available year for its Medicare
Part D Geography and Drug dataset. The local HHS Medicaid Provider Spending by
NDC artifact ends in **December 2024**; the [official federal release notice](https://www.govinfo.gov/content/pkg/FR-2026-07-23/pdf/2026-14947.pdf)
also defines that source's published claim window as January 2018–December
2024. Those sources cannot supply direct
2026 pharmacy demand observations. See the [CMS dataset](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-geography-and-drug)
and [HHS catalog](https://opendata.hhs.gov/). The old `model/prod_pipeline.py`
previously used a zero-filled ATC inference placeholder. That emission path is
now disabled; the script cannot overwrite its checked-in legacy example or
publish a live-looking forecast without a new, evaluated runner.

The legacy `model/artifacts/evaluation/publishability_audit.json` reports
`publishable: false` with 15 failed diagnostics. The current project auditor
explicitly treats that legacy report as nonblocking compatibility evidence;
it must not be presented as the sole reason the daily model is unavailable.
On 2026-10-05, rerunning `model/scripts/audit_project_status.py` against the
current artifacts independently returned `proxy_library_ready: false`,
`learned_architecture_ready: false`, and `project_complete: false`. Its
blocking failures were proxy coverage, the learned end-to-end accuracy
contract, and missing research-exhaustion evidence. A stricter rerun now also
rejects the operational forecast surface: it has 43,366 unique rows across 14
targets, but only the annual Part D target is qualified in the metric-library
audit. The other 13 targets retain stale qualified labels in that research
artifact; they cannot pass the project status gate. These results
do not prove that the forecast rows can be regenerated from today's inputs.
The historical metric feature-store and NADAC panel builders were restored
from repository history after model tests exposed their missing imports. The
feature-store key now includes therapeutic class and pathogen, preserving 52
ATC class rows that otherwise collided under the older key. On 2026-10-05,
the local ignored research forecast was found to contain 43,366 rows across 14 targets,
while the current metric audit qualifies only the annual Part D target. The
builder now requires that audit and fails if rejected targets are present
unless `--qualified-only` is explicit. The fixture packager records source and
audit hashes and writes only the 1,212 qualified rows to a tracked, 23 KB
forecast snapshot. CI verifies the fixture hashes and builds one feature from
it in a clean checkout; the filter behavior has a separate focused test.
This verifies an audit-qualified feature-store
transformation, not the model's live publishability.
The broader model research suite is not yet green: some tests still expect
proxy promotions that the checked-in metric audit now rejects, and many
research inputs remain unhydrated Git LFS pointers. Those tests and artifacts
need source-backed reconciliation before they can serve as a release gate.
The referenced `existing_models/news_signal_model` directory is absent, and
`model/artifacts/trained/` contains no files. A repository-wide search on
2026-10-05 included hidden and ignored files, model and data subtrees, Git LFS
paths, every reachable commit, and three recovered checkpoint commits reported
by `git fsck --no-reflogs --unreachable`. It found a dated 97-row, 20-column
news output CSV and completion metadata in `6d13c14`, plus a recovery README
in `d6ca4b7` explicitly stating that the original inference code, weights,
and validation artifacts were absent. The current `news_signals.py` is a
separate article-event feature layer; it does not generate those 20 named
news-model columns. Restoring the historical files would therefore recover
old observations, not a runnable daily generator.
The available GDELT article research corpus has 354 retained articles from
2023–2025, including 292 full-text rows, while the saved news signal table
contains 97 monthly observations from 2018–2026. That sampled corpus is not
the original monthly extraction input and cannot reproduce or validate its
20 outputs by itself. A replacement needs its own dated acquisition contract,
article-level category labels or other independent validation, and a measured
chronological evaluation before publication under a new model version.
Completing the requested daily
model therefore still requires a reproducible news-model runner and its
trained artifact, compatible current ATC inputs or an explicitly evaluated
new target, and chronological validation of the resulting daily pipeline.
To repeat the audit without replacing checked-in evidence:

```bash
PYTHONPATH=model website/.venv/bin/python model/scripts/audit_project_status.py --output /tmp/pulse-project-status.json
```

The Arkansas annual Part D five-state forecast currently uses the latest
published year's claim bucket as a next-year persistence state; its evaluation
reports identical 0.891 model and persistence accuracy. These artifacts do
not establish a validated 2026 drug-demand output, even though the historical
2025 states are useful as clearly dated public context.

A separate annual baseline job now reads the official Arkansas Part D
generic-drug claim file through 2024 and publishes **1,324 projections for
2026** to the public drug-ranking endpoint. It carries each drug's observed
claims category forward two years, with thresholds learned only from prior
published claim years. Four rolling holdouts (target years 2021–2024) achieved
84.2% mean balanced five-state accuracy. Training, validation, and test labels
respect the annual publication lag. The worker reruns the evaluation daily,
records the source SHA-256 and metrics, and writes no duplicate rows when the
source is unchanged. The bundled source manifest now pins its SHA-256, and
runtime source generations use a SHA-256 filename. Readers verify the file
bytes against that digest before evaluating, publishing, or backing up a
ranking. A mismatch fails the refresh and leaves the prior ranking in place.
The output is an annual Medicare Part D claims proxy,
not an estimate of current dispensing, pharmacy inventory, or units to order.
The drug-ranking API returns the four historical annual-claims cutoffs used to
assign its five ordinal states as `state_thresholds_claims`; it does not turn
those states into pharmacy purchase units.
The existing 1,312 catalog IDs and their original 2025 records remain intact.
The 1,212 matching drug IDs have an additional 2026 forecast-horizon record;
`/latest` selects that horizon, while `/history` retains both horizons. The
separate ranking includes 112 newer drugs that have no legacy catalog ID.
This baseline does not clear the universal model audit, restore the missing
news-model runner, or provide a daily news-conditioned drug forecast.

## Live inputs and keys

| Input | Current implementation | Credential needed |
| --- | --- | --- |
| [openFDA drug shortages](https://open.fda.gov/apis/drug/drugshortages/how-to-use-the-endpoint/) | Daily full snapshot, source `last_updated`, raw record JSON, record change dates, and an immutable membership list for each successful fetch. The latest API view reads one complete fetch even when the source changes twice on the same UTC day. It does not refresh a model signal. | A free `OPENFDA_API_KEY` is recommended for sustained deployment; the local two-page daily pull works without a key under [openFDA's lower unauthenticated limits](https://open.fda.gov/apis/authentication/). |
| [GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) | Daily recent article metadata, title relevance, and GDELT `seendate` (first seen by the feed; publisher publication time is not verified). This endpoint has intermittently returned connection resets and HTTP 429 from the local network; its attempt and error are recorded separately. [GDELT confirms rate limiting and request shedding](https://blog.gdeltproject.org/scaling-gdelt-for-a-new-era-migrating-to-spanner-with-agentic-interactive-gemini/). | No key for the DOC API, but outbound access to `api.gdeltproject.org` is needed. |
| [FDA Drugs RSS](https://www.fda.gov/about-fda/contact-fda/subscribe-podcasts-and-news-feeds) | Checked independently each day for recent FDA drug approval, recall, shortage, and safety event titles, even when GDELT fails. A named medicine can qualify without a generic “drug” keyword; generic approval-notification index pages are excluded. A feed update is not necessarily the date a medical event occurred. | No key. |
| [FDA MedWatch RSS](https://www.fda.gov/safety/medwatch-fda-safety-information-and-adverse-event-reporting-program/medwatch-rss-feed) | Checked independently for drug-keyword safety alerts. Device-only paths and items without drug terms are excluded; an empty three-day window stays empty. The RSS date is the feed item date, not a demand observation. | No key. |
| [FDA Recalls RSS](https://www.fda.gov/about-fda/contact-fda/subscribe-podcasts-and-news-feeds) | Checked independently for recent drug-related recall notices. Food-only notices are excluded. The RSS date is the feed item date, not a demand observation. | No key. |
| [FDA Press Releases RSS](https://www.fda.gov/about-fda/contact-fda/subscribe-podcasts-and-news-feeds) | Checked independently for recent drug-related announcements. Device-only and general regulatory releases are excluded. The RSS date is the feed item date, not a demand observation. | No key. |
| [BLS Public Data API](https://www.bls.gov/developers/api_faqs.htm) | Daily poll of three published monthly series through unregistered v1; records source months and preliminary flags, and derives the catalog's preceding-observation lags, differences, rolling values, and same-month historical baselines/anomalies. The first retrieval time is stored because the response omits a publication timestamp. | No key for v1. The current three-series history refresh uses nine queries per daily run, within its 25-query/day limit; registered v2 has higher limits but is not used. |
| [Medicaid and CHIP enrollment and performance](https://data.medicaid.gov/dataset/6165f45b-ca93-5bb5-9d06-db29c692a360) | Daily poll of eight Arkansas monthly measures via the official data API. Preliminary and final revisions are distinguished; absent cells are skipped. The first retrieval time is stored because the row publication timestamp is unavailable. | No key for the public data API. |
| [CMS Medicare Geographic Variation](https://data.cms.gov/summary-statistics-on-use-and-payments/medicare-geographic-comparisons/medicare-geographic-variation-by-national-state-county) | Daily check of nine national annual catalog measures. The latest official year is 2024 as of this check; no 2025 or 2026 value is inferred. | Public data API, no key. |
| [NADAC 2026](https://data.medicaid.gov/dataset/fbb83258-11c7-47f5-8b18-5f8e79f7e704) | Daily full latest-date snapshot; computes rate counts and min/mean/max within each rate classification and pricing unit. It does not publish to four legacy IDs with collapsed dimensions. | No key for the public data API. Annual dataset ID rollover must be checked before 2027. |
| [CMS Part D Geography and Drug](https://data.cms.gov/provider-summary-by-type-of-service/medicare-part-d-prescribers/medicare-part-d-prescribers-by-geography-and-drug) | The daily worker checks CMS's `data.json` catalog for published annual CSVs and the latest CSV's `Last-Modified` header for same-URL revisions. It aggregates Arkansas state rows by generic drug and total claims, commits a versioned normalized panel through an atomic manifest pointer, then reruns the evaluated two-year persistence baseline. After an outage it imports every newer published year in order; unpublished years stay absent. As of this check, 2024 remains the latest release. The stronger news-conditioned runner is unavailable. | Public download/API, no key required. |
| [HHS Medicaid Provider Spending by NDC](https://opendata.hhs.gov/datasets/medicaid-provider-spending-ndc/) | Historical local source through 2024-12. The official July 2026 release also describes coverage ending in December 2024; no 2025 or 2026 monthly claim line is inferred. | Public download, no key identified. |
| [CMS State Drug Utilization Data 2026](https://data.medicaid.gov/dataset/2957a7f9-9a15-453e-9afd-3bbdcbac8fd3) | The worker discovers the latest annual SDUD release from the official Medicaid metastore each day, then captures Arkansas rows by NDC and utilization type. It stores the catalog modification time, retrieval time, and separate content revisions. Suppressed counts remain unknown. On 2026-10-03 only Q1 was published: 22,703 Arkansas rows, 12,264 suppressed. The 1,368,827 reported prescriptions are a lower bound. This is a quarterly statewide Medicaid source, not the monthly pharmacy-provider target of the historical ATC model. | No key for the public data API. |
| CDC, NOAA, NWS, FEMA, NADAC | Research-package adapters exist, but the 1,312-signal production refresh has not been connected and verified. | Check each official source's current access rules before deployment. |

Run `python -m scripts.audit_atc_bridge` from `website/` to compare the
latest stored Arkansas SDUD period with the manifest-verified historical
RxNorm/ATC crosswalk. For the 2026-Q1 snapshot checked on 2026-10-04, only
1,205 of 14,510 distinct SDUD NDCs (8.3%) map to an ATC group in that
crosswalk. Mapped rows account for 614,797 of 1,368,827 **reported**
prescriptions (44.9%); another 12,264 rows have suppressed, unknown counts.
All 18 target ATC groups appear somewhere in the mapped subset, but that does
not make the subset representative. The audit reports source and mapping
hashes and fails if the mapping no longer matches its source manifest. The
quarterly statewide prescription source differs from the historical monthly
pharmacy-provider claim-line target, so it cannot supply a valid current
feature vector for the 18 model IDs. This remains true even if the crosswalk
is expanded; source and outcome compatibility must be evaluated separately.

On 2026-10-05 UTC, GDELT still returned HTTP 429 after scheduled retries.
The FDA Drugs feed succeeded and recorded one recent drug-event item. A third
adapter now reads the [official FDA MedWatch RSS feed](https://www.fda.gov/safety/medwatch-fda-safety-information-and-adverse-event-reporting-program/medwatch-rss-feed)
and retains drug-keyword safety items, excluding device-only alerts.
The fourth adapter reads FDA's official Recalls RSS feed and retains drug-related
notices while excluding food-only recalls. Its first local run found no
qualifying items in the three-day display window.
The fifth adapter reads FDA's official Press Releases RSS feed and retains
drug-related announcements while excluding device-only releases.
The freshness endpoint keeps independent status for all five feeds; working
FDA feeds do not claim broad news coverage. Do not fill
the missing GDELT window with inferred articles or model-news values. A cloud
host should monitor `failed_sources` and verify its own access to GDELT before
depending on that feed for recent context.

On 2026-10-03, the worker downloaded the official 2024 CMS CSV once to verify
the previously imported normalized panel before trusting its modification
header. The Arkansas drug rows matched, so the source generation and forecast
values were unchanged. Future same-URL changes to that header trigger another
content check; a missing header requires a content check on each daily run.

Source adapters must preserve release and retrieval timestamps. A daily worker
may observe the same monthly or annual value repeatedly; it must not create a
new observation for every day without a new source record.
Worker schedules, source period checks, shortage snapshot days, and coverage
ages use the UTC calendar date, including when the laptop timezone differs.

## Private workspace and security

The React dashboard reads public signal history from the API for the selected
one-to-three-year window ending on the uploaded sales history's last date, or
trains without public signals when the user chooses zero years.
Sales and inventory CSVs are read by a Web Worker in the browser and trained there with
the pinned Pyodide XGBoost runtime. The frontend build vendors Pyodide 0.29.4
and its 12 locked package wheels, verifies each wheel against Pyodide's SHA-256
lockfile, and serves the runtime from the app's origin. Chromium training
succeeded with every outside-host request blocked. The 14-day per-drug model uses a purged
validation split, and public signals are eligible only when their recorded
source timestamp is no later than the feature date. Older catalog rows with
unknown or later retrieval times are excluded. Lagged dated inventory is a
required model feature; the latest balance drives the replenishment calculation.
Selected public features are retained only when their model improves purged
holdout weighted absolute percentage error over the private sales and
inventory model; otherwise the plan uses private features alone. Saved plans
from the earlier browser model version require retraining.
The inventory CSV must cover at least 130 sales days per drug with a same-day
or previous-day snapshot. A single latest snapshot is rejected. Each row needs
a real date, and each drug's latest inventory date must equal its last sales date.
If a selected public-signal window yields no eligible features, the dashboard
states that explicitly while retaining the private-sales forecast and stock
plan.
The 14-day forecast and stock balance therefore share one starting day;
plans calculated under the earlier one-day tolerance require retraining.
Outstanding orders are counted only after their stated expected
arrival date; an order with unknown timing is shown but excluded from stock
coverage. The suggested order cannot be smaller than the largest forecast
shortfall through day 14. Saved plans from earlier inventory rules require
retraining before they can be displayed.
On 2026-10-04 UTC, an isolated Chromium end-to-end run used synthetic daily
sales and inventory files, selected one year of public signals, registered a
temporary account, fetched the catalog and eligible signal history, trained
the model, displayed a replenishment plan, and unlocked the encrypted saved
plan after a page reload. Request inspection found no private CSV content in
API POST bodies. This verifies the local preview path; it does not establish
cross-browser or cloud-container compatibility.

The private plan is encrypted on the device with **AES-256-GCM**, using a
**PBKDF2-SHA-256** key derived from the account password. It is stored in
IndexedDB, and a reload requires password unlock. Raw CSVs are not persisted
or sent to the API. The unlocked plan is bound to the authenticated username;
its first salt is created in one IndexedDB write transaction, and a revision
check rejects a stale tab's save instead of replacing a newer encrypted plan.
Account switches and logout clear the in-memory key, and a browser storage
event locks other open tabs. The browser rechecks the authenticated account
every minute and when the tab becomes visible. It locks after 15 minutes of
inactivity and cancels any active training when locked. A real Chromium check
confirmed that revoking the server session and simulating an idle interval
both returned the dashboard to its locked state. New server password hashes use
**Argon2id** with 19 MiB memory, two passes, and one lane;
the account database and its SQLite journal are owner-readable only (`0600`)
on the local Linux host;
legacy scrypt and bcrypt hashes are verified and upgraded after a successful
login without changing the account password. On 2026-10-05, 31 authentication
and API tests passed, including both legacy hash migrations; a live Chromium
run also passed registration, local training, encrypted reload, idle locking,
and session revocation with Argon2id enabled. A ten-sample laptop benchmark
measured a 97 ms median for hashing plus verification; cloud capacity still
needs measurement on the chosen host. Browser sessions use HttpOnly,
SameSite cookies with a double-submit CSRF check. The browser's `/auth/session`
and `/auth/register` responses contain no bearer token; `/auth/token` creates
a separate bearer session for explicit API clients and sets no browser cookie.
Logging out one session revokes that session only. Login attempts are rate limited, and
registration is limited to five attempts per source IP in 15 minutes. Login
uses both a five-attempt account/IP limit and a 30-attempt IP limit; checks
reserve an attempt atomically before password verification. Browser login and
signup reject cross-site Fetch Metadata and unlisted Origin headers. For a
cloud reverse proxy, set `PUBLIC_ORIGIN` to the exact public HTTPS origin and
use `CORS_ORIGINS=[]` for a same-origin frontend. The browser authentication
origin check then works even if the API sees the proxy's internal HTTP URL.
Other CORS origins may access public signal routes but cannot create browser
sessions; the private dashboard requires the configured same-origin frontend.
Set `TRUSTED_PROXY_IPS` or `TRUSTED_PROXY_HOSTNAMES` only to proxy peers that
overwrite `X-Real-IP`; never trust arbitrary client-supplied forwarding
headers. Compose trusts its `frontend` service name so per-IP signup and login
limits use the actual client address when requests pass through Nginx. If a
cloud load balancer sits in front of Nginx, configure Nginx's real-IP trust
for that balancer's exact address ranges before relying on per-client limits.
The API marks account responses `Cache-Control: no-store`. The local built
frontend preview and Compose Nginx serve a response-header CSP with
`frame-ancestors 'none'`, `X-Frame-Options: DENY`, MIME sniffing protection,
and restricted browser permissions. The Nginx policy is repeated in static
asset locations because their cache headers override inherited `add_header`
directives. Set a unique
`SECRET_KEY`, `PUBLIC_ORIGIN`, `ENVIRONMENT=production`, and `COOKIE_SECURE=true` behind HTTPS
before any hosted deployment.

The old `/api/demand/*` server upload and training routes are disabled by
default. They can be enabled for local migration testing with
`ENABLE_LEGACY_SERVER_TRAINING=true`, but production configuration rejects
that setting. The current SQLite store is suitable
for a single local machine; concurrent cloud replicas require a shared
database and migrations. These items, source refresh coverage, account
recovery, monitoring, and a deployment rehearsal remain open production gates.
