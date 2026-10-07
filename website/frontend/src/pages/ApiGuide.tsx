import { Link } from 'react-router-dom'

const examples = [
  { title: 'List the catalog', code: 'GET /api/v1/signals/catalog?limit=1500' },
  { title: 'Newly recorded public observations', code: 'GET /api/v1/signals/recent?days=3&limit=12' },
  { title: 'Arkansas Medicaid prescriptions (partial observed counts)', code: 'GET /api/v1/signals/sources/sdud/latest?search=insulin&limit=50' },
  { title: 'Latest values for selected IDs', code: 'POST /api/v1/signals/latest\nContent-Type: application/json\n\n{"ids":["sig_...","sig_..."]}' },
  { title: 'One date or a range', code: 'POST /api/v1/signals/history\nContent-Type: application/json\n\n{"ids":["sig_..."],"start_date":"2025-01-01","end_date":"2025-12-31"}' },
  { title: 'Every recorded revision for point-in-time use', code: 'POST /api/v1/signals/history\nContent-Type: application/json\n\n{"ids":["sig_..."],"date":"2025-01-31","include_revisions":true}' },
]

export default function ApiGuide() {
  return <div className="mx-auto max-w-4xl space-y-8">
    <div>
      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Developer guide</p>
      <h1 className="mt-3 font-serif text-5xl text-ink">PULSE signal API</h1>
      <p className="mt-4 text-slate-600">Use the same-origin <code>/api/v1</code> paths locally. A later domain can point to the same routes. Interactive OpenAPI documentation is available at <a className="underline" href="/docs">/docs</a>.</p>
    </div>

    <section className="paper-card rounded-2xl p-6">
      <h2 className="text-xl font-semibold">What a signal means</h2>
      <p className="mt-3 text-sm leading-relaxed text-slate-600">Each stable <code>sig_</code> ID identifies an origin, measure, cadence, geography, and entity. A value can be an observed public statistic, a news-derived score, or a predicted demand state. Read its <code>unit</code>, <code>state_definition</code>, source, observation date, and forecast horizon before interpreting it. A five-state demand value of 4 means the model’s highest relative category; it is not four units, four prescriptions, or an order recommendation.</p>
      <p className="mt-3 text-sm text-slate-600">The checked-in seed manifest has 1,312 identities. Validated CMS refreshes may add stable IDs for newer drug outputs, so the live catalog can be larger; see <code>/freshness</code>. Not every ID has a current usable observation. <code>/latest</code> returns validated values in <code>values</code>. Its <code>latest_recorded_values</code> array gives the absolute latest stored row for every requested ID, including rows flagged <code>usable: false</code>. The <code>unusable_recorded_values</code> array highlights archived rows for IDs that lack a usable value, with their real dates and <code>unusable_reason</code>. Do not treat those numbers as current demand or use them for purchasing. <code>missing_ids</code> means no usable value exists; <code>missing_details</code> gives the gap reason and last recorded period, or <code>unknown_id</code> for an unrecognized ID.</p>
      <p className="mt-3 text-sm text-slate-600">Six legacy IDs collapse unrelated FDA events or NADAC pricing units. The 18 ATC model rows used a zero-filled feature vector, and the 20 news-model columns have no verified live generator. Those historical records remain queryable and explicitly unusable. The <code>/freshness</code> response includes a <code>news_model</code> status with its signal count, latest recorded period, and any live run. A value of <code>not_configured</code> means no live inference run has been recorded; model rows remain excluded from usable latest values until a validated output passes its promotion gate. Conflicting values for one period carry <code>ambiguous: true</code>; callers should not choose one automatically. The evaluated annual Part D baseline is a separate projection. A five-state value is a relative category, not units to buy.</p>
    </section>

    <section className="space-y-4">
      <h2 className="font-serif text-3xl">Requests</h2>
      {examples.map(item => <div className="paper-card rounded-xl p-5" key={item.title}>
        <h3 className="font-semibold">{item.title}</h3>
        <pre className="mt-3 overflow-x-auto rounded-lg bg-slate-900 p-4 text-xs text-white">{item.code}</pre>
      </div>)}
    </section>

    <section className="paper-card rounded-2xl p-6">
      <h2 className="text-xl font-semibold">History size</h2>
      <p className="mt-3 text-sm text-slate-600">History accepts up to 100 IDs. If a request spans more than 10,000 recorded source rows, it returns HTTP 413 without truncating values. Retry with fewer IDs or a shorter date range. The private browser trainer splits oversized ID batches automatically.</p>
      <p className="mt-3 text-sm text-slate-600">Public signal requests have a per-client budget that refills over time. If the API returns HTTP 429, wait the number of seconds in its <code>Retry-After</code> header before retrying. This limit also applies to browser training history downloads.</p>
    </section>

    <section className="paper-card rounded-2xl p-6">
      <h2 className="text-xl font-semibold">Coverage and freshness</h2>
      <p className="mt-3 text-sm text-slate-600"><code>GET /api/v1/signals/freshness</code> returns catalog counts, latest observation periods, <code>failed_sources</code> for adapters whose most recent attempt failed, and the separate <code>news_model</code> inference status. A failed adapter does not invalidate other verified source data; use each row’s status and timestamp. <code>GET /api/v1/signals/coverage</code> lists per-signal gaps. <code>GET /api/v1/signals/demand/drugs</code> ranks stored drug demand states; <code>GET /api/v1/signals/news/recent?days=3</code> and <code>GET /api/v1/signals/sources/shortages/recent?days=3</code> return dated source context.</p>
      <p className="mt-3 text-sm text-slate-600">News rows include <code>timestamp_kind</code>: GDELT dates are first seen by its feed, while FDA dates come from RSS <code>pubDate</code>. The news response also includes <code>source_checks</code> with each feed’s latest status, attempt time, last success time, and a sanitized <code>last_error_code</code> / <code>last_error_http_status</code> when a check fails. Raw upstream exception text is not returned; FDA fallback items remain visible when general news fails.</p>
      <p className="mt-3 text-sm text-slate-600"><code>GET /api/v1/signals/sources/nadac/latest</code> returns current NADAC acquisition-rate groups by classification and pricing unit. Its four legacy catalog IDs combine unlike units and are marked <code>identity_dimensions_collapsed</code> in coverage; use the grouped snapshot for current price context. The drug ranking uses a separately evaluated two-year persistence projection from the latest published Arkansas Medicare Part D claims. Its target year, source year, method, holdout accuracy, and four <code>state_thresholds_claims</code> accompany the rows. Those thresholds divide historical annual claim counts into five relative categories; they are not purchase quantities. Newer CMS drug outputs receive stable IDs during a successful evaluated baseline refresh; earlier periods may have no observations. The API never advances a source’s date or fills a missing value with a forecast.</p>
      <p className="mt-3 text-sm text-slate-600">Catalog, drug-ranking, and SDUD product searches treat <code>%</code> and <code>_</code> as literal text. The public signal routes are read-only. Private sales and inventory files stay in the browser workspace and are not part of this API.</p>
    </section>

    <Link to="/signals" className="dark-pill inline-flex px-6 py-3 text-sm">Browse signals</Link>
  </div>
}
