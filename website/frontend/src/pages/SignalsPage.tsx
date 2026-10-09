import { useEffect, useState } from 'react'
import { AlertCircle, ArrowUpRight, Search } from 'lucide-react'
import { apiFetch } from '../api'
import SdudProductTable from './SdudProductTable'

type Drug = { id: string | null; drug_name: string; demand_state: number; observed_claims: number; observation_date: string; forecast_horizon: string; state_definition: string }
type News = { title: string; url: string; source: string; published_at: string; timestamp_kind: 'gdelt_first_seen' | 'rss_pub_date'; relevance_score: number }
type NewsSourceCheck = { last_status: string; last_attempt_at: string | null; last_success_at: string | null; last_error_code?: string | null; last_error_http_status?: number | null }
type Shortage = { generic_name: string; status: string; change_date: string; source_last_updated: string; retrieved_at: string }
type PublicSignal = { id: string; signal_id: string; unit: string | null; source_name: string | null; value: number; observation_date: string; source_timestamp: string; ingested_at: string; data_quality: string | null; source_url: string | null }
type RecentItem =
  | { type: 'news'; key: string; timestamp: string; article: News }
  | { type: 'shortage'; key: string; timestamp: string; shortage: Shortage }
  | { type: 'signal'; key: string; timestamp: string; signal: PublicSignal }
type Market = { drugs: Drug[]; count: number; total: number; filtered_total: number; meaning: string; status?: string; source_year?: number; target_year?: number; state_thresholds_claims?: number[] | null; method?: string; evaluation?: { fold_count: number; mean_balanced_accuracy: number } }
type Freshness = { definitions: number; latest_observation_date: string | null; latest_model_target_date?: string | null; checked_at: string; worker_recent: boolean; worker_last_check_at: string | null; failed_sources?: string[]; news_model?: { status: string; publishable: boolean; reason: string; signal_count: number; latest_recorded_observation_date: string | null; latest_target_period: string | null; latest_run: { source_name?: string; started_at: string; finished_at: string | null; status: string; rows_written: number | null } | null } }
type SeedCoverage = { as_of_date: string; seeded_id_count: number; seeded_ids_with_usable_latest_value: number; seeded_ids_with_recent_period_value: number; seeded_ids_with_recent_source_observation: number; seeded_ids_with_current_target_projection: number; seeded_ids_with_older_usable_value: number; seeded_ids_without_usable_latest_value: number; missing_by_reason: Record<string, number> }
type Sdud = { period: string | null; source_rows?: number; suppressed_rows?: number; reported_prescriptions_lower_bound?: number; source_url?: string; last_checked_at?: string }

const stateNames = ['Lowest', 'Low', 'Middle', 'High', 'Highest']
const missingReasonLabels: Record<string, string> = {
  news_model_output_no_validated_live_pipeline: 'article-model outputs awaiting validation',
  atc_model_output_no_validated_live_pipeline: 'ATC outputs awaiting a compatible source and validated rerun',
  legacy_zero_filled_feature_vector: 'ATC outputs backed only by the legacy zero-filled model',
  identity_dimensions_collapsed: 'legacy IDs with collapsed units or dimensions',
  conflicting_latest_revision: 'IDs with conflicting latest revisions',
  no_recorded_observation: 'IDs with no recorded observation',
  unusable_latest_value: 'IDs with an unusable latest value',
}
const DATA_REFRESH_MS = 5 * 60_000
const SEARCH_DELAY_MS = 300

export default function SignalsPage() {
  const [search, setSearch] = useState('')
  const [committedSearch, setCommittedSearch] = useState('')
  const [page, setPage] = useState(0)
  const [market, setMarket] = useState<Market | null>(null)
  const [news, setNews] = useState<News[]>([])
  const [newsSources, setNewsSources] = useState<Record<string, NewsSourceCheck> | null>(null)
  const [shortages, setShortages] = useState<Shortage[]>([])
  const [publicSignals, setPublicSignals] = useState<PublicSignal[]>([])
  const [freshness, setFreshness] = useState<Freshness | null>(null)
  const [seedCoverage, setSeedCoverage] = useState<SeedCoverage | null>(null)
  const [sdud, setSdud] = useState<Sdud | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setPage(0)
      setCommittedSearch(search.trim())
    }, SEARCH_DELAY_MS)
    return () => window.clearTimeout(timer)
  }, [search])

  useEffect(() => {
    let active = true
    let requestNumber = 0
    const refresh = async () => {
      const currentRequest = ++requestNumber
      try {
        const cat = await apiFetch(`/v1/signals/catalog?search=${encodeURIComponent(committedSearch)}&limit=1500`)
        if (!active || currentRequest !== requestNumber) return
        
        const demandSignals = cat.signals.filter((s: any) => s.signal_origin === 'cms_partd_two_year_persistence_v2')
        if (demandSignals.length === 0) {
            setMarket({ drugs: [], count: 0, total: 0, filtered_total: 0, meaning: 'No drugs match this search.' })
            setError('')
            return
        }
        
        const ids = demandSignals.map((s: any) => s.id)
        
        const batches = []
        for (let i = 0; i < ids.length; i += 1500) {
            batches.push(ids.slice(i, i + 1500))
        }
        
        const latestValues = []
        for (const batch of batches) {
            const res = await apiFetch('/v1/signals/latest', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ids: batch })
            })
            latestValues.push(...res.values)
        }
        
        if (!active || currentRequest !== requestNumber) return
        
        const latestMap = new Map(latestValues.map((v: any) => [v.id, v]))
        
        let allDrugs = demandSignals.map((s: any) => {
            const val = latestMap.get(s.id)
            return {
                id: s.id,
                drug_name: s.entity_key,
                demand_state: val ? val.value : 0,
                observed_claims: 0,
                observation_date: val ? val.observation_date : '',
                forecast_horizon: val ? String(parseInt(val.observation_date.slice(0, 4)) + 2) : '',
                state_definition: s.state_definition
            }
        })
        
        allDrugs.sort((a: any, b: any) => b.demand_state - a.demand_state || a.drug_name.localeCompare(b.drug_name))
        
        const filteredTotal = allDrugs.length
        const pagedDrugs = allDrugs.slice(page * 50, (page + 1) * 50)
        
        setMarket({
            drugs: pagedDrugs,
            count: pagedDrugs.length,
            total: cat.count,
            filtered_total: filteredTotal,
            meaning: ''
        })
        setError('')
      } catch (err) {
        if (active && currentRequest === requestNumber) {
          setMarket(null)
          setError(err instanceof Error ? err.message : 'Could not load drug rankings')
        }
      }
    }
    refresh()
    const timer = window.setInterval(refresh, DATA_REFRESH_MS)
    const onVisible = () => { if (!document.hidden) refresh() }
    document.addEventListener('visibilitychange', onVisible)
    return () => { active = false; window.clearInterval(timer); document.removeEventListener('visibilitychange', onVisible) }
  }, [committedSearch, page])

  useEffect(() => {
    let active = true
    let requestNumber = 0
    const refresh = () => {
      const currentRequest = ++requestNumber
      Promise.allSettled([
        apiFetch('/v1/signals/news/recent?days=3'),
        apiFetch('/v1/signals/sources/shortages/recent?days=3'),
        apiFetch('/v1/signals/freshness'),
        apiFetch('/v1/signals/recent?days=3&limit=12'),
        apiFetch('/v1/signals/sources/sdud/latest?limit=1'),
        apiFetch('/v1/signals/coverage/summary'),
      ]).then(([recent, shortageChanges, coverage, observations, sdudSnapshot, seedSummary]) => {
        if (!active || currentRequest !== requestNumber) return
        setNews(recent.status === 'fulfilled' ? recent.value.articles : [])
        setNewsSources(recent.status === 'fulfilled' ? recent.value.source_checks : null)
        setShortages(shortageChanges.status === 'fulfilled' ? shortageChanges.value.changes : [])
        setFreshness(coverage.status === 'fulfilled' ? coverage.value : null)
        setPublicSignals(observations.status === 'fulfilled' ? observations.value.signals : [])
        setSdud(sdudSnapshot.status === 'fulfilled' ? sdudSnapshot.value : null)
        setSeedCoverage(seedSummary.status === 'fulfilled' ? seedSummary.value : null)
      })
    }
    refresh()
    const timer = window.setInterval(refresh, DATA_REFRESH_MS)
    const onVisible = () => { if (!document.hidden) refresh() }
    document.addEventListener('visibilitychange', onVisible)
    const freshnessTimer = window.setInterval(() => {
      apiFetch('/v1/signals/freshness')
        .then(value => { if (active) setFreshness(value) })
        .catch(() => { if (active) setFreshness(null) })
    }, 60_000)
    return () => { active = false; window.clearInterval(timer); window.clearInterval(freshnessTimer); document.removeEventListener('visibilitychange', onVisible) }
  }, [])

  const targetYear = market?.target_year
  const stale = targetYear && targetYear < new Date().getFullYear()
  const newsSourceLabels: Record<string, string> = {
    gdelt_recent_news: 'general news (GDELT)',
    fda_drugs_rss: 'FDA Drugs RSS',
    fda_medwatch_rss: 'FDA MedWatch RSS',
    fda_recalls_rss: 'FDA Recalls RSS',
    fda_press_rss: 'FDA Press RSS',
  }
  const failedNewsSources = Object.entries(newsSources ?? {})
    .filter(([, check]) => check.last_status === 'failed')
    .map(([name, check]) => {
      const label = newsSourceLabels[name] ?? name
      const error = check.last_error_http_status
        ? `HTTP ${check.last_error_http_status}`
        : check.last_error_code ?? 'refresh failed'
      const lastSuccess = check.last_success_at
        ? `; last success ${new Date(check.last_success_at).toLocaleString()}`
        : '; no successful run recorded'
      return `${label}: ${error}${lastSuccess}`
    })
  const missingCoverage = Object.entries(seedCoverage?.missing_by_reason ?? {})
    .filter(([, count]) => count > 0)
    .map(([reason, count]) => `${count} ${missingReasonLabels[reason] ?? reason.replace(/_/g, ' ')}`)
  const recentItems: RecentItem[] = [
    ...news.slice(0, 8).map(article => ({
      type: 'news' as const, key: `news:${article.url}`, timestamp: article.published_at,
      article,
    })),
    ...shortages.slice(0, 4).map(item => ({
      type: 'shortage' as const, key: `shortage:${item.generic_name}:${item.change_date}`,
      timestamp: item.change_date || item.source_last_updated || item.retrieved_at,
      shortage: item,
    })),
    ...publicSignals.map(signal => ({
      type: 'signal' as const, key: `signal:${signal.id}`, timestamp: signal.source_timestamp,
      signal,
    })),
  ].filter(item => Number.isFinite(Date.parse(item.timestamp)))
    .sort((left, right) => Date.parse(right.timestamp) - Date.parse(left.timestamp) ||
      (right.type === 'news' ? right.article.relevance_score : 0) -
      (left.type === 'news' ? left.article.relevance_score : 0))
    .slice(0, 12)

  return <div className="space-y-8">
    <div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Public signal library</p><h1 className="mt-3 font-serif text-5xl text-ink">Drug demand signals</h1><p className="mt-4 max-w-3xl text-sm leading-relaxed text-slate-600">Browse the recorded Arkansas CMS Part D demand states and recent pharmacy news. These are public data proxies. A higher state indicates higher relative demand in this model, not a number of prescriptions or units to buy.</p></div>
    {error && <p role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-700">{error}</p>}
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-5"><Metric label="Catalog signals" value={freshness?.definitions ?? '—'} /><Metric label="Seed IDs with usable records" value={seedCoverage ? `${seedCoverage.seeded_ids_with_usable_latest_value} / ${seedCoverage.seeded_id_count}` : '—'} /><Metric label="Recent source periods" value={seedCoverage ? `${seedCoverage.seeded_ids_with_recent_source_observation} / ${seedCoverage.seeded_id_count}` : '—'} /><Metric label="Current-target drug projections" value={seedCoverage?.seeded_ids_with_current_target_projection ?? '—'} /><Metric label="Latest catalog period" value={freshness?.latest_observation_date ?? '—'} /></div>
    {freshness && !freshness.worker_recent && <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><AlertCircle className="mt-0.5 h-5 w-5 shrink-0" /><p>The public-source refresh worker has not checked in within two hours. Recorded values remain available, but new source updates may be delayed. Last check: {freshness.worker_last_check_at ? new Date(freshness.worker_last_check_at).toLocaleString() : 'none recorded'}.</p></div>}
    {freshness?.failed_sources?.length ? <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><AlertCircle className="mt-0.5 h-5 w-5 shrink-0" /><p>Some source adapters failed on their latest attempt: <span className="font-medium">{freshness.failed_sources.join(', ')}</span>. Other verified sources remain available; inspect freshness and source timestamps before interpreting recent context.</p></div> : null}
    {seedCoverage?.seeded_ids_without_usable_latest_value || seedCoverage?.seeded_ids_with_older_usable_value ? <div role="status" className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><AlertCircle className="mt-0.5 h-5 w-5 shrink-0" /><p><span className="font-medium">{seedCoverage?.seeded_ids_with_recent_source_observation} of {seedCoverage?.seeded_id_count} seeded IDs have recent source observation periods.</span> {seedCoverage?.seeded_ids_with_current_target_projection} drug projections target the current year or later, but may use older source data. {seedCoverage?.seeded_ids_with_older_usable_value} usable values have older periods or targets. {seedCoverage?.seeded_ids_without_usable_latest_value ? `${seedCoverage.seeded_ids_without_usable_latest_value} have no usable value: ${missingCoverage.join('; ')}.` : ''} Coverage as of {seedCoverage?.as_of_date}.</p></div> : null}
    {freshness?.news_model && !freshness.news_model.publishable && freshness.news_model.signal_count > 0 && <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><AlertCircle className="mt-0.5 h-5 w-5 shrink-0" /><p>The 20-signal article-text model is {freshness.news_model.status.replace(/_/g, ' ')}. It runs as an unvalidated shadow estimate from captured article titles and summaries; if the previous month has no text, it estimates the current month-end from month-to-date articles. Its training set has only ten matched months. Its latest recorded target period is {freshness.news_model.latest_target_period ?? 'none'}; the latest historical output period is {freshness.news_model.latest_recorded_observation_date ?? 'unknown'}. These model estimates are available in history but excluded from usable latest values and purchasing.</p></div>}
    {stale && <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><AlertCircle className="mt-0.5 h-5 w-5 shrink-0" /><p>The drug rankings below target {targetYear}. They are historical model outputs and are not current {new Date().getFullYear()} demand estimates. The daily refresh must obtain new source data and run a validated model before current rankings can be published.</p></div>}
    {market?.method && <div className="flex gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900"><AlertCircle className="mt-0.5 h-5 w-5 shrink-0" /><p>These {market.target_year} rankings are two-year projections from {market.source_year} Arkansas Medicare Part D claims. The evaluated persistence baseline achieved {(100 * (market.evaluation?.mean_balanced_accuracy ?? 0)).toFixed(1)}% balanced accuracy across {market.evaluation?.fold_count} chronological holdouts. They are relative claim categories, not current dispensing, inventory, or units to purchase. News above is context and was not used to change these projections.{market.state_thresholds_claims?.length === 4 && <span className="mt-1 block">The five states use historical annual claim-count cutoffs of {market.state_thresholds_claims.map(value => new Intl.NumberFormat().format(value)).join(', ')}. Crossing a cutoff moves a drug into the next higher relative state.</span>}</p></div>}
    {sdud?.period && <div className="rounded-xl border border-slate-200 bg-white p-4 text-sm text-slate-700"><p className="font-medium text-ink">Latest Arkansas Medicaid prescription source: {sdud.period}</p><p className="mt-1">{sdud.suppressed_rows?.toLocaleString()} of {sdud.source_rows?.toLocaleString()} NDC rows have suppressed counts. The {sdud.reported_prescriptions_lower_bound?.toLocaleString()} reported prescriptions are a lower bound, not a complete current demand estimate. This quarterly statewide source has not been used to replace the monthly ATC model outputs.</p>{sdud.source_url && <a className="mt-2 inline-flex items-center gap-1 text-primary-800 underline" href={sdud.source_url} target="_blank" rel="noopener noreferrer">View official source <ArrowUpRight className="h-3 w-3" /></a>}</div>}
    <section className="paper-card rounded-2xl p-6"><div className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-xs font-semibold uppercase tracking-[0.15em] text-slate-500">Last three days</p><h2 className="mt-2 font-serif text-3xl text-ink">Recent news & signals</h2></div><span className="text-xs text-slate-500">Source context, not model attribution</span></div>{failedNewsSources.length > 0 && <p className="mt-4 rounded-lg bg-amber-50 p-3 text-sm text-amber-900">Latest news-feed checks: {failedNewsSources.join("; ")}. Recent items may cover only the sources that succeeded.</p>}{recentItems.length ? <div className="mt-5 grid gap-3 md:grid-cols-2">{recentItems.map(item => <RecentContextCard key={item.key} item={item} />)}</div> : <p className="mt-5 rounded-xl bg-slate-50 p-5 text-sm text-slate-600">No verified pharmacy-related update has been recorded in the last three days.</p>}</section>
    {sdud?.period && <SdudProductTable period={sdud.period} />}
    <section className="paper-card overflow-hidden rounded-2xl"><div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-200 p-6"><div><h2 className="font-serif text-3xl text-ink">All drugs</h2><p className="mt-1 text-sm text-slate-500">Highest relative demand state first · {market?.filtered_total ?? 0} drugs</p></div><label className="flex w-full max-w-xs items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm"><Search className="h-4 w-4 text-slate-400" /><input value={search} maxLength={120} onChange={event => setSearch(event.target.value)} placeholder="Search a drug" className="w-full bg-transparent outline-none" aria-label="Search drugs" /></label></div><div className="overflow-x-auto"><table className="w-full text-sm"><thead className="bg-slate-50 text-slate-600"><tr><th className="px-6 py-3 text-left">Drug</th><th className="px-6 py-3 text-left">Demand state</th><th className="px-6 py-3 text-right">Observed Part D claims ({market?.source_year ?? 'source year'})</th><th className="px-6 py-3 text-left">Target year</th><th className="px-6 py-3 text-left">Signal ID</th></tr></thead><tbody className="divide-y divide-slate-100">{market?.drugs.map(drug => <tr key={drug.drug_name}><td className="px-6 py-3 font-medium text-ink">{drug.drug_name}</td><td className="px-6 py-3"><span className="rounded-full bg-primary-50 px-3 py-1 text-xs font-medium text-primary-800">{stateNames[drug.demand_state] ?? `State ${drug.demand_state}`} ({drug.demand_state}/4)</span></td><td className="px-6 py-3 text-right tabular-nums text-slate-600">{new Intl.NumberFormat().format(drug.observed_claims)}</td><td className="px-6 py-3 text-slate-600">{drug.forecast_horizon || drug.observation_date.slice(0, 4)}</td><td className="px-6 py-3 font-mono text-xs text-slate-500">{drug.id ?? 'ID pending validated refresh'}</td></tr>)}</tbody></table>{market && !market.count && <p className="p-8 text-center text-sm text-slate-500">{market?.status === 'evaluated_baseline_not_available' ? 'No evaluated drug ranking is available yet. The source refresh worker must complete a validated run.' : 'No drugs match this search.'}</p>}</div><div className="flex items-center justify-between border-t border-slate-200 px-6 py-4 text-sm"><span className="text-slate-500">Page {page + 1} of {Math.max(1, Math.ceil((market?.filtered_total ?? 0) / 50))}</span><div className="flex gap-2"><button disabled={page === 0} onClick={() => setPage(page - 1)} className="rounded-full border border-slate-200 px-4 py-2 disabled:opacity-40">Previous</button><button disabled={!market || (page + 1) * 50 >= market.filtered_total} onClick={() => setPage(page + 1)} className="rounded-full border border-slate-200 px-4 py-2 disabled:opacity-40">Next</button></div></div></section>
    <p className="text-xs text-slate-500">Signal values carry source and date metadata through the API. News relevance is a keyword match. Neither is direct pharmacy inventory evidence.</p>
  </div>
}

function Metric({ label, value }: { label: string; value: string | number }) { return <div className="paper-card rounded-xl p-5"><p className="text-xs uppercase tracking-wider text-slate-500">{label}</p><p className="mt-2 text-2xl font-semibold text-ink">{value}</p></div> }

function RecentContextCard({ item }: { item: RecentItem }) {
  if (item.type === 'signal') return <PublicSignalCard signal={item.signal} />
  if (item.type === 'news') return <a href={item.article.url} target="_blank" rel="noopener noreferrer" className="rounded-xl border border-slate-200 p-4 transition hover:bg-slate-50">
    <span className="text-xs text-slate-500">{item.article.source} · {item.article.timestamp_kind === 'gdelt_first_seen' ? 'first seen by GDELT' : 'FDA feed date'} {new Date(item.article.published_at).toLocaleDateString()} · {item.article.relevance_score} keyword matches</span>
    <h3 className="mt-2 font-medium text-ink">{item.article.title}</h3>
    <ArrowUpRight className="mt-3 h-4 w-4 text-slate-500" />
  </a>
  return <div className="rounded-xl border border-slate-200 p-4">
    <span className="text-xs text-slate-500">openFDA shortage update · {item.shortage.change_date || item.shortage.source_last_updated}</span>
    <h3 className="mt-2 font-medium text-ink">{item.shortage.generic_name}</h3>
    <p className="mt-2 text-xs text-slate-600">Reported status: {item.shortage.status}</p>
  </div>
}

function PublicSignalCard({ signal }: { signal: PublicSignal }) {
  const siteCount = Number(signal.data_quality?.match(/contributing_site_rows=(\d+)/)?.[1])
  const value = signal.unit === 'share'
    ? `${(signal.value * 100).toFixed(1)}%`
    : signal.unit === 'percent_outpatient_visits'
      ? `${signal.value.toFixed(2)}% of outpatient visits`
    : `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(signal.value)} ${signal.unit?.replace(/_/g, ' ') ?? ''}`
  return <div className="rounded-xl border border-slate-200 p-4">
    <p className="text-xs text-slate-500">Public source context · updated {new Date(signal.source_timestamp).toLocaleDateString()} · recorded {new Date(signal.ingested_at).toLocaleDateString()}</p>
    <h3 className="mt-2 font-medium capitalize text-ink">{signal.signal_id.replace(/_/g, ' ')}</h3>
    <p className="mt-2 text-xl font-semibold text-ink">{value}</p>
    <p className="mt-2 text-xs text-slate-600">Source period: {signal.observation_date} · {signal.source_name ?? 'Official public source'}</p>
    {Number.isFinite(siteCount) && <p className="mt-2 text-xs text-amber-800">Mean of {siteCount} reporting {siteCount === 1 ? 'site' : 'sites'}; not CDC’s official state median.</p>}
    {signal.source_url && <a className="mt-2 inline-flex items-center gap-1 text-xs text-primary-800 underline" href={signal.source_url} target="_blank" rel="noopener noreferrer">View source <ArrowUpRight className="h-3 w-3" /></a>}
  </div>
}
