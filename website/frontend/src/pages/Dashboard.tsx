import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { useNavigate } from 'react-router-dom'
import { AlertTriangle, ArrowUpRight, Brain, Check, FileSpreadsheet, Loader2, PackageCheck, Upload } from 'lucide-react'
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { apiFetch, getToken, trainUploadedData } from '../api'

type PlanItem = { drug_name: string; on_hand_units: number; forecast_1d_units: number; forecast_7d_units: number; forecast_14d_units: number; days_until_stockout: number | null; minimum_buy_1d_units: number; minimum_buy_7d_units: number; minimum_buy_14d_units: number; recommended_order_units: number; inventory_status: string }
type Status = { ready: boolean; metrics?: { drugs_trained: number; history_start: string; history_end: string; signal_candidate_count: number; signal_feature_count: number; signal_selection_rule: string; planning_summary: { reorder_now: number; total_recommended_order_units: number } } }
type Signal = { signal_id: string; kind: 'demand' | 'news'; score: number; drugs: string[]; source: string; article?: { title: string; url: string; source: string; published_at: string; match_note: string } | null; demand_context?: { scope: string; method: string; points: { day: number; predicted_units: number; cumulative_units: number }[] } | null }

const stages = ['Checking your files', 'Finding signals for each medication', 'Training demand models', 'Building your replenishment plan']

export default function Dashboard() {
  const navigate = useNavigate()
  const [status, setStatus] = useState<Status | null>(null)
  const [items, setItems] = useState<PlanItem[]>([])
  const [signals, setSignals] = useState<Signal[]>([])
  const [salesFile, setSalesFile] = useState<File | null>(null)
  const [inventoryFile, setInventoryFile] = useState<File | null>(null)
  const [busy, setBusy] = useState(false)
  const [stage, setStage] = useState(0)
  const [error, setError] = useState('')

  async function load() {
    const nextStatus = await apiFetch('/demand/status')
    setStatus(nextStatus)
    if (nextStatus.ready) {
      const [plan, relevant] = await Promise.all([apiFetch('/demand/planning'), apiFetch('/demand/signals/relevant')])
      setItems(plan.items)
      setSignals(relevant.signals)
    }
  }

  useEffect(() => {
    if (!getToken()) { navigate('/'); return }
    load().catch(err => setError(err.message))
  }, [navigate])

  useEffect(() => {
    if (!busy) return
    const timer = window.setInterval(() => setStage(current => Math.min(current + 1, stages.length - 1)), 8000)
    return () => window.clearInterval(timer)
  }, [busy])

  async function train(event: FormEvent) {
    event.preventDefault()
    if (!salesFile || !inventoryFile) { setError('Choose both a sales history and an inventory history CSV.'); return }
    setBusy(true); setError(''); setStage(0)
    try {
      await trainUploadedData(salesFile, inventoryFile)
      await load()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed')
    } finally { setBusy(false) }
  }

  if (error && !status) return <Notice text={error} />
  if (!status) return <Loading />
  if (busy) return <Training stage={stage} />
  if (!status.ready) return <UploadPanel salesFile={salesFile} inventoryFile={inventoryFile} onSales={setSalesFile} onInventory={setInventoryFile} onSubmit={train} error={error} />

  const urgent = items.filter(item => item.recommended_order_units > 0)

  return <div className="space-y-6">
    <div className="flex flex-wrap items-end justify-between gap-3"><div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Your workspace</p><h1 className="mt-2 font-serif text-5xl font-normal text-ink">Demand & inventory</h1><p className="mt-2 text-sm text-slate-500">History {status.metrics?.history_start} to {status.metrics?.history_end}</p></div><button onClick={() => { setStatus({ ready: false }); setError('') }} className="rounded-full border border-slate-200 bg-white px-5 py-2.5 text-sm font-medium text-ink shadow-sm transition hover:bg-slate-100">Train with new files</button></div>
    <div className="grid gap-4 sm:grid-cols-3"><Metric icon={<PackageCheck />} label="Medications" value={items.length} /><Metric icon={<AlertTriangle />} label="Need attention" value={urgent.length} /><Metric icon={<ArrowUpRight />} label="Suggested order units" value={items.reduce((sum, item) => sum + item.recommended_order_units, 0).toLocaleString()} /></div>
    <section className="rounded-2xl border border-primary-100 bg-primary-50/40 p-6 shadow-sm sm:p-8"><div className="mb-6 flex items-start gap-3"><span className="rounded-xl bg-primary-100 p-2 text-primary-700"><Brain className="h-6 w-6" /></span><div><h2 className="text-xl font-semibold text-slate-900">Relevant news & signals</h2><p className="mt-1 text-sm text-slate-600">Foundational signals most associated with medications projected to exceed available stock.</p></div></div>{signals.length ? <div className="grid gap-4 md:grid-cols-2">{signals.slice(0, 4).map(signal => <SignalCard key={signal.signal_id} signal={signal} />)}</div> : <p className="rounded-xl bg-white p-6 text-sm text-slate-600">No foundational signal ranked highly for the currently oversold drugs.</p>}</section>
    <section className="overflow-hidden rounded-xl border border-slate-200 bg-white"><div className="border-b border-slate-200 p-5"><h2 className="font-semibold text-slate-900">Restock priorities</h2><p className="mt-1 text-sm text-slate-500">Minimum units to cover demand in each horizon; target order includes the inventory policy's safety stock.</p></div><div className="overflow-x-auto"><table className="w-full text-sm"><thead className="bg-slate-50 text-slate-600"><tr><th className="px-4 py-3 text-left">Drug</th><th className="px-4 py-3 text-right">On hand</th><th className="px-4 py-3 text-right">Demand 1d / 7d / 14d</th><th className="px-4 py-3 text-right">Stockout</th><th className="px-4 py-3 text-right">Buy 1d / 7d / 14d</th><th className="px-4 py-3 text-right">Target order</th></tr></thead><tbody className="divide-y divide-slate-100">{urgent.map(item => <tr key={item.drug_name}><td className="px-4 py-3 font-medium text-slate-900">{item.drug_name}</td><td className="px-4 py-3 text-right">{item.on_hand_units}</td><td className="px-4 py-3 text-right">{Math.ceil(item.forecast_1d_units)} / {Math.ceil(item.forecast_7d_units)} / {Math.ceil(item.forecast_14d_units)}</td><td className="px-4 py-3 text-right">{item.days_until_stockout ? `Day ${item.days_until_stockout}` : 'Beyond 14 days'}</td><td className="px-4 py-3 text-right font-medium text-risk-high">{item.minimum_buy_1d_units} / {item.minimum_buy_7d_units} / {item.minimum_buy_14d_units}</td><td className="px-4 py-3 text-right font-semibold">{item.recommended_order_units}</td></tr>)}</tbody></table>{!urgent.length && <p className="p-8 text-center text-sm text-slate-500">No medications currently need a purchase based on this forecast.</p>}</div></section>
    <p className="text-xs text-slate-500">The model predicts the next 14-day total directly; 1-day and 7-day figures are allocations of that total based on recent day-of-week sales. It selected an average of {status.metrics?.signal_feature_count} signals per drug from {status.metrics?.signal_candidate_count?.toLocaleString()} candidates using training history only. Forecast guidance should be checked against pharmacy practice.</p>
  </div>
}

function UploadPanel({ salesFile, inventoryFile, onSales, onInventory, onSubmit, error }: { salesFile: File | null; inventoryFile: File | null; onSales: (file: File | null) => void; onInventory: (file: File | null) => void; onSubmit: (event: FormEvent) => void; error: string }) {
  return <div className="mx-auto max-w-4xl space-y-8"><div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Start with your data</p><h1 className="mt-3 font-serif text-5xl font-normal text-ink">Build your demand plan.</h1><p className="mt-4 max-w-2xl text-slate-600">Upload dated sales and inventory records. Files and the trained model are stored privately with your account.</p></div><form onSubmit={onSubmit} className="grid gap-5 md:grid-cols-2"><FileCard title="Past sales" description="CSV columns: date, drug_name, units_sold. Include at least 130 consecutive days per drug, with 0 for no-sales days." file={salesFile} onChange={onSales} /><FileCard title="Inventory history" description="CSV columns: date, drug_name, on_hand_units. Optional: on_order_units. The latest balance for each drug is used." file={inventoryFile} onChange={onInventory} /><div className="md:col-span-2">{error && <p role="alert" className="mb-3 text-sm text-risk-high">{error}</p>}<button className="dark-pill flex w-full items-center justify-center gap-2 px-5 py-3.5 text-sm font-medium"><Brain className="h-4 w-4" /> Train my demand model</button><p className="mt-3 text-xs text-slate-500">The model checks 1,312 dated signals, selects up to five per drug from your training period, then predicts the next 14-day total.</p></div></form><div className="rounded-xl border border-slate-200 bg-white p-6"><h2 className="font-semibold text-slate-900">Expected file columns</h2><div className="mt-3 grid gap-4 sm:grid-cols-2 text-sm text-slate-600"><p><span className="font-medium text-ink">Sales:</span> date, drug_name, units_sold</p><p><span className="font-medium text-ink">Inventory:</span> date, drug_name, on_hand_units, on_order_units (optional)</p></div></div></div>
}

function FileCard({ title, description, file, onChange }: { title: string; description: string; file: File | null; onChange: (file: File | null) => void }) {
  return <label className="cursor-pointer rounded-xl border border-dashed border-slate-300 bg-white p-6 hover:border-primary-500"><span className="flex items-center gap-2 font-semibold text-slate-900"><FileSpreadsheet className="h-5 w-5 text-primary-600" />{title}</span><span className="mt-2 block text-sm text-slate-500">{description}</span><span className="mt-4 flex items-center gap-2 text-sm font-medium text-primary-700"><Upload className="h-4 w-4" />{file?.name ?? 'Choose CSV file'}</span><input type="file" accept=".csv,text/csv" className="sr-only" onChange={event => onChange(event.target.files?.[0] ?? null)} /></label>
}

function Training({ stage }: { stage: number }) {
  return <div className="mx-auto max-w-xl rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm"><div className="mx-auto flex h-14 w-14 items-center justify-center rounded-full bg-primary-50"><Loader2 className="h-7 w-7 animate-spin text-primary-600" /></div><h1 className="mt-5 font-serif text-4xl text-ink">Training your workspace model.</h1><p className="mt-2 text-sm text-slate-500">The backend is preparing per-drug features and forecasting demand. This can take a little while.</p><ol className="mt-7 space-y-3 text-left">{stages.map((label, index) => <li key={label} className={`flex items-center gap-3 rounded-lg px-3 py-2 text-sm ${index === stage ? 'bg-primary-50 font-medium text-primary-800' : 'text-slate-500'}`}>{index < stage ? <Check className="h-4 w-4 text-risk-low" /> : index === stage ? <Loader2 className="h-4 w-4 animate-spin" /> : <span className="h-4 w-4 rounded-full border border-slate-300" />}{label}</li>)}</ol></div>
}

function SignalCard({ signal }: { signal: Signal }) {
  return <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><div className="flex items-center justify-between gap-3"><p className="text-base font-semibold text-slate-900">{signal.signal_id.replace(/_/g, ' ')}</p><span className="shrink-0 text-xs font-medium text-primary-700">{(signal.score * 100).toFixed(0)}% correlation</span></div><p className="mt-2 text-sm text-slate-500">Relevant to {signal.drugs.slice(0, 2).join(', ')}{signal.drugs.length > 2 ? ` +${signal.drugs.length - 2}` : ''}</p>{signal.kind === 'demand' && signal.demand_context && <div className="mt-4"><div className="h-32"><ResponsiveContainer width="100%" height="100%"><LineChart data={signal.demand_context.points}><XAxis dataKey="day" hide /><YAxis hide /><Tooltip labelFormatter={day => `Day ${day}`} /><Line type="monotone" dataKey="cumulative_units" name="Cumulative units" stroke="#3E6548" dot={false} strokeWidth={2} /></LineChart></ResponsiveContainer></div><p className="text-xs text-slate-500">14-day demand across all your medications</p></div>}{signal.kind === 'news' && signal.article && <p className="mt-4 text-base font-medium leading-snug text-slate-800">{signal.article.title}</p>}</div>
}

function Metric({ icon, label, value }: { icon: ReactNode; label: string; value: string | number }) {
  return <div className="flex items-center gap-3 rounded-xl border border-slate-200 bg-white p-5"><span className="text-primary-600">{icon}</span><div><p className="text-sm text-slate-500">{label}</p><p className="mt-1 text-2xl font-bold text-slate-900">{value}</p></div></div>
}

function Loading() { return <div className="flex justify-center py-20"><Loader2 className="h-8 w-8 animate-spin text-primary-600" /></div> }
function Notice({ text }: { text: string }) { return <p role="alert" className="rounded-lg bg-red-50 p-5 text-risk-high">{text}</p> }
