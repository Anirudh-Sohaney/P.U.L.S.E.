import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { AlertTriangle, ArrowDownRight, ArrowUpRight, Check, Clock3, Database, Package, RefreshCw, Search, Sparkles, Truck } from 'lucide-react'
import { Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { apiFetch } from '../api'

type Horizon = '1' | '3' | '7' | '14'
const horizons: Horizon[] = ['1', '3', '7', '14']

interface Row {
  drug_name: string
  current_inventory: number | null
  reorder_point: number | null
  lead_time_days: number | null
  inventory_source: string | null
  forecast_units: Record<Horizon, number>
  purchase_units: Record<Horizon, number | null>
  status: 'buy' | 'covered' | 'inventory unavailable'
  selected_signal_count: number
}

interface Data {
  generated_at: string
  model: { candidate_signal_features: number; selected_features_per_drug: number; variant: string; training_cutoff: string; history_start: string; history_end: string; drugs: number }
  summary: { forecast_drugs: number; inventory_backed: number; needs_purchase: number; inventory_unavailable: number; purchase_units: Record<Horizon, number> }
  rows: Row[]
}

const fmt = (value: number) => new Intl.NumberFormat('en-US', { maximumFractionDigits: 0 }).format(value)

export default function Dashboard() {
  const [data, setData] = useState<Data | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [error, setError] = useState('')
  const [filter, setFilter] = useState<'all' | 'buy' | 'covered'>('all')
  const [query, setQuery] = useState('')
  const [chartDrug, setChartDrug] = useState('')

  const load = async (refresh = false) => {
    setError('')
    refresh ? setRefreshing(true) : setLoading(true)
    try {
      const response = await apiFetch(refresh ? '/purchasing/dashboard/refresh' : '/purchasing/dashboard', refresh ? { method: 'POST' } : {})
      setData(response)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unable to load the purchasing plan.')
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => { void load() }, [])

  const rows = useMemo(() => {
    if (!data) return []
    return data.rows
      .filter((row) => row.status !== 'inventory unavailable')
      .filter((row) => filter === 'all' || row.status === filter)
      .filter((row) => row.drug_name.toLowerCase().includes(query.toLowerCase()))
      .sort((a, b) => {
        const urgency: Record<Row['status'], number> = { buy: 0, covered: 1, 'inventory unavailable': 2 }
        return urgency[a.status] - urgency[b.status] || (b.purchase_units['14'] ?? 0) - (a.purchase_units['14'] ?? 0) || a.drug_name.localeCompare(b.drug_name)
      })
  }, [data, filter, query])

  if (loading) return <div className="grid min-h-screen place-items-center bg-[#f3f6fa] text-sm text-slate-500">Preparing purchasing intelligence…</div>

  const urgent = data?.rows.filter((row) => row.status === 'buy').sort((a, b) => (b.purchase_units['14'] ?? 0) - (a.purchase_units['14'] ?? 0))[0]
  const chartRow = data?.rows.find((row) => row.drug_name === chartDrug) ?? urgent ?? data?.rows.find((row) => row.status !== 'inventory unavailable')
  const chartData = chartRow ? horizons.map((horizon) => ({ horizon: `${horizon}d`, demand: Math.round(chartRow.forecast_units[horizon]), target: Math.round(chartRow.forecast_units[horizon] + (chartRow.reorder_point ?? 0)), inventory: Math.round(chartRow.current_inventory ?? 0), buy: chartRow.purchase_units[horizon] ?? 0 })) : []

  return <div className="min-h-screen bg-[#f3f6fa] text-slate-950">
    <header className="bg-slate-950 text-white"><div className="mx-auto flex max-w-[1540px] items-center justify-between px-6 py-5 lg:px-10"><div className="flex items-center gap-3"><div className="grid h-10 w-10 place-items-center rounded-2xl bg-cyan-400 text-slate-950"><Package className="h-5 w-5" /></div><div><p className="text-lg font-semibold tracking-tight">PULSE</p><p className="text-xs text-slate-400">Inventory intelligence</p></div></div><button onClick={() => void load(true)} disabled={refreshing} className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2.5 text-sm font-medium text-white transition hover:bg-white/20 disabled:opacity-50"><RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} /> Recalculate plan</button></div></header>

    <main className="mx-auto max-w-[1540px] space-y-6 px-6 py-8 lg:px-10">
      <section className="flex flex-col justify-between gap-6 lg:flex-row lg:items-end"><div><p className="mb-3 text-xs font-bold uppercase tracking-[0.22em] text-cyan-700">Daily procurement brief</p><h1 className="max-w-3xl text-4xl font-semibold tracking-[-0.04em] text-slate-950 sm:text-5xl">Buy with confidence.<br /><span className="text-slate-400">See the need before it arrives.</span></h1><p className="mt-5 max-w-2xl text-base leading-7 text-slate-500">Forecasted dispensing demand is compared with active inventory to produce a practical buy plan for the next 1, 3, 7, and 14 days.</p></div>{data && <div className="flex items-center gap-3 text-sm text-slate-500"><Clock3 className="h-4 w-4" /> Updated {new Date(data.generated_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</div>}</section>

      {error && <div className="flex items-center gap-3 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-700"><AlertTriangle className="h-5 w-5" />{error}</div>}
      {data && <>
        <section className="grid gap-4 lg:grid-cols-[1.4fr_1fr_1fr]">
          <div className="relative overflow-hidden rounded-3xl bg-cyan-400 p-6 text-slate-950 shadow-sm"><div className="relative z-10"><p className="text-sm font-semibold text-cyan-950">Immediate attention</p>{urgent ? <><h2 className="mt-3 text-3xl font-semibold tracking-tight">{urgent.drug_name}</h2><p className="mt-2 max-w-sm text-sm text-cyan-950">Projected demand exceeds current stock over the 14-day planning window.</p><div className="mt-6 flex items-end gap-8"><div><p className="text-xs font-medium uppercase tracking-wide text-cyan-900">Current stock</p><p className="mt-1 text-2xl font-semibold">{fmt(urgent.current_inventory ?? 0)}</p></div><div><p className="text-xs font-medium uppercase tracking-wide text-cyan-900">Buy for 14 days</p><p className="mt-1 text-2xl font-semibold">{fmt(urgent.purchase_units['14'] ?? 0)}</p></div></div></> : <><h2 className="mt-3 text-3xl font-semibold tracking-tight">Stock is covered</h2><p className="mt-2 text-sm text-cyan-950">No matched medication currently has a modeled shortfall.</p></>} </div><Sparkles className="absolute -bottom-8 -right-5 h-44 w-44 text-cyan-300/70" /></div>
          <Kpi icon={<Truck />} label="Buy recommendations" value={fmt(data.summary.needs_purchase)} detail="inventory-backed drugs" />
          <Kpi icon={<Database />} label="Inventory coverage" value={`${fmt(data.summary.inventory_backed)} / ${fmt(data.summary.forecast_drugs)}`} detail={`${fmt(data.summary.inventory_unavailable)} excluded from view`} />
        </section>

        <section className="grid grid-cols-2 gap-3 sm:grid-cols-4"><HorizonCard horizon="1" units={data.summary.purchase_units['1']} /><HorizonCard horizon="3" units={data.summary.purchase_units['3']} /><HorizonCard horizon="7" units={data.summary.purchase_units['7']} /><HorizonCard horizon="14" units={data.summary.purchase_units['14']} emphasized /></section>

        {chartRow && <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"><div className="flex flex-col justify-between gap-4 sm:flex-row sm:items-start"><div><p className="text-xs font-bold uppercase tracking-[0.16em] text-cyan-700">Demand runway</p><h2 className="mt-1 text-xl font-semibold tracking-tight">Forecast demand vs. protected stock</h2><p className="mt-1 text-sm text-slate-500">Required stock includes predicted demand plus the medication’s reorder-point buffer.</p></div><select value={chartRow.drug_name} onChange={(event) => setChartDrug(event.target.value)} className="rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-medium text-slate-700 outline-none focus:border-cyan-500">{data.rows.filter((row) => row.status !== 'inventory unavailable').map((row) => <option key={row.drug_name} value={row.drug_name}>{row.drug_name}</option>)}</select></div><div className="mt-6 h-72 w-full"><ResponsiveContainer width="100%" height="100%"><ComposedChart data={chartData} margin={{ top: 8, right: 12, left: -12, bottom: 0 }}><CartesianGrid stroke="#e2e8f0" strokeDasharray="4 4" vertical={false} /><XAxis dataKey="horizon" axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 12 }} /><YAxis axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 12 }} /><Tooltip cursor={{ fill: '#f8fafc' }} contentStyle={{ borderRadius: 12, border: '1px solid #e2e8f0', boxShadow: '0 8px 24px rgba(15,23,42,.08)' }} formatter={(value: number, name: string) => [fmt(value), name === 'demand' ? 'Forecast demand' : name === 'target' ? 'Required stock' : name === 'inventory' ? 'On hand' : 'Buy gap']} /><Bar dataKey="demand" name="demand" fill="#22d3ee" radius={[6, 6, 0, 0]} barSize={38} /><Line type="monotone" dataKey="target" name="target" stroke="#f97316" strokeWidth={2} strokeDasharray="6 4" dot={false} /><Line type="monotone" dataKey="inventory" name="inventory" stroke="#0f172a" strokeWidth={3} dot={{ r: 4, fill: '#0f172a' }} /></ComposedChart></ResponsiveContainer></div><div className="mt-4 flex flex-wrap gap-5 text-xs text-slate-500"><span className="flex items-center gap-2"><i className="h-2.5 w-2.5 rounded-full bg-cyan-400" />Forecast demand</span><span className="flex items-center gap-2"><i className="h-0.5 w-4 bg-orange-500" />Required stock + reorder buffer</span><span className="flex items-center gap-2"><i className="h-2.5 w-2.5 rounded-full bg-slate-900" />On hand</span></div></section>}

        <section className="overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-[0_12px_40px_rgba(15,23,42,0.06)]"><div className="flex flex-col gap-5 border-b border-slate-200 p-6 xl:flex-row xl:items-center xl:justify-between"><div><div className="flex items-center gap-2"><h2 className="text-xl font-semibold tracking-tight">Procurement queue</h2><span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs font-semibold text-slate-500">{rows.length} shown</span></div><p className="mt-1 text-sm text-slate-500">Urgent rows stay pinned to the top. Only drugs with active inventory are included.</p></div><div className="flex flex-col gap-3 sm:flex-row"><label className="flex items-center gap-2 rounded-xl border border-slate-200 px-3 py-2 text-sm text-slate-400"><Search className="h-4 w-4" /><input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search medication" className="w-full bg-transparent text-slate-800 outline-none placeholder:text-slate-400 sm:w-48" /></label><div className="flex rounded-xl bg-slate-100 p-1 text-sm">{([['all', 'All'], ['buy', 'Buy'], ['covered', 'Covered']] as const).map(([key, label]) => <button key={key} onClick={() => setFilter(key)} className={`rounded-lg px-4 py-2 font-medium ${filter === key ? 'bg-white text-slate-950 shadow-sm' : 'text-slate-500'}`}>{label}</button>)}</div></div></div><div className="overflow-x-auto"><table className="w-full min-w-[1060px] text-left"><thead className="bg-slate-50 text-[11px] font-bold uppercase tracking-[0.12em] text-slate-500"><tr><th className="px-6 py-4">Medication</th><th className="px-4 py-4 text-right">On hand</th>{horizons.map((h) => <th key={h} className="px-4 py-4 text-right">{h} day</th>)}<th className="px-6 py-4">Action</th></tr></thead><tbody className="divide-y divide-slate-100">{rows.map((row, index) => <ProcurementRow key={row.drug_name} row={row} rank={index + 1} />)}</tbody></table>{rows.length === 0 && <div className="px-6 py-16 text-center text-sm text-slate-500">No medications match this view.</div>}</div></section>

        <section className="flex flex-col justify-between gap-4 rounded-2xl border border-slate-200 bg-white px-6 py-5 text-sm shadow-sm md:flex-row md:items-center"><div className="flex items-center gap-3"><div className="grid h-9 w-9 place-items-center rounded-xl bg-slate-100"><Database className="h-4 w-4 text-slate-600" /></div><div><p className="font-semibold">Forecast engine</p><p className="text-slate-500">{fmt(data.model.candidate_signal_features)} candidate signals · {data.model.selected_features_per_drug} selected per drug · {data.model.variant}</p></div></div><p className="text-slate-400">History through {data.model.history_end} · cutoff {data.model.training_cutoff}</p></section>
      </>}
    </main>
  </div>
}

function Kpi({ icon, label, value, detail }: { icon: ReactNode; label: string; value: string; detail: string }) { return <div className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm"><div className="flex items-center justify-between"><div className="grid h-10 w-10 place-items-center rounded-2xl bg-slate-100 text-slate-700">{icon}</div><ArrowUpRight className="h-4 w-4 text-slate-300" /></div><p className="mt-8 text-sm text-slate-500">{label}</p><p className="mt-1 text-3xl font-semibold tracking-tight">{value}</p><p className="mt-1 text-xs text-slate-400">{detail}</p></div> }

function HorizonCard({ horizon, units, emphasized = false }: { horizon: Horizon; units: number; emphasized?: boolean }) { return <div className={`rounded-2xl border p-4 ${emphasized ? 'border-slate-950 bg-slate-950 text-white' : 'border-slate-200 bg-white'}`}><p className={`text-xs font-semibold uppercase tracking-[0.14em] ${emphasized ? 'text-slate-400' : 'text-slate-400'}`}>{horizon}-day buy plan</p><p className="mt-2 text-2xl font-semibold">{fmt(units)} <span className={`text-sm font-normal ${emphasized ? 'text-slate-400' : 'text-slate-400'}`}>units</span></p><div className={`mt-3 h-1 rounded-full ${emphasized ? 'bg-cyan-400' : 'bg-slate-200'}`} /></div> }

function ProcurementRow({ row, rank }: { row: Row; rank: number }) { return <tr className={`${row.status === 'buy' ? 'bg-orange-50/40' : 'bg-white'} hover:bg-slate-50`}><td className="px-6 py-5"><div className="flex items-center gap-3"><span className={`grid h-7 w-7 place-items-center rounded-lg text-xs font-bold ${row.status === 'buy' ? 'bg-orange-200 text-orange-800' : 'bg-slate-100 text-slate-400'}`}>{rank}</span><div><p className="font-semibold text-slate-900">{row.drug_name}</p><p className="mt-1 text-xs text-slate-400">{row.selected_signal_count} selected signals · reorder buffer {fmt(row.reorder_point ?? 0)} · {fmt(row.lead_time_days ?? 0)}d lead time</p></div></div></td><td className="px-4 py-5 text-right"><p className="font-semibold">{fmt(row.current_inventory ?? 0)}</p><p className="mt-1 text-xs text-slate-400">units</p></td>{horizons.map((h) => { const buy = row.purchase_units[h] ?? 0; return <td key={h} className="px-4 py-5 text-right"><p className="text-xs text-slate-400">{fmt(row.forecast_units[h])} demand</p><p className={`mt-1 font-bold ${buy > 0 ? 'text-orange-700' : 'text-emerald-700'}`}>{buy > 0 ? `Buy ${fmt(buy)}` : <span className="inline-flex items-center gap-1"><Check className="h-3 w-3" /> Covered</span>}</p></td> })}<td className="px-6 py-5">{row.status === 'buy' ? <span className="inline-flex items-center gap-1.5 rounded-full bg-orange-100 px-3 py-1.5 text-xs font-bold text-orange-800"><ArrowDownRight className="h-3.5 w-3.5" /> Reorder</span> : <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-1.5 text-xs font-bold text-emerald-800"><Check className="h-3.5 w-3.5" /> Covered</span>}</td></tr> }
