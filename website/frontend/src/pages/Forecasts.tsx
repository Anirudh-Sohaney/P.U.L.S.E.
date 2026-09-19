import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { AlertTriangle, CalendarDays, Loader2, TrendingUp } from 'lucide-react'
import { apiFetch, getToken } from '../api'

interface Forecast { date: string; drug_name: string; predicted_units: number; horizon_day: number }

export default function Forecasts() {
  const [forecasts, setForecasts] = useState<Forecast[]>([])
  const [metrics, setMetrics] = useState<any>(null)
  const [selectedDrug, setSelectedDrug] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!getToken()) { setError('Not logged in. Please log in first.'); setLoading(false); return }
    apiFetch('/demand/forecasts').then((result) => { setForecasts(result.forecasts); setMetrics(result.metrics); setLoading(false) }).catch((err) => { setError(err.message); setLoading(false) })
  }, [])

  const drugs = useMemo(() => ['all', ...Array.from(new Set(forecasts.map(row => row.drug_name))).sort()], [forecasts])
  const visible = selectedDrug === 'all' ? forecasts : forecasts.filter(row => row.drug_name === selectedDrug)

  if (loading) return <div className="flex justify-center py-20"><Loader2 className="w-8 h-8 text-primary-600 animate-spin" /></div>
  if (error) return <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center"><AlertTriangle className="w-12 h-12 text-risk-medium mx-auto mb-3" /><p className="text-slate-700">{error}</p><a href="/data" className="text-primary-600 underline text-sm">View the locked demo dataset.</a></div>

  return <div className="space-y-6">
    <div className="flex flex-wrap justify-between gap-3"><div><h1 className="text-2xl font-bold text-slate-900">Demand Forecasts</h1><p className="text-sm text-slate-500">Daily predicted dispensing units from the locked synthetic pharmacy history.</p></div><select value={selectedDrug} onChange={(event) => setSelectedDrug(event.target.value)} className="border border-slate-200 rounded-lg px-3 py-2 text-sm">{drugs.map(drug => <option value={drug} key={drug}>{drug === 'all' ? 'All medications' : drug}</option>)}</select></div>
    <div className="grid sm:grid-cols-3 gap-4"><Summary icon={<CalendarDays className="w-5 h-5" />} label="Forecast horizon" value={`${metrics?.horizon_days ?? '—'} days`} /><Summary icon={<TrendingUp className="w-5 h-5" />} label="Holdout WAPE" value={metrics?.pooled ? `${(metrics.pooled.wape * 100).toFixed(1)}%` : '—'} /><Summary icon={<TrendingUp className="w-5 h-5" />} label="Signal features used" value={metrics?.signal_feature_count ?? 0} /></div>
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden"><table className="w-full"><thead className="bg-slate-50 text-left text-sm text-slate-600"><tr><th className="px-4 py-3">Date</th><th className="px-4 py-3">Medication</th><th className="px-4 py-3 text-right">Forecast units</th><th className="px-4 py-3 text-right">Day ahead</th></tr></thead><tbody className="divide-y divide-slate-200">{visible.map((row) => <tr key={`${row.date}-${row.drug_name}`} className="hover:bg-slate-50"><td className="px-4 py-3 text-sm">{row.date}</td><td className="px-4 py-3 font-medium text-sm">{row.drug_name}</td><td className="px-4 py-3 text-right text-sm">{row.predicted_units.toFixed(2)}</td><td className="px-4 py-3 text-right text-sm text-slate-500">{row.horizon_day}</td></tr>)}</tbody></table>{visible.length === 0 && <div className="p-8 text-center text-slate-500">No forecasts match this medication.</div>}</div>
  </div>
}

function Summary({ icon, label, value }: { icon: ReactNode; label: string; value: string | number }) {
  return <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4"><div className="flex gap-2 text-primary-600">{icon}<span className="text-sm text-slate-500">{label}</span></div><p className="mt-2 text-xl font-bold text-slate-900">{value}</p></div>
}
