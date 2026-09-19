import { useEffect, useState } from 'react'
import { Brain, CheckCircle, Database, Loader2 } from 'lucide-react'
import { apiFetch, getToken } from '../api'

interface DemoStatus {
  mode: string
  dataset: string
  uploads_enabled: boolean
  has_history: boolean
  has_model: boolean
  has_forecasts: boolean
  metrics?: {
    drugs_trained: number
    history_start: string
    history_end: string
    source_rows: number
    horizon_days: number
    forecast_rows: number
    pooled?: { wape: number }
  }
}

export default function DataUpload() {
  const [status, setStatus] = useState<DemoStatus | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!getToken()) {
      setError('Not logged in. Please log in first.')
      return
    }
    apiFetch('/demand/status').then(setStatus).catch((err) => setError(err.message))
  }, [])

  if (error) return <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center text-slate-700">{error}</div>
  if (!status) return <div className="flex justify-center py-20"><Loader2 className="w-8 h-8 text-primary-600 animate-spin" /></div>

  const metrics = status.metrics
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Synthetic Demo Dataset</h1>
        <p className="text-sm text-slate-500 mt-1">This demonstration is locked to a prepared synthetic pharmacy-sales dataset.</p>
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-primary-200 p-6">
        <div className="flex items-start gap-4">
          <div className="p-3 bg-primary-50 rounded-lg"><Database className="w-7 h-7 text-primary-600" /></div>
          <div className="flex-1">
            <div className="flex items-center gap-2"><h2 className="font-semibold text-slate-900">{status.dataset}</h2><span className="text-xs bg-primary-100 text-primary-700 px-2 py-1 rounded-full">Locked</span></div>
            <p className="text-sm text-slate-600 mt-2">The admin demo uses a fixed daily sales history. Uploading data and retraining are disabled for this version.</p>
          </div>
          <CheckCircle className="w-6 h-6 text-risk-low" />
        </div>
      </div>

      <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Metric label="Medications" value={metrics?.drugs_trained ?? '—'} />
        <Metric label="Sales-history rows" value={metrics?.source_rows?.toLocaleString() ?? '—'} />
        <Metric label="History period" value={metrics ? `${metrics.history_start} – ${metrics.history_end}` : '—'} />
        <Metric label="Forecast horizon" value={metrics ? `${metrics.horizon_days} days` : '—'} />
      </div>

      <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-5">
        <div className="flex items-center gap-2 mb-2"><Brain className="w-5 h-5 text-primary-600" /><h2 className="font-semibold text-slate-900">Prepared model</h2></div>
        <p className="text-sm text-slate-600">A separate chronological XGBoost demand model is already trained for each medication. The current demo contains {metrics?.forecast_rows?.toLocaleString() ?? '—'} forecast rows{metrics?.pooled ? ` and a ${Math.round(metrics.pooled.wape * 100)}% historical holdout WAPE.` : '.'}</p>
      </div>
    </div>
  )
}

function Metric({ label, value }: { label: string; value: string | number }) {
  return <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-4"><p className="text-lg font-bold text-slate-900">{value}</p><p className="text-xs text-slate-500 mt-1">{label}</p></div>
}
