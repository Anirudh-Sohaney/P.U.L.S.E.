import { useState, useEffect, useCallback } from 'react'
import { useLocation } from 'react-router-dom'
import { AlertTriangle, CheckCircle, TrendingUp, Pill, Loader2, RefreshCw } from 'lucide-react'
import { apiFetch, getToken } from '../api'

interface DashboardStats {
  total_medications: number
  high_risk: number
  low_risk: number
  avg_risk_score: number
}

interface RiskMedication {
  medication_id: string
  medication_name: string
  medication_category: string
  risk_score: number
  risk_label: string
  confidence: number
}

interface CategoryBreakdown {
  medication_category: string
  count: number
  avg_risk: number
  high_risk: number
}

function RiskBadge({ label }: { label: string }) {
  const colors: Record<string, string> = {
    low: 'bg-risk-low/10 text-risk-low',
    high: 'bg-risk-high/10 text-risk-high',
  }
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium ${colors[label] || 'bg-slate-100 text-slate-600'}`}>
      {label.charAt(0).toUpperCase() + label.slice(1)}
    </span>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null)
  const [topRisk, setTopRisk] = useState<RiskMedication[]>([])
  const [categories, setCategories] = useState<CategoryBreakdown[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const location = useLocation()

  const fetchData = useCallback(() => {
    if (!getToken()) {
      setError('Not logged in. Please log in first.')
      setLoading(false)
      return
    }

    setLoading(true)
    setError(null)
    apiFetch('/predictions/dashboard')
      .then((data) => {
        setStats(data.stats)
        setTopRisk(data.top_risk_medications)
        setCategories(data.category_breakdown)
        setLoading(false)
      })
      .catch((err) => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  useEffect(() => {
    fetchData()
  }, [location.key, fetchData])

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-primary-600 animate-spin" />
      </div>
    )
  }

  if (error) {
    return (
      <div className="space-y-6">
        <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center">
          <AlertTriangle className="w-12 h-12 text-risk-medium mx-auto mb-4" />
          <p className="text-slate-600 mb-2">{error}</p>
          <p className="text-sm text-slate-400">
            Upload data on the <a href="/data" className="text-primary-600 underline">Data</a> page to get started.
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-900">Dashboard</h1>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchData}
            disabled={loading}
            className="flex items-center gap-1 px-3 py-1.5 text-sm text-slate-600 bg-white border border-slate-200 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <span className="text-xs text-risk-low bg-risk-low/10 px-2 py-1 rounded font-medium">
            Live Data
          </span>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">Total Medications</p>
              <p className="text-2xl font-bold text-slate-900">{stats?.total_medications}</p>
            </div>
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <Pill className="w-5 h-5 text-primary-600" />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">High Risk</p>
              <p className="text-2xl font-bold text-risk-high">{stats?.high_risk}</p>
            </div>
            <div className="w-10 h-10 bg-risk-high/10 rounded-lg flex items-center justify-center">
              <AlertTriangle className="w-5 h-5 text-risk-high" />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">Low Risk</p>
              <p className="text-2xl font-bold text-risk-low">{stats?.low_risk}</p>
            </div>
            <div className="w-10 h-10 bg-risk-low/10 rounded-lg flex items-center justify-center">
              <CheckCircle className="w-5 h-5 text-risk-low" />
            </div>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl shadow-sm border border-slate-200">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-slate-500">Avg Risk Score</p>
              <p className="text-2xl font-bold text-slate-900">
                {stats ? (stats.avg_risk_score * 100).toFixed(1) : 0}%
              </p>
            </div>
            <div className="w-10 h-10 bg-risk-medium/10 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-risk-medium" />
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* High Risk Medications */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200">
          <div className="p-4 border-b border-slate-200">
            <h2 className="font-semibold text-slate-900">Highest Risk Medications</h2>
          </div>
          <div className="p-4">
            <div className="space-y-3">
              {topRisk.slice(0, 8).map((med) => (
                <div
                  key={med.medication_id}
                  className="flex items-center justify-between p-3 bg-slate-50 rounded-lg"
                >
                  <div>
                    <p className="font-medium text-slate-900">{med.medication_name}</p>
                    <p className="text-xs text-slate-500">{med.medication_category}</p>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="text-right">
                      <p className="text-sm font-medium text-slate-900">
                        {(med.risk_score * 100).toFixed(0)}%
                      </p>
                    </div>
                    <RiskBadge label={med.risk_label} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Category Breakdown */}
        <div className="bg-white rounded-xl shadow-sm border border-slate-200">
          <div className="p-4 border-b border-slate-200">
            <h2 className="font-semibold text-slate-900">Risk by Category</h2>
          </div>
          <div className="p-4">
            <div className="space-y-3">
              {categories
                .sort((a, b) => b.avg_risk - a.avg_risk)
                .map((cat) => (
                  <div key={cat.medication_category} className="flex items-center gap-4">
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm font-medium text-slate-700 capitalize">
                          {cat.medication_category}
                        </span>
                        <span className="text-xs text-slate-500">
                          {cat.count} meds
                        </span>
                      </div>
                      <div className="w-full bg-slate-100 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full ${
                            cat.avg_risk >= 0.5 ? 'bg-risk-high' : cat.avg_risk >= 0.3 ? 'bg-risk-medium' : 'bg-risk-low'
                          }`}
                          style={{ width: `${Math.min(cat.avg_risk * 100, 100)}%` }}
                        />
                      </div>
                      <div className="flex justify-between mt-1">
                        <span className="text-xs text-slate-500">
                          Avg: {(cat.avg_risk * 100).toFixed(0)}%
                        </span>
                        <span className="text-xs text-risk-high">
                          {cat.high_risk} high risk
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        </div>
      </div>

      {/* Risk Factors */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200">
        <div className="p-4 border-b border-slate-200">
          <h2 className="font-semibold text-slate-900">How Risk Is Calculated</h2>
        </div>
        <div className="p-4">
          <div className="grid md:grid-cols-3 gap-4">
            <div className="p-4 bg-slate-50 rounded-lg">
              <p className="text-sm font-medium text-slate-700 mb-1">Supply Chain</p>
              <p className="text-xs text-slate-500">
                Supplier reliability scores and lead times directly impact risk predictions
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-lg">
              <p className="text-sm font-medium text-slate-700 mb-1">Inventory Levels</p>
              <p className="text-xs text-slate-500">
                Current inventory relative to capacity and reorder points
              </p>
            </div>
            <div className="p-4 bg-slate-50 rounded-lg">
              <p className="text-sm font-medium text-slate-700 mb-1">Historical Patterns</p>
              <p className="text-xs text-slate-500">
                Past shortage events and their recency influence future risk
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
