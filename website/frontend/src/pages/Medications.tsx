import { useState, useEffect, useCallback } from 'react'
import { useLocation } from 'react-router-dom'
import { Search, Filter, ArrowUpDown, Loader2, AlertTriangle } from 'lucide-react'
import { apiFetch, getToken } from '../api'

interface Medication {
  medication_id: string
  medication_name: string
  medication_category: string
  risk_score: number
  risk_label: string
  confidence: number
}

type SortKey = 'medication_name' | 'medication_category' | 'risk_score'

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

export default function Medications() {
  const [medications, setMedications] = useState<Medication[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [search, setSearch] = useState('')
  const [sortKey, setSortKey] = useState<SortKey>('risk_score')
  const [sortAsc, setSortAsc] = useState(false)
  const [filterCategory, setFilterCategory] = useState('all')
  const location = useLocation()

  const fetchData = useCallback(() => {
    if (!getToken()) {
      setError('Not logged in. Please log in first.')
      setLoading(false)
      return
    }

    setLoading(true)
    apiFetch('/predictions/all')
      .then((data) => {
        setMedications(data.predictions)
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

  const categories = ['all', ...new Set(medications.map(m => m.medication_category))]

  const filtered = medications
    .filter(m =>
      (filterCategory === 'all' || m.medication_category === filterCategory) &&
      (m.medication_name.toLowerCase().includes(search.toLowerCase()) ||
       m.medication_id.toLowerCase().includes(search.toLowerCase()))
    )
    .sort((a, b) => {
      const aVal = a[sortKey]
      const bVal = b[sortKey]
      const cmp = typeof aVal === 'string' ? aVal.localeCompare(bVal as string) : (aVal as number) - (bVal as number)
      return sortAsc ? cmp : -cmp
    })

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(false)
    }
  }

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
        <h1 className="text-2xl font-bold text-slate-900">Medications</h1>
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
        <h1 className="text-2xl font-bold text-slate-900">Medications</h1>
        <span className="text-xs text-risk-low bg-risk-low/10 px-2 py-1 rounded font-medium">
          {medications.length} medications
        </span>
      </div>

      {/* Filters */}
      <div className="bg-white p-4 rounded-xl shadow-sm border border-slate-200">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-400" />
              <input
                type="text"
                placeholder="Search medications..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="w-full pl-10 pr-4 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
              />
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-slate-500" />
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value)}
              className="px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {categories.map(cat => (
                <option key={cat} value={cat}>
                  {cat === 'all' ? 'All Categories' : cat.charAt(0).toUpperCase() + cat.slice(1)}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-slate-200">
            <tr>
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('medication_name')}
                  className="flex items-center gap-1 text-sm font-medium text-slate-700 hover:text-slate-900"
                >
                  Medication
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th className="px-4 py-3 text-left">
                <button
                  onClick={() => handleSort('medication_category')}
                  className="flex items-center gap-1 text-sm font-medium text-slate-700 hover:text-slate-900"
                >
                  Category
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th className="px-4 py-3 text-right">
                <button
                  onClick={() => handleSort('risk_score')}
                  className="flex items-center gap-1 text-sm font-medium text-slate-700 hover:text-slate-900 ml-auto"
                >
                  Risk Score
                  <ArrowUpDown className="w-3 h-3" />
                </button>
              </th>
              <th className="px-4 py-3 text-center">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-200">
            {filtered.map((med) => (
              <tr key={med.medication_id} className="hover:bg-slate-50">
                <td className="px-4 py-3">
                  <p className="font-medium text-slate-900">{med.medication_name}</p>
                  <p className="text-xs text-slate-500">{med.medication_id}</p>
                </td>
                <td className="px-4 py-3">
                  <span className="text-sm text-slate-600 capitalize">{med.medication_category}</span>
                </td>
                <td className="px-4 py-3 text-right">
                  <span className="text-sm font-medium text-slate-900">
                    {(med.risk_score * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-4 py-3 text-center">
                  <RiskBadge label={med.risk_label} />
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        {filtered.length === 0 && (
          <div className="p-8 text-center text-slate-500">
            No medications found matching your criteria.
          </div>
        )}
      </div>
    </div>
  )
}
