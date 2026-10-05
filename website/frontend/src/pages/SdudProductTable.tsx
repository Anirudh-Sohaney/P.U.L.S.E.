import { useEffect, useState } from 'react'
import { Search } from 'lucide-react'
import { apiFetch } from '../api'

type Product = {
  product_name: string
  reported_prescriptions_lower_bound: number
  ndc_utilization_rows: number
  suppressed_rows: number
}

type Snapshot = {
  period: string | null
  products: Product[]
  filtered_total: number
  source_url?: string
}

const PAGE_SIZE = 25

export default function SdudProductTable({ period }: { period: string }) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const [query, setQuery] = useState('')
  const [page, setPage] = useState(0)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState('')

  useEffect(() => {
    const timer = window.setTimeout(() => {
      setPage(0)
      setQuery(search.trim())
    }, 300)
    return () => window.clearTimeout(timer)
  }, [search])

  useEffect(() => {
    if (!open) return
    let active = true
    setSnapshot(null)
    const refresh = () => {
      apiFetch(`/v1/signals/sources/sdud/latest?search=${encodeURIComponent(query)}&limit=${PAGE_SIZE}&offset=${page * PAGE_SIZE}`)
        .then((value: Snapshot) => { if (active) { setSnapshot(value); setError('') } })
        .catch((reason: unknown) => {
          if (active) {
            setSnapshot(null)
            setError(reason instanceof Error ? reason.message : 'Could not load observed prescriptions')
          }
        })
    }
    refresh()
    const timer = window.setInterval(refresh, 5 * 60_000)
    return () => { active = false; window.clearInterval(timer) }
  }, [open, query, page, period])

  return <section className="paper-card overflow-hidden rounded-2xl">
    <button type="button" aria-expanded={open} onClick={() => setOpen(value => !value)}
      className="flex w-full items-center justify-between gap-4 p-6 text-left">
      <span><span className="block font-serif text-2xl text-ink">Observed Medicaid prescriptions</span>
        <span className="mt-1 block text-sm text-slate-600">Explore reported Arkansas products for {period}</span></span>
      <span aria-hidden="true" className="text-xl text-slate-500">{open ? '−' : '+'}</span>
    </button>
    {open && <div className="border-t border-slate-200">
      <div className="flex flex-wrap items-start justify-between gap-4 p-6">
        <p className="max-w-2xl text-sm text-slate-600">Counts are lower bounds because suppressed NDC rows have unknown values. Product names appear as published by Medicaid and may be abbreviated. These are statewide quarterly prescriptions, not pharmacy inventory, a demand forecast, or units to order.</p>
        <label className="flex w-full max-w-xs items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-sm">
          <Search className="h-4 w-4 text-slate-400" />
          <input value={search} maxLength={120} onChange={event => setSearch(event.target.value)}
            placeholder="Search reported products" aria-label="Search observed Medicaid products"
            className="w-full bg-transparent outline-none" />
        </label>
      </div>
      {error && <p role="alert" className="mx-6 mb-4 rounded-lg bg-red-50 p-3 text-sm text-red-700">{error}</p>}
      {!snapshot && !error && <p className="px-6 pb-6 text-sm text-slate-500">Loading observed products…</p>}
      {snapshot && <>
        <div className="overflow-x-auto"><table className="w-full text-sm">
          <thead className="bg-slate-50 text-slate-600"><tr>
            <th className="px-6 py-3 text-left">Reported product name</th>
            <th className="px-6 py-3 text-right">Reported prescriptions, lower bound</th>
            <th className="px-6 py-3 text-right">Suppressed NDC rows</th>
          </tr></thead>
          <tbody className="divide-y divide-slate-100">{snapshot.products.map(product => <tr key={product.product_name}>
            <td className="px-6 py-3 font-medium text-ink">{product.product_name}</td>
            <td className="px-6 py-3 text-right tabular-nums">{product.reported_prescriptions_lower_bound.toLocaleString()}</td>
            <td className="px-6 py-3 text-right tabular-nums text-slate-600">{product.suppressed_rows.toLocaleString()}</td>
          </tr>)}</tbody>
        </table>{snapshot.products.length === 0 && <p className="p-6 text-sm text-slate-500">No reported products match this search.</p>}</div>
        <div className="flex items-center justify-between gap-3 border-t border-slate-200 px-6 py-4 text-sm">
          <span className="text-slate-500">{snapshot.period} · {snapshot.filtered_total.toLocaleString()} products · Page {page + 1} of {Math.max(1, Math.ceil(snapshot.filtered_total / PAGE_SIZE))}</span>
          <div className="flex gap-2">
            <button type="button" disabled={page === 0} onClick={() => setPage(value => value - 1)} className="rounded-full border border-slate-200 px-3 py-2 disabled:opacity-40">Previous</button>
            <button type="button" disabled={(page + 1) * PAGE_SIZE >= snapshot.filtered_total} onClick={() => setPage(value => value + 1)} className="rounded-full border border-slate-200 px-3 py-2 disabled:opacity-40">Next</button>
          </div>
        </div>
      </>}
    </div>}
  </section>
}
