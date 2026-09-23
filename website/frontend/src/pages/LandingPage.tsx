import { useEffect, useState, type FormEvent } from 'react'
import { useNavigate } from 'react-router-dom'
import { Activity, ArrowRight, BarChart3, Check, ChevronRight, FileSpreadsheet, LockKeyhole, Loader2, PackageCheck, Sparkles } from 'lucide-react'
import { getToken, login, register } from '../api'

const features = [
  { icon: BarChart3, title: 'Forecast each medication', copy: 'A separate XGBoost model uses past sales and the most relevant dated signals for each drug.' },
  { icon: PackageCheck, title: 'Plan the next purchase', copy: 'See projected demand and minimum units needed for the next day, week, and two weeks.' },
  { icon: Sparkles, title: 'Understand the context', copy: 'Review relevant demand signals and related news alongside the inventory plan.' },
]

const steps = [
  { number: '01', title: 'Add sales history', copy: 'Upload a daily CSV with a date, medication name, and units sold.' },
  { number: '02', title: 'Add inventory', copy: 'Upload on-hand quantities for the same medications. The latest balance is used.' },
  { number: '03', title: 'Review your plan', copy: 'PULSE trains your account’s models and shows demand, stockout timing, and order guidance.' },
]

export default function LandingPage() {
  const [mode, setMode] = useState<'login' | 'signup'>('login')
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const navigate = useNavigate()

  useEffect(() => { if (getToken()) navigate('/dashboard') }, [navigate])

  function showAccess(nextMode: 'login' | 'signup') {
    setMode(nextMode)
    setError('')
    document.getElementById('access')?.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }

  async function submit(event: FormEvent) {
    event.preventDefault()
    setBusy(true)
    setError('')
    try {
      if (mode === 'signup') await register(username, password)
      else await login(username, password)
      navigate('/dashboard')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not sign in')
    } finally {
      setBusy(false)
    }
  }

  return <div id="top" className="min-h-screen overflow-x-hidden bg-paper font-sans text-ink">
    <div className="border-b border-slate-200/80">
      <div className="page-rail mx-auto max-w-[1060px] px-4 py-4 sm:px-8 sm:py-5">
        <nav className="flex min-h-14 items-center justify-between gap-4 rounded-full border border-white bg-paper/90 px-4 shadow-[0_0_0_2px_white] backdrop-blur-sm sm:px-6" aria-label="Main navigation">
          <a href="#top" className="flex items-center gap-2.5 text-lg font-medium tracking-tight text-slate-900"><span className="flex h-8 w-8 items-center justify-center rounded-full bg-ink text-white"><Activity className="h-4 w-4" /></span>PULSE</a>
          <div className="hidden items-center gap-7 text-[13px] font-medium text-slate-600 sm:flex"><a className="hover:text-ink" href="#features">Features</a><a className="hover:text-ink" href="#how-it-works">How it works</a><a className="hover:text-ink" href="#questions">Questions</a></div>
          <button onClick={() => showAccess('login')} className="rounded-full bg-white px-4 py-2 text-sm font-medium text-ink shadow-[0_1px_2px_rgba(55,50,47,0.12)] transition hover:bg-slate-100">Log in</button>
        </nav>
      </div>
    </div>

    <main className="page-rail mx-auto max-w-[1060px]">
      <section className="grid gap-12 border-b border-slate-200 px-6 pb-16 pt-16 sm:px-10 sm:pt-20 lg:grid-cols-[minmax(0,1.15fr)_minmax(330px,0.85fr)] lg:items-center lg:gap-16 lg:px-14 lg:pb-24 lg:pt-28">
        <div>
          <span className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-1.5 text-xs font-medium text-slate-800 shadow-[0_0_0_4px_rgba(55,50,47,0.04)]"><span className="h-1.5 w-1.5 rounded-full bg-teal-600" /> A calmer way to plan replenishment</span>
          <h1 className="mt-7 max-w-[650px] font-serif text-[48px] font-normal leading-[1.02] tracking-[-0.025em] text-ink sm:text-[64px] lg:text-[76px]">A clearer pulse on pharmacy demand.</h1>
          <p className="mt-7 max-w-[520px] text-base font-medium leading-relaxed text-slate-600 sm:text-lg">Turn your sales and inventory history into a focused 14-day plan—so you can see which medicines may need attention, and when.</p>
          <div className="mt-9 flex flex-wrap items-center gap-4"><button onClick={() => showAccess('signup')} className="dark-pill inline-flex h-12 items-center gap-3 px-7 text-sm font-medium">Create a workspace <ArrowRight className="h-4 w-4" /></button><a href="#how-it-works" className="inline-flex h-12 items-center gap-1 text-sm font-medium text-slate-700 hover:text-ink">See how it works <ChevronRight className="h-4 w-4" /></a></div>
          <p className="mt-7 flex items-center gap-2 text-xs text-slate-500"><Check className="h-4 w-4 text-teal-600" /> Separate models for each medication <span className="text-slate-300">·</span> Your files stay with your account</p>
        </div>
        <div id="access" className="paper-card scroll-mt-8 rounded-2xl p-7 sm:p-8">
          <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Your workspace</p>
          <h2 className="mt-3 font-serif text-4xl leading-tight text-ink">{mode === 'login' ? 'Welcome back.' : 'Make room for clarity.'}</h2>
          <p className="mt-3 text-sm leading-relaxed text-slate-600">{mode === 'login' ? 'Sign in to your saved forecast and replenishment plan.' : 'Create an account to upload your files and train your own models.'}</p>
          <div className="mt-7 grid grid-cols-2 rounded-full bg-slate-100 p-1 text-sm" role="group" aria-label="Account action"><button type="button" onClick={() => { setMode('login'); setError('') }} className={`rounded-full py-2 font-medium transition ${mode === 'login' ? 'bg-white text-ink shadow-sm' : 'text-slate-500'}`}>Log in</button><button type="button" onClick={() => { setMode('signup'); setError('') }} className={`rounded-full py-2 font-medium transition ${mode === 'signup' ? 'bg-white text-ink shadow-sm' : 'text-slate-500'}`}>Sign up</button></div>
          <form onSubmit={submit} className="mt-6 space-y-4">
            <label className="block text-sm font-medium text-slate-700">Username<input value={username} onChange={event => setUsername(event.target.value)} autoComplete="username" required minLength={3} maxLength={40} className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-ink outline-none transition focus:border-slate-500 focus:ring-2 focus:ring-slate-200" placeholder="Your username" /></label>
            <label className="block text-sm font-medium text-slate-700">Password<input type="password" value={password} onChange={event => setPassword(event.target.value)} autoComplete={mode === 'login' ? 'current-password' : 'new-password'} required minLength={mode === 'signup' ? 8 : 1} className="mt-2 w-full rounded-xl border border-slate-200 bg-white px-4 py-3 text-ink outline-none transition focus:border-slate-500 focus:ring-2 focus:ring-slate-200" placeholder="Your password" />{mode === 'signup' && <span className="mt-1 block text-xs text-slate-500">Use at least 8 characters.</span>}</label>
            {error && <p role="alert" className="text-sm text-risk-high">{error}</p>}
            <button disabled={busy} className="dark-pill flex w-full items-center justify-center gap-2 px-5 py-3.5 text-sm font-medium">{busy && <Loader2 className="h-4 w-4 animate-spin" />}{mode === 'login' ? 'Open my workspace' : 'Create my workspace'} <ArrowRight className="h-4 w-4" /></button>
          </form>
          <p className="mt-5 flex items-start gap-2 text-xs leading-relaxed text-slate-500"><LockKeyhole className="mt-0.5 h-3.5 w-3.5 shrink-0" /> Your uploaded data and trained model are stored with your account.</p>
        </div>
      </section>

      <section className="relative border-b border-slate-200 px-6 py-14 sm:px-10 lg:px-12 lg:py-20" aria-label="Workspace preview">
        <div className="hatch absolute bottom-14 left-0 top-14 hidden w-6 opacity-70 lg:block" aria-hidden="true" /><div className="hatch absolute bottom-14 right-0 top-14 hidden w-6 opacity-70 lg:block" aria-hidden="true" />
        <div className="mb-7 flex flex-wrap items-end justify-between gap-3"><div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Inside the workspace</p><h2 className="mt-3 font-serif text-3xl text-ink sm:text-4xl">The important details, in one place.</h2></div><span className="rounded-full border border-slate-200 bg-white px-3 py-1.5 text-xs text-slate-500">Illustrative interface</span></div>
        <div className="paper-card overflow-hidden rounded-xl"><div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-200 px-5 py-4 sm:px-7"><div className="flex items-center gap-2 text-sm font-semibold text-ink"><span className="flex h-7 w-7 items-center justify-center rounded-full bg-ink text-white"><Activity className="h-3.5 w-3.5" /></span>PULSE <span className="font-normal text-slate-400">/ Demand & inventory</span></div><span className="rounded-full bg-slate-100 px-3 py-1 text-xs text-slate-600">Account workspace</span></div><div className="grid gap-0 lg:grid-cols-[1.25fr_0.75fr]"><div className="border-b border-slate-200 p-6 lg:border-b-0 lg:border-r lg:p-8"><p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Restock priorities</p><h3 className="mt-2 font-serif text-3xl text-ink">See what may need ordering.</h3><p className="mt-2 max-w-[490px] text-sm leading-relaxed text-slate-600">Each medication gets a clear view of expected demand and inventory coverage for the next 1, 7, and 14 days.</p><div className="mt-7 overflow-hidden rounded-xl border border-slate-200"><div className="grid grid-cols-3 border-b border-slate-200 bg-slate-50 px-4 py-3 text-xs font-semibold text-slate-600"><span>Medication</span><span className="text-center">Demand horizon</span><span className="text-right">Plan</span></div><div className="grid grid-cols-3 items-center px-4 py-4 text-sm text-slate-700"><span>Drug-specific view</span><span className="text-center text-slate-500">1 · 7 · 14 days</span><span className="text-right"><span className="rounded-full bg-primary-50 px-3 py-1 text-xs font-medium text-ink">Stock check</span></span></div><div className="grid grid-cols-3 items-center border-t border-slate-100 px-4 py-4 text-sm text-slate-700"><span>Inventory balance</span><span className="text-center text-slate-500">Latest upload</span><span className="text-right"><span className="rounded-full bg-teal-50 px-3 py-1 text-xs font-medium text-teal-700">Order guidance</span></span></div></div></div><div className="bg-slate-50/60 p-6 lg:p-8"><p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-500">Relevant signals</p><h3 className="mt-2 font-serif text-3xl text-ink">A little more context.</h3><p className="mt-2 text-sm leading-relaxed text-slate-600">A short list of signals associated with medications that may run short.</p><div className="mt-7 space-y-3"><div className="rounded-xl border border-slate-200 bg-white p-4"><div className="flex items-center gap-2"><BarChart3 className="h-4 w-4 text-teal-700" /><span className="text-sm font-semibold text-slate-800">Demand context</span></div><div className="mt-4 flex h-12 items-end gap-2" aria-hidden="true">{[35, 44, 40, 57, 62, 58, 74, 80, 76, 90].map((height, index) => <span key={index} className="flex-1 rounded-t-sm bg-teal-200" style={{ height: `${height}%` }} />)}</div></div><div className="rounded-xl border border-slate-200 bg-white p-4"><div className="flex items-center gap-2"><Sparkles className="h-4 w-4 text-slate-700" /><span className="text-sm font-semibold text-slate-800">News context</span></div><p className="mt-2 text-xs leading-relaxed text-slate-500">Related article headlines appear alongside relevant signals.</p></div></div></div></div></div><p className="mt-4 text-xs text-slate-500">This is a layout preview, not a forecast. Your dashboard is populated only after you upload your own data.</p>
      </section>

      <section id="features" className="border-b border-slate-200 px-6 py-16 sm:px-10 lg:px-14 lg:py-24">
        <div className="max-w-[650px]"><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">What PULSE does</p><h2 className="mt-4 font-serif text-4xl leading-tight text-ink sm:text-5xl">Answers you can act on, without the noise.</h2><p className="mt-4 max-w-[570px] leading-relaxed text-slate-600">Daily sales and current stock become a concise view of demand, purchase needs, and the signals behind them.</p></div>
        <div className="mt-10 grid overflow-hidden rounded-2xl border border-slate-200 bg-white md:grid-cols-3">{features.map(({ icon: Icon, title, copy }) => <div key={title} className="border-b border-slate-200 p-7 last:border-0 md:border-b-0 md:border-r md:last:border-r-0 lg:p-9"><span className="flex h-11 w-11 items-center justify-center rounded-xl bg-slate-100 text-ink"><Icon className="h-5 w-5" /></span><h3 className="mt-7 text-base font-semibold text-slate-800">{title}</h3><p className="mt-3 text-sm leading-relaxed text-slate-600">{copy}</p></div>)}</div>
      </section>

      <section id="how-it-works" className="border-b border-slate-200 px-6 py-16 sm:px-10 lg:px-14 lg:py-24">
        <div className="grid gap-10 lg:grid-cols-[0.8fr_1.2fr] lg:gap-20"><div><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">A simple workflow</p><h2 className="mt-4 font-serif text-4xl leading-tight text-ink sm:text-5xl">Two files in. A clearer plan out.</h2><p className="mt-5 leading-relaxed text-slate-600">No complicated setup. Start with the records you already have, then let the backend handle the heavy modeling work.</p><span className="mt-8 inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-medium text-slate-700"><FileSpreadsheet className="h-4 w-4" /> CSV uploads · account-scoped</span></div><div className="divide-y divide-slate-200 border-y border-slate-200">{steps.map(step => <div key={step.number} className="grid grid-cols-[54px_1fr] gap-4 py-7"><span className="font-serif text-3xl text-slate-400">{step.number}</span><div><h3 className="text-base font-semibold text-slate-800">{step.title}</h3><p className="mt-2 text-sm leading-relaxed text-slate-600">{step.copy}</p></div></div>)}</div></div>
      </section>

      <section id="questions" className="border-b border-slate-200 px-6 py-16 sm:px-10 lg:px-14 lg:py-24"><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Good to know</p><h2 className="mt-4 font-serif text-4xl text-ink sm:text-5xl">A few practical details.</h2><div className="mt-10 grid gap-5 md:grid-cols-2"><div className="paper-card rounded-xl p-6"><h3 className="font-semibold text-slate-800">What files do I need?</h3><p className="mt-3 text-sm leading-relaxed text-slate-600">A daily sales CSV and an inventory CSV for the same medications. The upload screen shows the required columns.</p></div><div className="paper-card rounded-xl p-6"><h3 className="font-semibold text-slate-800">How are signals used?</h3><p className="mt-3 text-sm leading-relaxed text-slate-600">Each medication’s model selects up to five relevant dated signals from the catalog using only its training history.</p></div><div className="paper-card rounded-xl p-6"><h3 className="font-semibold text-slate-800">Can I update the plan?</h3><p className="mt-3 text-sm leading-relaxed text-slate-600">Yes. Upload newer sales and inventory files from your workspace to train a fresh model and replace its plan.</p></div><div className="paper-card rounded-xl p-6"><h3 className="font-semibold text-slate-800">Is this a clinical decision tool?</h3><p className="mt-3 text-sm leading-relaxed text-slate-600">No. PULSE is a research demo. Forecasts and purchase guidance need review by qualified pharmacy staff.</p></div></div></section>

      <section className="relative overflow-hidden px-6 py-16 text-center sm:px-10 lg:px-14 lg:py-24"><div className="hatch absolute inset-x-0 top-0 h-4 opacity-60" aria-hidden="true" /><p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">Ready when you are</p><h2 className="mx-auto mt-5 max-w-[620px] font-serif text-4xl leading-tight text-ink sm:text-5xl">Make the next replenishment decision a little clearer.</h2><p className="mx-auto mt-4 max-w-[500px] text-sm leading-relaxed text-slate-600">Start a private workspace, upload your history, and see what may need attention over the next 14 days.</p><button onClick={() => showAccess('signup')} className="dark-pill mt-8 inline-flex items-center gap-2 px-8 py-3 text-sm font-medium">Create a workspace <ArrowRight className="h-4 w-4" /></button></section>
    </main>
    <footer className="border-t border-slate-200"><div className="mx-auto flex max-w-[1060px] flex-wrap items-center justify-between gap-3 px-6 py-7 text-xs text-slate-500 sm:px-10 lg:px-14"><span className="font-medium tracking-wide text-slate-700">PULSE</span><span>Pharmacy demand planning · Research demo</span></div></footer>
  </div>
}
