import { Outlet, NavLink, useNavigate } from 'react-router-dom'
import { Activity, LayoutDashboard, LogOut } from 'lucide-react'
import { getToken, clearToken } from '../api'

export default function Layout() {
  const navigate = useNavigate()
  const isLoggedIn = !!getToken()

  function handleLogout() {
    clearToken()
    navigate('/')
  }

  return <div className="min-h-screen bg-paper font-sans text-ink">
    <header className="border-b border-slate-200"><div className="page-rail mx-auto max-w-[1060px] px-4 py-4 sm:px-8 sm:py-5"><nav className="flex min-h-14 items-center justify-between gap-3 rounded-full border border-white bg-paper/90 px-4 shadow-[0_0_0_2px_white] backdrop-blur-sm sm:px-6" aria-label="Workspace navigation"><NavLink to="/" className="flex items-center gap-2.5 text-lg font-medium tracking-tight text-ink"><span className="flex h-8 w-8 items-center justify-center rounded-full bg-ink text-white"><Activity className="h-4 w-4" /></span>PULSE</NavLink><div className="flex items-center gap-2"><NavLink to="/dashboard" className={({ isActive }) => `inline-flex items-center gap-2 rounded-full px-4 py-2 text-sm font-medium ${isActive ? 'bg-white text-ink shadow-[0_1px_2px_rgba(55,50,47,0.12)]' : 'text-slate-600 hover:text-ink'}`}><LayoutDashboard className="h-4 w-4" /> <span className="hidden sm:inline">Dashboard</span></NavLink>{isLoggedIn && <button onClick={handleLogout} className="inline-flex items-center gap-2 rounded-full px-3 py-2 text-sm font-medium text-slate-600 hover:bg-white hover:text-ink sm:px-4"><LogOut className="h-4 w-4" /><span className="hidden sm:inline">Log out</span></button>}</div></nav></div></header>
    <main className="page-rail mx-auto min-h-[calc(100vh-96px)] max-w-[1060px] px-4 py-8 sm:px-8 sm:py-12 lg:px-12"><Outlet /></main>
  </div>
}
