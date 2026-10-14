import { useState } from 'react'
import { Outlet, useNavigate, useLocation } from 'react-router-dom'
import PublicHeader from './PublicHeader'
import { logout, logoutAll } from '../api'
import { lockBrowserPlan } from '../browserForecast'

export default function Layout() {
  const navigate = useNavigate()
  const isDemo = useLocation().pathname === '/demo'
  const [logoutError, setLogoutError] = useState('')

  async function handleLogout(allSessions = false) {
    lockBrowserPlan()
    setLogoutError('')
    try {
      if (allSessions) await logoutAll()
      else await logout()
    } catch (err) {
      if (!(err instanceof Error) || err.message !== 'Unauthorized') {
        setLogoutError(`Could not end ${allSessions ? 'all server sessions' : 'the server session'}. Your local plan is locked; retry when connected.`)
        return
      }
    }
    navigate('/')
  }

  return <div className="pulse-fork-workspace min-h-screen bg-paper font-sans text-ink">
    <PublicHeader overlay workspace demo={isDemo} onLogout={() => handleLogout()} onLogoutAll={() => handleLogout(true)} />
    {logoutError && <p role="alert" className="mx-auto max-w-[1060px] px-4 text-sm text-risk-high sm:px-8">{logoutError}</p>}
    <main className="pulse-fork-content page-rail mx-auto min-h-[calc(100vh-96px)] max-w-[1060px] px-4 py-8 sm:px-8 sm:py-12 lg:px-12"><Outlet /></main>
  </div>
}
