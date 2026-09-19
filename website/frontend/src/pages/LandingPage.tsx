import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { ArrowRight, Shield, Brain, BarChart3, Pill, Loader2 } from 'lucide-react'
import { login, getToken } from '../api'

export default function LandingPage() {
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [loggingIn, setLoggingIn] = useState(false)
  const [loginError, setLoginError] = useState<string | null>(null)
  const [isLoggedIn, setIsLoggedIn] = useState(!!getToken())
  const navigate = useNavigate()

  useEffect(() => {
    if (getToken()) {
      setIsLoggedIn(true)
    }
  }, [])

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoggingIn(true)
    setLoginError(null)
    try {
      await login(username, password)
      setIsLoggedIn(true)
      navigate('/dashboard')
    } catch (err: any) {
      setLoginError(err.message)
    }
    setLoggingIn(false)
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16 items-center">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-primary-600 rounded-lg flex items-center justify-center">
                <Pill className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-semibold text-slate-900">
                Pharmacy Risk
              </span>
            </div>
            {isLoggedIn ? (
              <Link
                to="/dashboard"
                className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
              >
                Open Dashboard
              </Link>
            ) : null}
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="py-20 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-4xl font-bold text-slate-900 mb-6">
            Pharmacy Risk Prediction Platform
          </h1>
          <p className="text-xl text-slate-600 mb-8">
            Predict and manage pharmaceutical supply risks using machine learning.
            Make data-driven decisions to prevent medication shortages.
          </p>

          {!isLoggedIn ? (
            <div className="max-w-md mx-auto bg-white rounded-xl shadow-sm border border-slate-200 p-6">
              <h2 className="text-lg font-semibold text-slate-900 mb-4">Sign In</h2>
              <form onSubmit={handleLogin} className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Username</label>
                  <input
                    type="text"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                    placeholder="admin"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700 mb-1">Password</label>
                  <input
                    type="password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="w-full px-3 py-2 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500"
                    placeholder="admin123"
                    required
                  />
                </div>
                {loginError && (
                  <p className="text-sm text-risk-high">{loginError}</p>
                )}
                <button
                  type="submit"
                  disabled={loggingIn}
                  className="w-full px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {loggingIn && <Loader2 className="w-4 h-4 animate-spin" />}
                  {loggingIn ? 'Signing in...' : 'Sign In'}
                </button>
                <p className="text-xs text-slate-400 text-center">
                  Demo: admin / admin123
                </p>
              </form>
            </div>
          ) : (
            <div className="flex justify-center gap-4">
              <Link
                to="/dashboard"
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors flex items-center gap-2"
              >
                Go to Dashboard <ArrowRight className="w-4 h-4" />
              </Link>
            </div>
          )}
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-16 bg-slate-50">
        <div className="max-w-7xl mx-auto px-4">
          <h2 className="text-2xl font-bold text-center text-slate-900 mb-12">
            How It Works
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
                <BarChart3 className="w-6 h-6 text-primary-600" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">
                1. Analyze Data
              </h3>
              <p className="text-slate-600">
                Upload your pharmacy inventory, demand, and supplier data for analysis.
              </p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div className="w-12 h-12 bg-teal-100 rounded-lg flex items-center justify-center mb-4">
                <Brain className="w-6 h-6 text-teal-600" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">
                2. ML Prediction
              </h3>
              <p className="text-slate-600">
                XGBoost model analyzes patterns and predicts supply risk levels.
              </p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-sm">
              <div className="w-12 h-12 bg-risk-low/10 rounded-lg flex items-center justify-center mb-4">
                <Shield className="w-6 h-6 text-risk-low" />
              </div>
              <h3 className="text-lg font-semibold text-slate-900 mb-2">
                3. Take Action
              </h3>
              <p className="text-slate-600">
                Get clear risk scores and actionable insights to prevent shortages.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Disclaimer */}
      <section className="py-12 bg-slate-100">
        <div className="max-w-4xl mx-auto px-4 text-center">
          <p className="text-sm text-slate-500">
            This is a decision-support tool. Predictions are model estimates and should not
            replace professional pharmacist judgment. Always verify critical decisions with
            qualified healthcare professionals.
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-8 border-t border-slate-200">
        <div className="max-w-7xl mx-auto px-4 text-center text-slate-500 text-sm">
          <p>Pharmacy Risk Prediction Platform - Research/Development System</p>
        </div>
      </footer>
    </div>
  )
}
