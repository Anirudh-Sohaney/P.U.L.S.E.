import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import SignalsPage from './pages/SignalsPage'
import ApiGuide from './pages/ApiGuide'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route element={<Layout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/signals" element={<SignalsPage />} />
          <Route path="/api-guide" element={<ApiGuide />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App
