import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import Medications from './pages/Medications'
import DataUpload from './pages/DataUpload'
import ModelPage from './pages/ModelPage'
import Forecasts from './pages/Forecasts'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route element={<Layout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/medications" element={<Medications />} />
          <Route path="/data" element={<DataUpload />} />
          <Route path="/forecasts" element={<Forecasts />} />
          <Route path="/model" element={<ModelPage />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App
