import { useState, useEffect } from 'react'
import { Brain, TrendingUp, CheckCircle, Loader2 } from 'lucide-react'
import { apiFetch, getToken } from '../api'

interface ModelMetrics {
  accuracy: number
  precision: number
  recall: number
  f1_score: number
  roc_auc: number
  true_positives: number
  true_negatives: number
  false_positives: number
  false_negatives: number
}

interface Feature {
  feature: string
  importance: number
}

const featureDescriptions: Record<string, string> = {
  supply_risk_score: 'Combined supplier reliability and lead time risk',
  inventory_coverage_days: 'Number of days inventory will last at current demand',
  demand_supply_mismatch: 'Ratio of demand to available supply',
  shortage_frequency: 'How often shortages occur (monthly rate)',
  days_of_supply_remaining: 'Estimated days until stockout',
  inventory_ratio: 'Current inventory level relative to maximum capacity',
  recency_weighted_shortage: 'Historical shortages weighted by how recent they were',
  demand_per_capacity: 'Weekly demand as percentage of storage capacity',
  reliability_inventory_interaction: 'Supplier reliability weighted by inventory level',
  inventory_deficit: 'Inventory level compared to reorder point',
  below_reorder: 'Whether inventory is currently below reorder point',
  low_supplier_reliability: 'Whether supplier reliability is below threshold',
  high_shortage_history: 'Whether medication has high historical shortage count',
  demand_volatility: 'Variation in demand over time',
}

export default function ModelPage() {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null)
  const [features, setFeatures] = useState<Feature[]>([])
  const [modelInfo, setModelInfo] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [training, setTraining] = useState(false)

  useEffect(() => {
    if (!getToken()) {
      setError('Not logged in. Please log in first.')
      setLoading(false)
      return
    }

    Promise.all([
      apiFetch('/model/info').catch(() => null),
      apiFetch('/model/features').catch(() => null),
    ])
      .then(([info, featuresData]) => {
        if (info) {
          setModelInfo(info)
          if (info.metrics) setMetrics(info.metrics)
        }
        if (featuresData) {
          setFeatures(featuresData.features.sort((a: Feature, b: Feature) => b.importance - a.importance))
        }
        setLoading(false)
      })
      .catch(() => {
        setLoading(false)
      })
  }, [])

  const handleRetrain = async () => {
    setTraining(true)
    try {
      const result = await apiFetch('/model/train', { method: 'POST' })
      if (result.metrics) {
        setMetrics(result.metrics)
      }
      // Reload features
      const featuresData = await apiFetch('/model/features').catch(() => null)
      if (featuresData) {
        setFeatures(featuresData.features.sort((a: Feature, b: Feature) => b.importance - a.importance))
      }
    } catch (err: any) {
      alert(err.message)
    }
    setTraining(false)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center py-20">
        <Loader2 className="w-8 h-8 text-primary-600 animate-spin" />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-slate-900">Model</h1>
        {metrics ? (
          <span className="text-xs text-risk-low bg-risk-low/10 px-2 py-1 rounded font-medium">
            Active
          </span>
        ) : (
          <span className="text-xs text-slate-500 bg-slate-100 px-2 py-1 rounded">
            No Model Trained
          </span>
        )}
      </div>

      {!metrics && !error && (
        <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-8 text-center">
          <Brain className="w-12 h-12 text-slate-400 mx-auto mb-4" />
          <p className="text-slate-600 mb-2">No model has been trained yet</p>
          <p className="text-sm text-slate-400 mb-4">
            Upload data on the <a href="/data" className="text-primary-600 underline">Data</a> page to train a model automatically.
          </p>
          <button
            onClick={handleRetrain}
            disabled={training}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors disabled:opacity-50"
          >
            {training ? 'Training...' : 'Train Model'}
          </button>
        </div>
      )}

      {metrics && (
        <>
          {/* Model Overview */}
          <div className="grid md:grid-cols-2 gap-6">
            {/* Model Info */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200">
              <div className="p-4 border-b border-slate-200">
                <div className="flex items-center gap-2">
                  <Brain className="w-5 h-5 text-primary-600" />
                  <h2 className="font-semibold text-slate-900">Model Information</h2>
                </div>
              </div>
              <div className="p-4 space-y-3">
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Algorithm</span>
                  <span className="text-sm font-medium text-slate-900">XGBoost</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Version</span>
                  <span className="text-sm font-medium text-slate-900">
                    {modelInfo?.metadata?.model_params?.random_state ? 'v1' : 'v1'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-slate-500">Trained At</span>
                  <span className="text-sm font-medium text-slate-900">
                    {modelInfo?.metadata?.trained_at
                      ? new Date(modelInfo.metadata.trained_at).toLocaleString()
                      : 'Unknown'}
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-slate-500">Status</span>
                  <span className="flex items-center gap-1 text-sm font-medium text-risk-low">
                    <CheckCircle className="w-4 h-4" /> Active
                  </span>
                </div>
                <div className="pt-2">
                  <button
                    onClick={handleRetrain}
                    disabled={training}
                    className="w-full px-3 py-2 border border-slate-300 text-slate-700 rounded-lg hover:bg-slate-50 transition-colors disabled:opacity-50 text-sm"
                  >
                    {training ? 'Retraining...' : 'Retrain Model'}
                  </button>
                </div>
              </div>
            </div>

            {/* Performance Metrics */}
            <div className="bg-white rounded-xl shadow-sm border border-slate-200">
              <div className="p-4 border-b border-slate-200">
                <div className="flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-teal-600" />
                  <h2 className="font-semibold text-slate-900">Performance Metrics</h2>
                </div>
              </div>
              <div className="p-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="text-center p-3 bg-slate-50 rounded-lg">
                    <p className="text-2xl font-bold text-primary-600">
                      {(metrics.roc_auc * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500">ROC AUC</p>
                  </div>
                  <div className="text-center p-3 bg-slate-50 rounded-lg">
                    <p className="text-2xl font-bold text-primary-600">
                      {(metrics.f1_score * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500">F1 Score</p>
                  </div>
                  <div className="text-center p-3 bg-slate-50 rounded-lg">
                    <p className="text-2xl font-bold text-primary-600">
                      {(metrics.precision * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500">Precision</p>
                  </div>
                  <div className="text-center p-3 bg-slate-50 rounded-lg">
                    <p className="text-2xl font-bold text-primary-600">
                      {(metrics.recall * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-slate-500">Recall</p>
                  </div>
                </div>
                <div className="mt-4 p-3 bg-slate-50 rounded-lg">
                  <div className="grid grid-cols-4 gap-2 text-center text-xs">
                    <div>
                      <p className="font-medium text-slate-700">{metrics.true_positives}</p>
                      <p className="text-slate-500">True Pos</p>
                    </div>
                    <div>
                      <p className="font-medium text-slate-700">{metrics.true_negatives}</p>
                      <p className="text-slate-500">True Neg</p>
                    </div>
                    <div>
                      <p className="font-medium text-risk-high">{metrics.false_positives}</p>
                      <p className="text-slate-500">False Pos</p>
                    </div>
                    <div>
                      <p className="font-medium text-risk-high">{metrics.false_negatives}</p>
                      <p className="text-slate-500">False Neg</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Feature Importance */}
          {features.length > 0 && (
            <div className="bg-white rounded-xl shadow-sm border border-slate-200">
              <div className="p-4 border-b border-slate-200">
                <h2 className="font-semibold text-slate-900">Feature Importance</h2>
              </div>
              <div className="p-4">
                <div className="space-y-3">
                  {features.slice(0, 10).map((feature, i) => (
                    <div key={i} className="flex items-center gap-4">
                      <div className="flex-1">
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm font-medium text-slate-700">
                            {feature.feature}
                          </span>
                          <span className="text-xs text-slate-500">
                            {(feature.importance * 100).toFixed(0)}%
                          </span>
                        </div>
                        <div className="w-full bg-slate-100 rounded-full h-2">
                          <div
                            className="bg-primary-600 h-2 rounded-full"
                            style={{ width: `${feature.importance * 100}%` }}
                          />
                        </div>
                        <p className="text-xs text-slate-500 mt-1">
                          {featureDescriptions[feature.feature] || feature.feature}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Disclaimer */}
      <div className="bg-slate-50 rounded-xl p-4 border border-slate-200">
        <p className="text-xs text-slate-500 text-center">
          Feature importance indicates association with predictions, not causation.
          Model estimates should be used alongside professional pharmacist judgment.
        </p>
      </div>
    </div>
  )
}
