import { apiFetch, ApiError } from './api'
import { savePrivatePlan, unlockPrivateStore, lockPrivateStore } from './privateStore'

export type BrowserPlan = {
  items: Array<{ drug_name: string; on_hand_units: number; on_order_units?: number; expected_arrival_date?: string | null; incoming_eta_overdue?: boolean; incoming_counted_by_day_14?: boolean; as_of_date?: string; forecast_1d_units: number; forecast_7d_units: number; forecast_14d_units: number; days_until_stockout: number | null; minimum_buy_1d_units: number; minimum_buy_7d_units: number; minimum_buy_14d_units: number; recommended_order_units: number; inventory_status: string }>
  per_drug: Array<{ drug_name: string; wape: number; signals_used: string[] }>
  history_start: string
  history_end: string
  signal_years: number
  model: string
  note: string
}

let currentPlan: BrowserPlan | null = null
let currentUsername: string | null = null
let workspaceGeneration = 0
let lastActivityAt = Date.now()
let sessionCheckRunning = false
let activeTrainingCancel: (() => void) | null = null
const LOCK_SIGNAL_KEY = 'pulse-private-lock'
const IDLE_LOCK_MS = 15 * 60 * 1000
const CATALOG_PAGE_SIZE = 1500

type CatalogSignal = {
  id: string
  latest_observation_date: string | null
  signal_origin: string
  signal_id: string
}

async function loadSignalCatalog(assertWorkspace: () => void): Promise<CatalogSignal[]> {
  const signals: CatalogSignal[] = []
  for (let offset = 0; ; offset += CATALOG_PAGE_SIZE) {
    assertWorkspace()
    const page = await apiFetch(`/v1/signals/catalog?limit=${CATALOG_PAGE_SIZE}&offset=${offset}`)
    if (!Array.isArray(page.signals)) throw new Error('Signal catalog response is incomplete')
    signals.push(...page.signals)
    if (page.signals.length < CATALOG_PAGE_SIZE) break
  }
  return signals
}

async function loadSignalHistory(ids: string[], startDate: string, endDate: string,
  assertWorkspace: () => void): Promise<unknown[]> {
  for (let attempt = 0; attempt < 4; attempt++) {
    assertWorkspace()
    try {
      const response = await apiFetch('/v1/signals/history', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ids, start_date: startDate, end_date: endDate,
          include_revisions: true }),
      })
      return response.values
    } catch (error) {
      if (error instanceof ApiError && error.status === 413 && ids.length >= 2) {
        const middle = Math.floor(ids.length / 2)
        const first = await loadSignalHistory(ids.slice(0, middle), startDate, endDate, assertWorkspace)
        const second = await loadSignalHistory(ids.slice(middle), startDate, endDate, assertWorkspace)
        return first.concat(second)
      }
      if (error instanceof ApiError && error.status === 429 &&
        error.retryAfterSeconds !== null && attempt < 3) {
        await new Promise(resolve => window.setTimeout(resolve,
          Math.min(30, Math.ceil(error.retryAfterSeconds ?? 1)) * 1000))
        continue
      }
      throw error
    }
  }
  throw new Error('Signal history request limit persisted after retries')
}

function clearBrowserPlan() {
  workspaceGeneration += 1
  activeTrainingCancel?.()
  currentPlan = null
  currentUsername = null
  lockPrivateStore()
  window.dispatchEvent(new Event('pulse-private-locked'))
}

async function checkPrivateSession() {
  const username = currentUsername
  const generation = workspaceGeneration
  if (!username) return
  if (Date.now() - lastActivityAt >= IDLE_LOCK_MS) {
    lockBrowserPlan()
    return
  }
  if (sessionCheckRunning) return
  sessionCheckRunning = true
  try {
    const user = await apiFetch('/auth/me', { signal: AbortSignal.timeout(10_000) })
    if (generation === workspaceGeneration && currentUsername === username && user.username !== username) {
      lockBrowserPlan()
    }
  } catch {
    if (generation === workspaceGeneration && currentUsername === username) lockBrowserPlan()
  } finally {
    sessionCheckRunning = false
  }
}

const markActivity = () => { if (currentUsername) lastActivityAt = Date.now() }
window.addEventListener('pointerdown', markActivity)
window.addEventListener('keydown', markActivity)
window.addEventListener('focus', () => { void checkPrivateSession() })
document.addEventListener('visibilitychange', () => {
  if (!document.hidden) void checkPrivateSession()
})
window.setInterval(() => { void checkPrivateSession() }, 60_000)

function broadcastPrivateLock() {
  try { localStorage.setItem(LOCK_SIGNAL_KEY, crypto.randomUUID()) } catch { /* Other tabs still recheck on focus. */ }
}

window.addEventListener('storage', event => {
  if (event.key === LOCK_SIGNAL_KEY) clearBrowserPlan()
})

export function getBrowserPlan(username: string) {
  return currentUsername === username ? currentPlan : null
}
export async function unlockBrowserPlan(username: string, password: string): Promise<BrowserPlan | null> {
  clearBrowserPlan()
  broadcastPrivateLock()
  const generation = workspaceGeneration
  const plan = await unlockPrivateStore(username, password) as BrowserPlan | null
  if (generation !== workspaceGeneration) throw new Error('The workspace changed while it was unlocking')
  currentPlan = plan
  currentUsername = username
  lastActivityAt = Date.now()
  return currentPlan
}
export function lockBrowserPlan() {
  const wasUnlocked = currentUsername !== null
  clearBrowserPlan()
  if (wasUnlocked) broadcastPrivateLock()
}

export async function trainInBrowser(sales: File, inventory: File, signalYears: number, onProgress: (message: string) => void): Promise<BrowserPlan> {
  const trainingUsername = currentUsername
  const generation = workspaceGeneration
  if (!trainingUsername) throw new Error('Unlock your private workspace before training')
  if (activeTrainingCancel) throw new Error('Training is already running in this tab')
  const assertWorkspace = () => {
    if (generation !== workspaceGeneration || currentUsername !== trainingUsername) {
      throw new Error('The workspace changed during training. Unlock it and train again.')
    }
  }
  if (sales.size > 50 * 1024 * 1024 || inventory.size > 50 * 1024 * 1024) throw new Error('Each CSV must be under 50 MB')
  if (![0, 1, 2, 3].includes(signalYears)) throw new Error('Choose zero to three years of public signals')
  const [salesCsv, inventoryCsv] = await Promise.all([sales.text(), inventory.text()])
  assertWorkspace()
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL('./forecast.worker.ts', import.meta.url), { type: 'module' })
    const cancel = () => fail(new Error('The workspace was locked during training'))
    const fail = (error: Error) => {
      if (activeTrainingCancel === cancel) activeTrainingCancel = null
      worker.terminate()
      reject(error)
    }
    activeTrainingCancel = cancel
    worker.onmessage = async event => {
      const message = event.data
      if (message.type === 'progress') onProgress(message.message)
      if (message.type === 'prepared') {
        try {
          assertWorkspace()
          let signalRows: unknown[] = []
          if (signalYears > 0) {
            onProgress('Downloading public signal history')
            const catalogSignals = await loadSignalCatalog(assertWorkspace)
            // The trainer discards records outside this observation window and
            // model outputs that have no validated live publication path.
            const ids: string[] = catalogSignals
              .filter((row: CatalogSignal) =>
                row.latest_observation_date !== null &&
                row.latest_observation_date >= message.window.start_date &&
                row.signal_origin !== 'model_news_output' &&
                !row.signal_id.startsWith('arkansas_atc_demand_state::'))
              .map((row: CatalogSignal) => row.id)
            for (let index = 0; index < ids.length; index += 100) {
              assertWorkspace()
              const rows = await loadSignalHistory(ids.slice(index, index + 100),
                message.window.start_date, message.window.end_date, assertWorkspace)
              signalRows = signalRows.concat(rows)
              onProgress(`Loaded ${Math.min(index + 100, ids.length)} of ${ids.length} signal identities`)
            }
          }
          assertWorkspace()
          worker.postMessage({ type: 'train', signal_rows: signalRows })
        } catch (error) {
          fail(error instanceof Error ? error : new Error(String(error)))
        }
      }
      if (message.type === 'success') {
        if (activeTrainingCancel === cancel) activeTrainingCancel = null
        worker.terminate()
        try { assertWorkspace() } catch (error) { reject(error); return }
        savePrivatePlan(message.result).then(() => {
          assertWorkspace()
          currentPlan = message.result as BrowserPlan
          resolve(currentPlan)
        }).catch(reject)
      }
      if (message.type === 'error') fail(new Error(message.message))
    }
    worker.onerror = error => fail(new Error(error.message || 'Browser training failed'))
    const today = new Date()
    const localToday = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-${String(today.getDate()).padStart(2, '0')}`
    worker.postMessage({ type: 'prepare', sales_csv: salesCsv, inventory_csv: inventoryCsv,
      signal_years: signalYears, local_today: localToday })
  })
}
