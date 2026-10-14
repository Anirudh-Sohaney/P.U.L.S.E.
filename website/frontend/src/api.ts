const API_BASE = '/api'

try {
  localStorage.removeItem('token')
} catch {
  // Cookie sessions still work when browser storage is unavailable.
}

export class ApiError extends Error {
  constructor(message: string, public status: number, public retryAfterSeconds: number | null = null) {
    super(message)
    this.name = 'ApiError'
  }
}

function csrfToken(): string | null {
  const item = document.cookie.split('; ').find(cookie => cookie.startsWith('pulse_csrf='))
  return item ? decodeURIComponent(item.slice('pulse_csrf='.length)) : null
}

export async function isSignedIn(): Promise<boolean> {
  try { await apiFetch('/auth/me'); return true } catch { return false }
}

export async function logout(): Promise<void> {
  await apiFetch('/auth/logout', { method: 'POST' })
}

export async function logoutAll(): Promise<void> {
  await apiFetch('/auth/logout-all', { method: 'POST' })
}

export async function login(username: string, password: string): Promise<void> {
  const formData = new URLSearchParams()
  formData.append('username', username)
  formData.append('password', password)

  const res = await fetch(`${API_BASE}/auth/session`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    const err = await res.json()
    throw new Error(err.detail || 'Login failed')
  }

}

export async function register(username: string, password: string): Promise<void> {
  const res = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Registration failed' }))
    throw new Error(err.detail || 'Registration failed')
  }
}

export async function apiFetch(path: string, options: RequestInit = {}): Promise<any> {
  const headers: Record<string, string> = {
    ...(options.headers as Record<string, string> || {}),
  }

  if (options.method && !['GET', 'HEAD', 'OPTIONS'].includes(options.method.toUpperCase())) {
    const csrf = csrfToken()
    if (csrf) headers['X-CSRF-Token'] = csrf
  }

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: 'same-origin',
  })

  if (res.status === 401) {
    throw new ApiError('Unauthorized', 401)
  }

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Request failed' }))
    const delay = Number(res.headers.get('Retry-After'))
    throw new ApiError(err.detail || 'Request failed', res.status,
      res.status === 429 && Number.isFinite(delay) && delay > 0 ? delay : null)
  }

  return res.json()
}
