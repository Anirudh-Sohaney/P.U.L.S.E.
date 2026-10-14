// Account-scoped encrypted browser storage for private forecast results.

const DB_NAME = 'pulse-private-v1'
const STORE_NAME = 'plans'
const ITERATIONS = 310_000

type EncryptedRecord = { username: string; salt: number[]; iv?: number[]; ciphertext?: number[]; revision?: number }

let activeUsername: string | null = null
let activeKey: CryptoKey | null = null
let activeSalt: number[] | null = null
let activeRevision: number | null = null
let unlockGeneration = 0

function openDatabase(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1)
    request.onupgradeneeded = () => { request.result.createObjectStore(STORE_NAME, { keyPath: 'username' }) }
    request.onsuccess = () => resolve(request.result)
    request.onerror = () => reject(request.error)
  })
}

async function readRecord(username: string): Promise<EncryptedRecord | undefined> {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readonly')
    const request = tx.objectStore(STORE_NAME).get(username)
    request.onsuccess = () => { db.close(); resolve(request.result as EncryptedRecord | undefined) }
    request.onerror = () => { db.close(); reject(request.error) }
  })
}

async function getOrCreateRecord(username: string): Promise<EncryptedRecord> {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    let record: EncryptedRecord
    const request = store.get(username)
    request.onsuccess = () => {
      record = request.result as EncryptedRecord | undefined ?? {
        username, salt: Array.from(crypto.getRandomValues(new Uint8Array(16))), revision: 0,
      }
      if (!request.result) store.add(record)
    }
    tx.oncomplete = () => { db.close(); resolve(record) }
    tx.onabort = () => { db.close(); reject(tx.error ?? new Error('Could not initialize the private workspace')) }
  })
}

async function writeIfCurrent(record: EncryptedRecord, expectedRevision: number): Promise<void> {
  const db = await openDatabase()
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORE_NAME, 'readwrite')
    const store = tx.objectStore(STORE_NAME)
    let conflict = false
    const request = store.get(record.username)
    request.onsuccess = () => {
      const current = request.result as EncryptedRecord | undefined
      if (!current || (current.revision ?? 0) !== expectedRevision ||
          current.salt.length !== record.salt.length ||
          current.salt.some((byte, index) => byte !== record.salt[index])) {
        conflict = true
        tx.abort()
        return
      }
      store.put(record)
    }
    tx.oncomplete = () => { db.close(); resolve() }
    tx.onabort = () => {
      db.close()
      reject(conflict ? new Error('This plan changed in another tab. Reload and unlock it before saving again.') :
        tx.error ?? new Error('Could not save the private plan'))
    }
  })
}

async function deriveKey(password: string, salt: Uint8Array): Promise<CryptoKey> {
  const material = await crypto.subtle.importKey('raw', new TextEncoder().encode(password), 'PBKDF2', false, ['deriveKey'])
  return crypto.subtle.deriveKey({ name: 'PBKDF2', salt: Uint8Array.from(salt).buffer, iterations: ITERATIONS, hash: 'SHA-256' },
    material, { name: 'AES-GCM', length: 256 }, false, ['encrypt', 'decrypt'])
}

export async function unlockPrivateStore(username: string, password: string): Promise<unknown | null> {
  lockPrivateStore()
  const generation = unlockGeneration
  const previous = await getOrCreateRecord(username)
  const salt = new Uint8Array(previous.salt)
  const key = await deriveKey(password, salt)
  let result: unknown | null = null
  if (previous?.iv && previous.ciphertext) {
    try {
      const bytes = await crypto.subtle.decrypt({ name: 'AES-GCM', iv: new Uint8Array(previous.iv) },
        key, new Uint8Array(previous.ciphertext))
      result = JSON.parse(new TextDecoder().decode(bytes))
    } catch {
      throw new Error('Could not unlock the saved plan with this password')
    }
  }
  if (generation !== unlockGeneration) throw new Error('The workspace changed while it was unlocking')
  activeUsername = username
  activeKey = key
  activeSalt = previous.salt
  activeRevision = previous.revision ?? 0
  return result
}

export async function savePrivatePlan(plan: unknown): Promise<void> {
  if (!activeUsername || !activeKey || !activeSalt || activeRevision === null) {
    throw new Error('Unlock your private workspace before training')
  }
  const username = activeUsername
  const key = activeKey
  const salt = activeSalt
  const revision = activeRevision
  const assertActive = () => {
    if (activeUsername !== username || activeKey !== key || activeSalt !== salt ||
        activeRevision !== revision) {
      throw new Error('The workspace changed during training')
    }
  }
  const previous = await readRecord(username)
  assertActive()
  if (!previous) throw new Error('Private workspace is not initialized')
  if (previous.salt.length !== salt.length ||
      previous.salt.some((byte, index) => byte !== salt[index])) {
    throw new Error('This workspace changed in another tab. Reload and unlock it before saving again.')
  }
  const iv = crypto.getRandomValues(new Uint8Array(12))
  const ciphertext = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key,
    new TextEncoder().encode(JSON.stringify(plan)))
  assertActive()
  await writeIfCurrent({ username, salt: previous.salt, revision: revision + 1,
    iv: Array.from(iv), ciphertext: Array.from(new Uint8Array(ciphertext)) }, revision)
  assertActive()
  activeRevision = revision + 1
}

export function privateStoreUnlocked(username: string): boolean {
  return !!activeKey && activeUsername === username
}
export function lockPrivateStore(): void {
  unlockGeneration += 1
  activeUsername = null
  activeKey = null
  activeSalt = null
  activeRevision = null
}
