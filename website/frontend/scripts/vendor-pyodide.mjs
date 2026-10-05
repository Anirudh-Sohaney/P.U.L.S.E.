import { createHash } from 'node:crypto'
import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

const version = '0.29.4'
const packageRoot = join(process.cwd(), 'node_modules', 'pyodide')
const outputRoot = join(process.cwd(), 'public', 'pyodide', version)
const lock = JSON.parse(await readFile(join(packageRoot, 'pyodide-lock.json'), 'utf8'))
const wanted = ['numpy', 'pandas', 'scikit-learn', 'xgboost']
const required = new Set()

function addPackage(name) {
  if (required.has(name)) return
  const item = lock.packages[name]
  if (!item?.file_name || !item?.sha256) throw new Error(`Pyodide lock is missing ${name}`)
  required.add(name)
  for (const dependency of item.depends || []) addPackage(dependency)
}

for (const name of wanted) addPackage(name)
await mkdir(outputRoot, { recursive: true })

for (const name of ['pyodide.mjs', 'pyodide.asm.js', 'pyodide.asm.wasm',
                    'python_stdlib.zip', 'pyodide-lock.json']) {
  await copyFile(join(packageRoot, name), join(outputRoot, name))
}

for (const name of [...required].sort()) {
  const { file_name: filename, sha256 } = lock.packages[name]
  const destination = join(outputRoot, filename)
  let bytes
  try { bytes = await readFile(destination) } catch { /* Download below. */ }
  const digest = data => createHash('sha256').update(data).digest('hex')
  if (!bytes || digest(bytes) !== sha256) {
    const response = await fetch(`https://cdn.jsdelivr.net/pyodide/v${version}/full/${filename}`)
    if (!response.ok) throw new Error(`Could not download ${filename}: HTTP ${response.status}`)
    bytes = Buffer.from(await response.arrayBuffer())
    if (digest(bytes) !== sha256) throw new Error(`Pyodide package hash mismatch: ${filename}`)
    await writeFile(destination, bytes)
  }
  process.stdout.write(`Verified ${name}\n`)
}

process.stdout.write(`Vendored Pyodide ${version} with ${required.size} locked packages\n`)
