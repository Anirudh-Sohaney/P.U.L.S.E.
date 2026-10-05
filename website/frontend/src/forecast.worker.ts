type PrepareMessage = { type: 'prepare'; sales_csv: string; inventory_csv: string; signal_years: number; local_today: string }
type TrainMessage = { type: 'train'; signal_rows: unknown[] }

let runtime: any = null
let salesCsv = ''
let inventoryCsv = ''
let signalYears = 0
let localToday = ''

self.onmessage = async (event: MessageEvent<PrepareMessage | TrainMessage>) => {
  try {
    if (event.data.type === 'prepare') {
      self.postMessage({ type: 'progress', message: 'Loading the local XGBoost runtime' })
      const pyodideUrl = new URL('/pyodide/0.29.4/pyodide.mjs', self.location.origin).href
      const module = await import(/* @vite-ignore */ pyodideUrl)
      runtime = await module.loadPyodide({ indexURL: '/pyodide/0.29.4/' })
      await runtime.loadPackage(['numpy', 'pandas', 'scikit-learn', 'xgboost'])
      const sourceResponse = await fetch('/browser_forecast.py', { cache: 'no-store' })
      if (!sourceResponse.ok) throw new Error('Could not load the browser model code')
      await runtime.runPythonAsync(await sourceResponse.text())
      salesCsv = event.data.sales_csv
      inventoryCsv = event.data.inventory_csv
      signalYears = event.data.signal_years
      localToday = event.data.local_today
      runtime.globals.set('sales_csv', salesCsv)
      runtime.globals.set('inventory_csv', inventoryCsv)
      runtime.globals.set('signal_years', signalYears)
      runtime.globals.set('local_today', localToday)
      await runtime.runPythonAsync(`
import json
_sales = _read_sales(sales_csv, today=local_today)
_read_inventory(inventory_csv, _sales, today=local_today)
_end = _sales.date.max()
_start = _end - pd.DateOffset(years=signal_years)
prepared_window = json.dumps({"start_date": _start.date().isoformat(), "end_date": _end.date().isoformat()})
`)
      self.postMessage({ type: 'prepared', window: JSON.parse(runtime.globals.get('prepared_window') as string) })
      return
    }
    if (!runtime || !salesCsv) throw new Error('Prepare the private files before training')
    self.postMessage({ type: 'progress', message: 'Training per-drug models on this device' })
    runtime.globals.set('payload', JSON.stringify({
      sales_csv: salesCsv, inventory_csv: inventoryCsv,
      signal_rows: event.data.signal_rows, signal_years: signalYears,
      local_today: localToday,
    }))
    await runtime.runPythonAsync('result = train(payload)')
    self.postMessage({ type: 'success', result: JSON.parse(runtime.globals.get('result') as string) })
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) })
  }
}
