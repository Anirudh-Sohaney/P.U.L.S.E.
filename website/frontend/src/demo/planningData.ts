// Sample data for the public homepage. Every figure here is illustrative and is
// labelled as sample data wherever it appears. The arithmetic mirrors the
// browser planner's `_plan_quantities` rules in public/browser_forecast.py:
// incoming stock is credited only on days after its dated arrival, the minimum
// purchase covers the largest cumulative gap, and the suggested purchase also
// considers a safety buffer derived from recent variation.

export const PLAN_AS_OF = '2026-10-11'
export const HORIZON = 14

export type SampleMedicine = {
  id: string
  name: string
  strength: string
  form: string
  bin: string
  onHand: number
  demand: number[]
  incoming: { units: number; arrival: string } | null
  trailingMean: number
  trailingStd: number
}

export type DayProjection = {
  day: number
  date: string
  demand: number
  remaining: number
  shortfall: number
  credited: boolean
}

export type MedicinePlan = SampleMedicine & {
  days: DayProjection[]
  demand14: number
  stockoutDay: number | null
  arrivalDay: number | null
  creditedFromDay: number | null
  minimum7: number
  minimum14: number
  safety: number
  targetStock: number
  suggested: number
  status: 'reorder_now' | 'reorder_soon' | 'covered'
  unitsPerBox: number
}

const MEDICINES: SampleMedicine[] = [
  {
    id: 'amox', name: 'Amoxicillin', strength: '500 mg', form: 'capsules', bin: 'A-03',
    onHand: 132,
    demand: [16, 15, 17, 18, 16, 12, 10, 17, 18, 19, 17, 16, 12, 11],
    incoming: { units: 120, arrival: '2026-10-21' },
    trailingMean: 15.5, trailingStd: 3,
  },
  {
    id: 'salb', name: 'Salbutamol', strength: '100 mcg', form: 'inhalers', bin: 'R-11',
    onHand: 15,
    demand: [3, 2, 3, 4, 3, 2, 1, 3, 3, 4, 3, 3, 2, 2],
    incoming: { units: 24, arrival: '2026-10-24' },
    trailingMean: 2.8, trailingStd: 1,
  },
  {
    id: 'levo', name: 'Levothyroxine', strength: '50 mcg', form: 'tablets', bin: 'T-07',
    onHand: 128,
    demand: [10, 9, 9, 10, 9, 7, 6, 10, 9, 10, 9, 9, 7, 6],
    incoming: null,
    trailingMean: 8.9, trailingStd: 1.3,
  },
  {
    id: 'metf', name: 'Metformin', strength: '500 mg', form: 'tablets', bin: 'M-02',
    onHand: 1260,
    demand: [72, 68, 75, 70, 66, 48, 40, 74, 71, 73, 69, 67, 50, 42],
    incoming: null,
    trailingMean: 63, trailingStd: 12,
  },
]

const DAY_MS = 86_400_000

export function addDays(iso: string, days: number): string {
  const time = Date.parse(`${iso}T00:00:00Z`) + days * DAY_MS
  return new Date(time).toISOString().slice(0, 10)
}

const shortDate = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric', timeZone: 'UTC' })
const weekday = new Intl.DateTimeFormat('en-US', { weekday: 'short', timeZone: 'UTC' })
const longDate = new Intl.DateTimeFormat('en-US', { month: 'long', day: 'numeric', year: 'numeric', timeZone: 'UTC' })

export const fmtShort = (iso: string) => shortDate.format(new Date(`${iso}T00:00:00Z`))
export const fmtWeekday = (iso: string) => weekday.format(new Date(`${iso}T00:00:00Z`))
export const fmtLong = (iso: string) => longDate.format(new Date(`${iso}T00:00:00Z`))
export const fmtUnits = (value: number) => new Intl.NumberFormat('en-US').format(value)

function buildPlan(med: SampleMedicine): MedicinePlan {
  const dates = Array.from({ length: HORIZON }, (_, index) => addDays(PLAN_AS_OF, index + 1))
  let cumulative = 0
  const days: DayProjection[] = []
  let maxDeficit = 0
  let maxDeficit7 = 0
  let stockoutDay: number | null = null
  let creditedFromDay: number | null = null
  let arrivalDay: number | null = null
  let lastInbound = 0
  dates.forEach((date, index) => {
    cumulative += med.demand[index]
    const credited = Boolean(med.incoming && date > med.incoming.arrival)
    if (med.incoming && date === med.incoming.arrival) arrivalDay = index + 1
    if (credited && creditedFromDay === null) creditedFromDay = index + 1
    const inbound = credited && med.incoming ? med.incoming.units : 0
    lastInbound = inbound
    const deficit = Math.max(cumulative - med.onHand - inbound, 0)
    if (deficit > 0 && stockoutDay === null) stockoutDay = index + 1
    maxDeficit = Math.max(maxDeficit, deficit)
    if (index < 7) maxDeficit7 = Math.max(maxDeficit7, deficit)
    days.push({
      day: index + 1, date, demand: med.demand[index],
      remaining: Math.max(med.onHand + inbound - cumulative, 0),
      shortfall: deficit, credited,
    })
  })
  const demand14 = cumulative
  const minimum14 = Math.ceil(maxDeficit)
  const minimum7 = Math.ceil(maxDeficit7)
  const safety = Math.ceil(1.65 * med.trailingStd * Math.sqrt(7))
  const targetStock = Math.ceil(Math.max(demand14, med.trailingMean * 14) + safety)
  const suggested = Math.max(minimum14, Math.ceil(targetStock - med.onHand - lastInbound), 0)
  const status = minimum7 > 0 ? 'reorder_now' : suggested > 0 ? 'reorder_soon' : 'covered'
  const peak = Math.max(med.onHand, ...days.map(day => day.remaining))
  const unitsPerBox = niceBox(peak / 6)
  return { ...med, days, demand14, stockoutDay, arrivalDay, creditedFromDay, minimum7, minimum14, safety, targetStock, suggested, status, unitsPerBox }
}

function niceBox(raw: number): number {
  const steps = [3, 5, 10, 12, 20, 24, 25, 30, 50, 60, 100, 120, 200, 210, 250]
  return steps.find(step => step >= raw) ?? Math.ceil(raw)
}

export const SAMPLE_PLANS: MedicinePlan[] = MEDICINES.map(buildPlan)

export const STATUS_LABEL: Record<MedicinePlan['status'], string> = {
  reorder_now: 'Order now',
  reorder_soon: 'Order soon',
  covered: 'Covered',
}

export const HORIZON_DATES = Array.from({ length: HORIZON }, (_, index) => addDays(PLAN_AS_OF, index + 1))

/** Boxes shown in one compartment for a given unit count (max 6 per compartment). */
export function boxesFor(units: number, unitsPerBox: number): number {
  if (units <= 0) return 0
  return Math.min(6, Math.ceil(units / unitsPerBox))
}

export function explain(plan: MedicinePlan): string[] {
  const lines: string[] = []
  const name = `${plan.name} ${plan.strength}`
  lines.push(`${fmtUnits(plan.onHand)} on hand against ${fmtUnits(plan.demand14)} units of projected demand over 14 days.`)
  if (plan.incoming && plan.creditedFromDay) {
    lines.push(`${fmtUnits(plan.incoming.units)} incoming, dated ${fmtShort(plan.incoming.arrival)}. PULSE counts it from ${fmtShort(plan.days[plan.creditedFromDay - 1].date)}, the day after arrival.`)
  }
  if (plan.stockoutDay) {
    const day = plan.days[plan.stockoutDay - 1]
    lines.push(`Projected to run out on day ${plan.stockoutDay} (${fmtShort(day.date)})${plan.incoming ? ', before the delivery is counted' : ''}. The largest gap is ${fmtUnits(plan.minimum14)} units.`)
  } else {
    lines.push(`Projected stock lasts the full 14 days, ending near ${fmtUnits(plan.days[HORIZON - 1].remaining)} units.`)
  }
  if (plan.suggested > 0 && plan.suggested === plan.minimum14) {
    lines.push(`Suggested purchase: ${fmtUnits(plan.suggested)} units, enough to close that gap. Review it against pack sizes and supplier timing.`)
  } else if (plan.suggested > 0) {
    lines.push(`Suggested purchase: ${fmtUnits(plan.suggested)} units, a small top-up to hold a safety buffer of ${fmtUnits(plan.safety)} units based on recent variation.`)
  } else {
    lines.push(`No purchase suggested for ${name} in this window.`)
  }
  return lines
}
