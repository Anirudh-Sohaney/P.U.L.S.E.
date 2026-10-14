import { useEffect, useState } from 'react'
import { Pause, Play } from 'lucide-react'
import '../pharmacy-scene.css'

const scenes = [
  { title: 'A normal day at the pharmacy.', detail: 'Medicines leave the shelf as your community needs them.', label: 'Stock on hand' },
  { title: 'An empty shelf changes the day.', detail: 'A stock gap means it is time to review the next order.', label: 'Stock gap' },
  { title: 'The next delivery makes a difference.', detail: 'Check what is arriving, then plan for what is still needed.', label: 'Delivery arrives' },
]

function Pack({ x, y, bottle = false, n }: { x: number; y: number; bottle?: boolean; n: number }) {
  return <g className={`pharmacy-pack pack-${n}`} style={{ transformOrigin: `${x + 11}px ${y + 25}px` }}>
    {bottle ? <><rect x={x+2} y={y} width="18" height="7" rx="2" className="bottle-cap" /><rect x={x} y={y+7} width="22" height="36" rx="5" className="bottle-body" /><rect x={x+1} y={y+19} width="20" height="15" rx="1" className="pack-label" /><path d={`M${x+6} ${y+23}h10m-10 4h6`} className="pack-line" /></> : <><rect x={x} y={y} width="24" height="43" rx="2" className="carton-body" /><path d={`M${x} ${y+8}h24`} className="carton-band" /><path d={`M${x+5} ${y+19}h13m-13 5h9m-9 9h7`} className="pack-line" /></>}
  </g>
}

export default function PharmacyScene() {
  const [phase, setPhase] = useState(0)
  const [playing, setPlaying] = useState(true)
  const [reduced, setReduced] = useState(false)
  const [held, setHeld] = useState(false)
  useEffect(() => {
    const media = matchMedia('(prefers-reduced-motion: reduce)')
    const update = () => setReduced(media.matches)
    update(); media.addEventListener('change', update)
    return () => media.removeEventListener('change', update)
  }, [])
  useEffect(() => {
    if (!playing || reduced || held) return
    const timer = window.setInterval(() => setPhase(value => (value + 1) % 3), 4200)
    return () => window.clearInterval(timer)
  }, [playing, reduced, held])
  return <figure className={`pharmacy-scene phase-${phase}`} aria-label="Illustrated pharmacy stock cycle" onMouseEnter={() => setHeld(true)} onMouseLeave={() => setHeld(false)} onFocusCapture={() => setHeld(true)} onBlurCapture={event => { if (!event.currentTarget.contains(event.relatedTarget)) setHeld(false) }}>
    <div className="pharmacy-scene-label"><span>Inside a community pharmacy</span><span>Illustrative animation</span></div>
    <div className="pharmacy-stage">
    <svg viewBox="0 0 1120 350" role="img" aria-label={`Pharmacy illustration: ${scenes[phase].label}.`}>
      <rect width="1120" height="350" className="scene-wall" />
      <path d="M0 306H1120V350H0Z" className="scene-floor" />
      <path d="M0 306H1120" className="scene-rule" />
      <g className="pharmacy-window">
        <rect x="45" y="28" width="168" height="213" rx="72" className="window-frame" />
        <path d="M56 236V112a73 73 0 0 1 146 0v124Z" className="window-glass" />
        <path d="M57 195q44-68 87-19t58-8v68H57Z" className="window-landscape" />
        <path d="M129 39V240M53 140H205" className="window-mullion" />
        <path d="M40 242H217" className="window-sill" />
      </g>
      <g className="pharmacy-sign">
        <rect x="271" y="28" width="110" height="40" rx="5" className="sign-paper" />
        <path d="M289 39v17m-8-8h16" className="sign-cross" />
        <text x="306" y="53">pharmacy</text>
      </g>
      <g className="pharmacy-cabinet">
        <rect x="433" y="27" width="633" height="210" rx="5" className="cabinet-back" />
        {[93, 160, 227].map(y => <g key={y}><path d={`M438 ${y}H1061`} className="shelf-edge" /><path d={`M446 ${y+7}H1052`} className="shelf-shadow" /></g>)}
        <path d="M642 32V227M850 32V227" className="cabinet-divider" />
        {[0, 1, 2].map(row => <g key={row}>{Array.from({length:18}, (_, col) => <Pack key={col} x={453+col*33} y={48+row*67} bottle={(row+col)%3===0} n={(col+row*5)%8} />)}</g>)}
        <g className="stock-alert"><rect x="664" y="114" width="154" height="34" rx="17" /><circle cx="682" cy="131" r="7" /><text x="682" y="135" textAnchor="middle" className="alert-mark">!</text><text x="699" y="135">Stock gap</text></g>
      </g>
      <g className="counter">
        <rect x="265" y="245" width="661" height="92" rx="4" className="counter-front" />
        <path d="M281 260H909M492 260V329M710 260V329" className="counter-detail" />
        <rect x="251" y="236" width="689" height="15" rx="4" className="counter-top" />
        <path d="M306 282h22m205 0h22m205 0h22" className="counter-handle" />
        <rect x="311" y="180" width="86" height="49" rx="4" className="screen-frame" />
        <rect x="317" y="186" width="74" height="36" rx="1" className="screen-glass" />
        <path d="M337 236h38m-19-8v8" className="screen-stand" />
        <path d="M329 204h12l6-8 7 16 6-9h18" className="screen-pulse" />
        <g className="pharmacy-bag"><path d="M553 191h47l6 45h-59Z" className="bag-paper" /><path d="M565 193v-8a11 11 0 0 1 22 0v8" className="bag-handle" /><path d="M577 204v19m-9-10h18" className="bag-cross" /></g>
      </g>
      <g className="pharmacy-delivery"><path d="M770 179h73l23 15v42h-96Z" className="delivery-front" /><path d="m770 179 23-14h73v29h-23Z" className="delivery-top" /><path d="M843 194v42m-42-63 24 15v19" className="delivery-tape" /><rect x="783" y="204" width="34" height="18" rx="2" className="delivery-label" /><path d="M791 213h18m-12-4-5 4 5 4" className="pack-line" /></g>
      <g className="pharmacy-plant"><path d="M108 301v-36q-45-9-35-37 33-1 35 37 0-56 31-54 10 32-31 54 39-33 49-10-11 28-49 29" className="plant-leaves" /><path d="M85 292h49l-7 43H92Z" className="plant-pot" /></g>
      <text x="997" y="326" className="scene-wordmark">pulse.</text>
    </svg>
    </div>
    <figcaption className="pharmacy-caption">
      <div><h2>{scenes[phase].title}</h2><p>{scenes[phase].detail}</p></div>
      <div className="pharmacy-scene-controls"><div role="group" aria-label="Pharmacy animation stages">{scenes.map((scene, index) => <button key={scene.label} type="button" aria-label={scene.label} aria-pressed={phase === index} onClick={() => {setPhase(index); setPlaying(false)}}><span />{scene.label}</button>)}</div>{!reduced && <button className="pharmacy-play" type="button" aria-label={playing ? 'Pause pharmacy animation' : 'Play pharmacy animation'} onClick={() => setPlaying(value => !value)}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>}</div>
    </figcaption>
  </figure>
}
