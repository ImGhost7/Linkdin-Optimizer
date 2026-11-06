import { useState } from 'react'

/** score (0-100) -> colour family, used consistently for rings, bars and badges */
export const tone = (s) =>
  s >= 70 ? { bar: 'bg-emerald-500', text: 'text-emerald-600', ring: '#10b981' }
  : s >= 50 ? { bar: 'bg-amber-500', text: 'text-amber-600', ring: '#f59e0b' }
  : { bar: 'bg-rose-500', text: 'text-rose-600', ring: '#f43f5e' }

export function ScoreRing({ score, grade, size = 150 }) {
  const r = 52, c = 2 * Math.PI * r
  return (
    <div className="relative" style={{ width: size, height: size }} role="img" aria-label={`Overall score ${score} out of 100, grade ${grade}`}>
      <svg viewBox="0 0 120 120" className="h-full w-full -rotate-90">
        <circle cx="60" cy="60" r={r} fill="none" stroke="#e2e8f0" strokeWidth="10" />
        <circle cx="60" cy="60" r={r} fill="none" stroke={tone(score).ring} strokeWidth="10" strokeLinecap="round"
          strokeDasharray={c} strokeDashoffset={c * (1 - score / 100)} style={{ transition: 'stroke-dashoffset .8s ease' }} />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-4xl font-bold tabular-nums">{Math.round(score)}</span>
        <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Grade {grade}</span>
      </div>
    </div>
  )
}

export function Bar({ value, label, detail, height = 'h-2.5' }) {
  return (
    <div>
      {label && (
        <div className="mb-1 flex items-baseline justify-between text-sm">
          <span className="font-medium text-slate-700">{label}</span>
          <span className={`font-semibold tabular-nums ${tone(value).text}`}>{Math.round(value)}</span>
        </div>
      )}
      <div className={`${height} w-full overflow-hidden rounded-full bg-slate-100`}>
        <div className={`${height} rounded-full ${tone(value).bar}`} style={{ width: `${Math.max(2, value)}%`, transition: 'width .6s ease' }} />
      </div>
      {detail && <p className="mt-1 text-xs text-slate-500">{detail}</p>}
    </div>
  )
}

const CHIP = {
  neutral: 'bg-slate-100 text-slate-700',
  good: 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200',
  bad: 'bg-rose-50 text-rose-700 ring-1 ring-rose-200',
  warn: 'bg-amber-50 text-amber-800 ring-1 ring-amber-200',
}
export const Chip = ({ children, kind = 'neutral', title }) => (
  <span title={title} className={`inline-block rounded-full px-2.5 py-1 text-xs font-medium ${CHIP[kind]}`}>{children}</span>
)

export function CopyButton({ text, label = 'Copy' }) {
  const [done, setDone] = useState(false)
  const copy = async () => {
    try { await navigator.clipboard.writeText(text); setDone(true); setTimeout(() => setDone(false), 1500) } catch { /* clipboard blocked */ }
  }
  return <button type="button" onClick={copy} className="btn-ghost !px-3 !py-1.5 text-xs">{done ? 'Copied ✓' : label}</button>
}

export function Spinner({ label }) {
  return (
    <div className="flex items-center gap-3 text-sm text-slate-500" role="status">
      <span className="h-5 w-5 animate-spin rounded-full border-2 border-slate-300 border-t-indigo-600" />
      {label}
    </div>
  )
}

export const ErrorBox = ({ children }) =>
  children ? <div role="alert" className="rounded-xl border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-700">{children}</div> : null

const STEPS = ['Upload', 'Profile report', 'Job matches']
export function Stepper({ step, maxStep, go }) {
  return (
    <nav aria-label="Progress" className="flex items-center gap-2">
      {STEPS.map((name, i) => {
        const enabled = i <= maxStep
        return (
          <div key={name} className="flex items-center gap-2">
            <button type="button" disabled={!enabled} onClick={() => go(i)} aria-current={i === step ? 'step' : undefined}
              className={`flex items-center gap-2 rounded-full px-3 py-1.5 text-sm font-medium transition
                ${i === step ? 'bg-indigo-600 text-white' : enabled ? 'bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50' : 'text-slate-400'}`}>
              <span className={`flex h-5 w-5 items-center justify-center rounded-full text-xs ${i === step ? 'bg-white/25' : 'bg-slate-100'}`}>{i + 1}</span>
              <span className="hidden sm:inline">{name}</span>
            </button>
            {i < STEPS.length - 1 && <span className="h-px w-4 bg-slate-300 sm:w-8" />}
          </div>
        )
      })}
    </nav>
  )
}
