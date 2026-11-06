import { useEffect, useState } from 'react'
import { getHealth } from './api.js'
import { Stepper } from './components/ui.jsx'
import MatchesPage from './pages/MatchesPage.jsx'
import ReportPage from './pages/ReportPage.jsx'
import UploadPage from './pages/UploadPage.jsx'

export default function App() {
  const [step, setStep] = useState(0)       // 0 upload · 1 report · 2 matches
  const [result, setResult] = useState(null) // { profile, analysis } from /api/analyze
  const [health, setHealth] = useState(null)

  useEffect(() => { getHealth().then(setHealth).catch(() => setHealth(null)) }, [])

  const maxStep = result ? 2 : 0
  const analysed = (r) => { setResult(r); setStep(1); window.scrollTo(0, 0) }

  return (
    <div className="min-h-screen">
      <header className="border-b border-slate-200 bg-white/80 backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-4 py-3">
          <button className="flex items-center gap-2 font-bold" onClick={() => setStep(0)}>
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-600 text-white" aria-hidden>in</span>
            Profile Optimizer
          </button>
          <Stepper step={step} maxStep={maxStep} go={setStep} />
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-4 py-8">
        {step === 0 && <UploadPage onAnalyzed={analysed} health={health} />}
        {step === 1 && result && <ReportPage result={result} onNext={() => { setStep(2); window.scrollTo(0, 0) }} />}
        {step === 2 && result && <MatchesPage profile={result.profile} />}
      </main>
      <footer className="pb-8 text-center text-xs text-slate-400">
        Built for an LMTA course project · scores are heuristic guidance, not a guarantee of recruiter behaviour
      </footer>
    </div>
  )
}
