import { useCallback, useEffect, useState } from 'react'
import { match } from '../api.js'
import { Bar, Chip, ErrorBox, Spinner, tone } from '../components/ui.jsx'

const DEFAULTS = { embedding: 50, tfidf: 50, skills: 0 }
const SLIDERS = [['embedding', 'Semantic (embeddings)'], ['tfidf', 'Keywords (TF-IDF)'], ['skills', 'Skill overlap']]

function JobCard({ m }) {
  const [open, setOpen] = useState(false)
  const t = tone(m.score)
  return (
    <article className="card p-5">
      <div className="flex items-start gap-4">
        <div className={`flex h-14 w-14 shrink-0 flex-col items-center justify-center rounded-xl bg-slate-50 ring-1 ring-slate-200 ${t.text}`}>
          <span className="text-xl font-bold tabular-nums leading-none">{Math.round(m.score)}</span>
          <span className="text-[10px] font-medium uppercase text-slate-400">match</span>
        </div>
        <div className="min-w-0 flex-1">
          <h3 className="font-semibold leading-snug">{m.rank ? `${m.rank}. ` : ''}{m.title}</h3>
          <p className="text-sm text-slate-500">{[m.company, m.location].filter(Boolean).join(' · ')}</p>
        </div>
        {m.url && <a href={m.url} target="_blank" rel="noreferrer noopener" className="btn-ghost !px-3 !py-1.5 text-xs">View posting ↗</a>}
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4">
        <Bar value={m.components.embedding * 100} label={<span className="text-xs">Semantic</span>} height="h-1.5" />
        <Bar value={m.components.tfidf * 100} label={<span className="text-xs">Keywords</span>} height="h-1.5" />
        <Bar value={m.components.skills * 100} height="h-1.5"
          label={<span className="text-xs">Skills <span className="font-normal text-slate-400">{m.matched_skills.length}/{m.matched_skills.length + m.missing_skills.length}</span></span>} />
      </div>

      <div className="mt-4 grid gap-4 sm:grid-cols-2">
        <div>
          <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">You have ({m.matched_skills.length})</h4>
          <div className="flex flex-wrap gap-1.5">{m.matched_skills.length ? m.matched_skills.map((s) => <Chip key={s} kind="good">{s}</Chip>) : <span className="text-xs text-slate-400">No listed skills detected</span>}</div>
        </div>
        <div>
          <h4 className="mb-1.5 text-xs font-semibold uppercase tracking-wide text-slate-500">Gap ({m.missing_skills.length})</h4>
          <div className="flex flex-wrap gap-1.5">{m.missing_skills.length ? m.missing_skills.slice(0, 14).map((s) => <Chip key={s} kind="bad">{s}</Chip>) : <span className="text-xs text-emerald-700">No gaps detected 🎉</span>}</div>
        </div>
      </div>

      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="mt-4 text-sm font-medium text-indigo-600 hover:underline">
        {open ? 'Hide' : 'Show'} 3 tips to close the gap
      </button>
      {open && (
        <ol className="mt-2 list-decimal space-y-2 pl-5 text-sm">
          {m.tips.map((tip) => <li key={tip}>{tip}</li>)}
        </ol>
      )}
    </article>
  )
}

export default function MatchesPage({ profile }) {
  const [mode, setMode] = useState('dataset')
  const [jd, setJd] = useState('')
  const [jdTitle, setJdTitle] = useState('')
  const [w, setW] = useState(DEFAULTS)
  const [topK, setTopK] = useState(10)
  const [state, setState] = useState({ loading: false, error: '', data: null })

  const run = useCallback(async (weights = w) => {
    setState((s) => ({ ...s, loading: true, error: '' }))
    try {
      const body = { profile, top_k: topK, weights: { embedding: weights.embedding, tfidf: weights.tfidf, skills: weights.skills } }
      if (mode === 'jd') { body.job_description = jd; body.job_title = jdTitle || 'Pasted job description' }
      setState({ loading: false, error: '', data: await match(body) })
    } catch (e) { setState({ loading: false, error: e.message, data: null }) }
  }, [profile, topK, w, mode, jd, jdTitle])

  useEffect(() => { if (mode === 'dataset') run() }, [mode]) // eslint-disable-line react-hooks/exhaustive-deps

  const total = w.embedding + w.tfidf + w.skills || 1
  const canRun = mode === 'dataset' || jd.trim().length > 30

  return (
    <div className="grid gap-6 lg:grid-cols-[320px_1fr]">
      <aside className="space-y-5 lg:sticky lg:top-6 lg:self-start">
        <div className="card p-5">
          <div role="tablist" className="mb-4 grid grid-cols-2 rounded-xl bg-slate-100 p-1 text-sm font-medium">
            {[['dataset', 'Browse jobs'], ['jd', 'Paste a JD']].map(([k, label]) => (
              <button key={k} role="tab" aria-selected={mode === k} onClick={() => setMode(k)}
                className={`rounded-lg py-1.5 transition ${mode === k ? 'bg-white shadow-sm' : 'text-slate-600'}`}>{label}</button>
            ))}
          </div>
          {mode === 'jd' && (
            <div className="mb-4 space-y-2">
              <input value={jdTitle} onChange={(e) => setJdTitle(e.target.value)} placeholder="Job title (optional)" aria-label="Job title"
                className="w-full rounded-lg border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none" />
              <textarea value={jd} onChange={(e) => setJd(e.target.value)} rows={8} placeholder="Paste the full job description…" aria-label="Job description"
                className="w-full rounded-lg border border-slate-300 p-3 text-sm focus:border-indigo-500 focus:outline-none" />
            </div>
          )}
          <h3 className="mb-3 text-sm font-semibold">Hybrid weights</h3>
          <div className="space-y-4">
            {SLIDERS.map(([k, label]) => (
              <div key={k}>
                <label htmlFor={`w-${k}`} className="flex justify-between text-xs font-medium text-slate-600">
                  <span>{label}</span><span className="tabular-nums">{Math.round((w[k] / total) * 100)}%</span>
                </label>
                <input id={`w-${k}`} type="range" min="0" max="100" value={w[k]} onChange={(e) => setW({ ...w, [k]: +e.target.value })} className="w-full accent-indigo-600" />
              </div>
            ))}
          </div>
          <div className="mt-4 flex items-center justify-between text-sm">
            <label htmlFor="topk" className="text-slate-600">Show top</label>
            <select id="topk" value={topK} onChange={(e) => setTopK(+e.target.value)} className="rounded-lg border border-slate-300 px-2 py-1">
              {[5, 10, 20, 50].map((n) => <option key={n}>{n}</option>)}
            </select>
          </div>
          <button className="btn-primary mt-4 w-full" disabled={!canRun || state.loading} onClick={() => run()}>
            {mode === 'jd' ? 'Score this job' : 'Re-rank jobs'}
          </button>
          <button type="button" className="mt-2 w-full text-xs text-slate-500 hover:underline" onClick={() => setW(DEFAULTS)}>Reset weights</button>
        </div>
        <p className="px-1 text-xs text-slate-500">Scores are <b>relative</b>: each signal is min–max normalised over the {state.data?.pool_size?.toLocaleString() ?? '…'} bundled jobs, so 100 means “best available for this profile”, not a probability.</p>
      </aside>

      <section aria-live="polite" className="space-y-4">
        <ErrorBox>{state.error}</ErrorBox>
        {state.loading && <Spinner label="Ranking jobs… (the first request may take a few seconds)" />}
        {state.data && !state.loading && (
          <>
            <h2 className="text-lg font-semibold">{state.data.mode === 'dataset' ? `Top ${state.data.matches.length} matches` : 'Your job description'}</h2>
            {state.data.matches.map((m) => <JobCard key={m.job_id + m.rank} m={m} />)}
          </>
        )}
        {!state.data && !state.loading && !state.error && mode === 'jd' && <p className="text-sm text-slate-500">Paste a job description to see how well you match and what to close.</p>}
      </section>
    </div>
  )
}
