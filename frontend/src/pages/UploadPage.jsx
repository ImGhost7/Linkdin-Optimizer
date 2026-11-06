import { useRef, useState } from 'react'
import { analyze } from '../api.js'
import { ErrorBox, Spinner } from '../components/ui.jsx'
import { SAMPLE_PROFILE } from '../sample.js'

export default function UploadPage({ onAnalyzed, health }) {
  const [tab, setTab] = useState('pdf')
  const [file, setFile] = useState(null)
  const [text, setText] = useState('')
  const [extra, setExtra] = useState('')
  const [useLlm, setUseLlm] = useState(true)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [drag, setDrag] = useState(false)
  const input = useRef(null)

  const ready = tab === 'pdf' ? !!file : text.trim().length > 20

  const pick = (f) => {
    if (!f) return
    if (!f.name.toLowerCase().endsWith('.pdf')) return setError('Please choose a PDF file (LinkedIn → Me → View profile → More → Save to PDF).')
    setError(''); setFile(f)
  }

  const submit = async () => {
    setBusy(true); setError('')
    try {
      const fd = new FormData()
      if (tab === 'pdf') fd.append('file', file)
      else fd.append('text', text)
      fd.append('extra_skills', extra)
      fd.append('use_llm', String(useLlm))
      onAnalyzed(await analyze(fd))
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  return (
    <div className="mx-auto max-w-3xl">
      <div className="mb-8 text-center">
        <h1 className="text-3xl font-bold tracking-tight sm:text-4xl">Optimise your LinkedIn profile</h1>
        <p className="mt-3 text-slate-600">Get a section-by-section score, skill gaps, rewrite suggestions and ranked job matches. Your profile is processed in memory and never stored.</p>
      </div>

      <div className="card p-6">
        <div role="tablist" className="mb-5 inline-flex rounded-xl bg-slate-100 p-1 text-sm font-medium">
          {[['pdf', 'Upload PDF'], ['paste', 'Paste text']].map(([k, label]) => (
            <button key={k} role="tab" aria-selected={tab === k} onClick={() => setTab(k)}
              className={`rounded-lg px-4 py-1.5 transition ${tab === k ? 'bg-white shadow-sm' : 'text-slate-600 hover:text-slate-900'}`}>{label}</button>
          ))}
        </div>

        {tab === 'pdf' ? (
          <div
            onDragOver={(e) => { e.preventDefault(); setDrag(true) }} onDragLeave={() => setDrag(false)}
            onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]) }}
            onClick={() => input.current?.click()} role="button" tabIndex={0}
            onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && input.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-12 text-center transition
              ${drag ? 'border-indigo-500 bg-indigo-50' : 'border-slate-300 hover:border-indigo-400 hover:bg-slate-50'}`}>
            <input ref={input} type="file" accept="application/pdf,.pdf" className="hidden" onChange={(e) => pick(e.target.files[0])} />
            <div className="text-4xl" aria-hidden>📄</div>
            {file ? (
              <p className="mt-3 font-medium">{file.name} <span className="font-normal text-slate-500">({(file.size / 1024).toFixed(0)} KB)</span></p>
            ) : (
              <>
                <p className="mt-3 font-medium">Drop your LinkedIn PDF here, or click to browse</p>
                <p className="mt-1 text-sm text-slate-500">LinkedIn → Me → View profile → More → Save to PDF</p>
              </>
            )}
          </div>
        ) : (
          <div>
            <textarea value={text} onChange={(e) => setText(e.target.value)} rows={12} aria-label="Profile text"
              placeholder={'Paste your profile. Section headings help:\n\nHeadline: …\nAbout\n…\nExperience\nRole at Company (Jan 2022 - Present)\n…\nEducation\n…\nSkills\n…'}
              className="w-full rounded-xl border border-slate-300 p-3 font-mono text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
            <button type="button" className="mt-2 text-sm font-medium text-indigo-600 hover:underline" onClick={() => setText(SAMPLE_PROFILE)}>
              Use a sample (fictional) profile
            </button>
          </div>
        )}

        <div className="mt-5">
          <label htmlFor="extra" className="text-sm font-medium">Full skills list <span className="font-normal text-slate-500">(optional)</span></label>
          <input id="extra" value={extra} onChange={(e) => setExtra(e.target.value)} placeholder="Python, SQL, Docker, …"
            className="mt-1 w-full rounded-xl border border-slate-300 px-3 py-2 text-sm focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-200" />
          <p className="mt-1 text-xs text-slate-500">LinkedIn’s PDF export only contains your <b>top 3</b> skills. Paste the rest here for an accurate skills score.</p>
        </div>

        <label className="mt-4 flex items-start gap-2 text-sm">
          <input type="checkbox" className="mt-0.5" checked={useLlm} onChange={(e) => setUseLlm(e.target.checked)} />
          <span>
            Use Gemini for rewrite suggestions
            <span className="block text-xs text-slate-500">
              {health?.gemini_configured
                ? 'API key found. Only your headline, about, recent roles and skills are sent — never your name or contact details.'
                : 'No GEMINI_API_KEY configured — rule-based suggestions will be used (works fully offline).'}
            </span>
          </span>
        </label>

        <div className="mt-6 space-y-3">
          <ErrorBox>{error}</ErrorBox>
          <div className="flex items-center gap-4">
            <button className="btn-primary" disabled={!ready || busy} onClick={submit}>Analyse my profile →</button>
            {busy && <Spinner label="Parsing and scoring…" />}
          </div>
        </div>
      </div>
      <p className="mt-4 text-center text-xs text-slate-500">We never scrape LinkedIn — you provide your own export, which keeps this within LinkedIn’s Terms of Service.</p>
    </div>
  )
}
