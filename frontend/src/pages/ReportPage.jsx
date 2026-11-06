import { useState } from 'react'
import { Bar, Chip, CopyButton, ScoreRing } from '../components/ui.jsx'

const LABELS = { headline: 'Headline', about: 'About', experience: 'Experience', education: 'Education', skills: 'Skills', certifications: 'Certifications' }
const METRIC = { completeness: 'Completeness', length: 'Length', action_verbs: 'Action verbs', quantified: 'Quantified impact', keywords: 'Keywords',
  readability: 'Readability', clarity: 'Clarity', detail: 'Detail', count: 'Count', recognised: 'Standard skill names', coverage: 'Coverage of your experience', diversity: 'Diversity' }

function SectionCard({ s }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="card p-4">
      <button type="button" className="w-full text-left" aria-expanded={open} onClick={() => setOpen(!open)}>
        <Bar value={s.score} label={<span>{LABELS[s.section]} <span className="ml-1 text-xs font-normal text-slate-400">{Math.round(s.weight * 100)}% of total{!s.present && ' · missing'}</span></span>} height="h-3" />
        <span className="mt-2 inline-block text-xs font-medium text-indigo-600">{open ? 'Hide breakdown' : 'Show breakdown'}</span>
      </button>
      {open && (
        <div className="mt-3 space-y-4 border-t border-slate-100 pt-4">
          {s.metrics.map((m) => (
            <div key={m.name}>
              <Bar value={m.score} height="h-1.5" label={<span className="text-xs">{METRIC[m.name] ?? m.name} <span className="text-slate-400">· weight {Math.round(m.weight * 100)}%</span></span>} detail={m.detail} />
              {m.tip && <p className="mt-1 rounded-lg bg-amber-50 px-2.5 py-1.5 text-xs text-amber-900">💡 {m.tip}</p>}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function ReportPage({ result, onNext }) {
  const { profile, analysis: a } = result
  const [skillsShown, setSkillsShown] = useState(false)
  const missing = a.skills.missing_from_skills_section
  const rw = a.rewrites

  return (
    <div className="space-y-6">
      {/* summary */}
      <div className="card flex flex-col items-center gap-6 p-6 sm:flex-row">
        <ScoreRing score={a.overall_score} grade={a.grade} />
        <div className="flex-1">
          <h2 className="text-xl font-bold">{profile.name || 'Your profile'}</h2>
          {profile.headline && <p className="text-slate-600">{profile.headline}</p>}
          <h3 className="mt-4 text-sm font-semibold uppercase tracking-wide text-slate-500">Biggest wins available</h3>
          <ul className="mt-2 space-y-1.5 text-sm">
            {a.top_tips.map((t) => <li key={t} className="flex gap-2"><span aria-hidden>→</span><span>{t}</span></li>)}
          </ul>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* section scores */}
        <section aria-labelledby="sec">
          <h2 id="sec" className="mb-3 text-lg font-semibold">Section scores</h2>
          <div className="space-y-3">{a.sections.map((s) => <SectionCard key={s.section} s={s} />)}</div>
        </section>

        <div className="space-y-6">
          {/* skills */}
          <section className="card p-5" aria-labelledby="sk">
            <h2 id="sk" className="text-lg font-semibold">Skills</h2>
            <h3 className="mt-3 text-sm font-medium text-slate-700">Mentioned in your text but missing from your Skills section</h3>
            {missing.length ? (
              <>
                <div className="mt-2 flex flex-wrap gap-2">
                  {missing.map((h) => <Chip key={h.skill} kind="warn" title={`Mentioned ${h.count}× in ${h.source}`}>+ {h.skill}</Chip>)}
                </div>
                <div className="mt-3"><CopyButton text={missing.map((h) => h.skill).join(', ')} label="Copy list to paste into LinkedIn" /></div>
              </>
            ) : <p className="mt-2 text-sm text-emerald-700">✓ Everything you mention is already listed.</p>}
            <h3 className="mt-5 text-sm font-medium text-slate-700">Listed ({a.skills.listed.length})</h3>
            <div className="mt-2 flex flex-wrap gap-2">
              {a.skills.listed.length ? a.skills.listed.map((s) => <Chip key={s} kind={a.skills.listed_recognised.length && a.skills.listed_recognised.some((r) => r.toLowerCase() === s.toLowerCase()) ? 'good' : 'neutral'}>{s}</Chip>)
                : <span className="text-sm text-slate-500">None found.</span>}
            </div>
            <button type="button" className="mt-4 text-xs font-medium text-indigo-600 hover:underline" onClick={() => setSkillsShown(!skillsShown)}>
              {skillsShown ? 'Hide' : 'Show'} all {a.skills.mentioned.length} skills detected in your text
            </button>
            {skillsShown && <div className="mt-2 flex flex-wrap gap-2">{a.skills.mentioned.map((h) => <Chip key={h.skill} title={h.category.replaceAll('_', ' ')}>{h.skill}{h.count > 1 ? ` ×${h.count}` : ''}</Chip>)}</div>}
          </section>

          {/* rewrites */}
          <section className="card p-5" aria-labelledby="rw">
            <div className="flex items-center justify-between">
              <h2 id="rw" className="text-lg font-semibold">Rewrite suggestions</h2>
              <Chip kind={rw.source === 'gemini' ? 'good' : 'neutral'}>{rw.source === 'gemini' ? '✨ Gemini' : 'Rule-based'}</Chip>
            </div>
            <p className="mt-1 text-xs text-slate-500">{rw.note}</p>
            <h3 className="mt-4 text-sm font-medium">Headline options</h3>
            <ul className="mt-2 space-y-2">
              {rw.headline_options.map((h) => (
                <li key={h} className="flex items-start justify-between gap-3 rounded-xl bg-slate-50 p-3 text-sm">
                  <span>{h}</span><CopyButton text={h} />
                </li>
              ))}
            </ul>
            <div className="mt-4 flex items-center justify-between">
              <h3 className="text-sm font-medium">About section</h3><CopyButton text={rw.about} />
            </div>
            <pre className="mt-2 whitespace-pre-wrap rounded-xl bg-slate-50 p-3 font-sans text-sm leading-relaxed">{rw.about}</pre>
          </section>
        </div>
      </div>

      <div className="flex justify-end"><button className="btn-primary" onClick={onNext}>Find matching jobs →</button></div>
    </div>
  )
}
