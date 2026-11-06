// Thin fetch wrappers. Every function throws an Error whose message is safe to show to the user.

async function handle(res) {
  if (res.ok) return res.json()
  let detail = `Request failed (${res.status})`
  try {
    const body = await res.json()
    if (typeof body.detail === 'string') detail = body.detail
    else if (Array.isArray(body.detail)) detail = body.detail.map((d) => d.msg).join('; ')
  } catch { /* non-JSON error body */ }
  throw new Error(detail)
}

const network = (e) => {
  throw e instanceof TypeError ? new Error('Cannot reach the API. Is the backend running on port 8000?') : e
}

export const getHealth = () => fetch('/api/health').then(handle).catch(network)

/** formData: file | text, extra_skills, use_llm  ->  { profile, analysis } */
export const analyze = (formData) =>
  fetch('/api/analyze', { method: 'POST', body: formData }).then(handle).catch(network)

/** body: { profile, job_description?, job_title?, top_k?, weights? }  ->  { mode, weights, pool_size, matches } */
export const match = (body) =>
  fetch('/api/match', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }).then(handle).catch(network)
