#!/usr/bin/env python3
"""Build the self-contained static demo page from demo_artifacts/.

Reads the precomputed (synthetic-only) JSON in demo_artifacts/ and emits:
  demo_artifacts/index.html       full standalone document (deploy anywhere static)
  demo_artifacts/index.body.html  body content only (for embedding / the Artifact tool)

Narrative: a small platform to ask (1) what does fine-tuning improve? and
(2) does the base model choice change how well it adapts? The hero is a
before -> after transformation for one base model on one synthetic patient.
No network, no model, no live inference. All data inlined. Read-only viewer.
"""
from __future__ import annotations

import json
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "demo_artifacts"
CONDITIONS = ["gemma3_base", "gemma3_qlora", "medgemma_base", "medgemma_qlora"]
# order for the aggregate table: the two bases first, then the two fine-tuned.
AGG_ORDER = ["gemma3_base", "medgemma_base", "gemma3_qlora", "medgemma_qlora", "gold_ceiling"]
COND_LABEL = {
    "gemma3_base": "Gemma 3 · base",
    "gemma3_qlora": "Gemma 3 · fine-tuned",
    "medgemma_base": "MedGemma · base",
    "medgemma_qlora": "MedGemma · fine-tuned",
    "gold_ceiling": "Gold (ceiling)",
}
MODEL_LABEL = {"gemma3": "Gemma 3", "medgemma": "MedGemma (medical)"}
DIMS = ["schema_conformance", "disclaimer_present", "non_diagnostic", "numeric_grounding"]
DIM_LABEL = {"schema_conformance": "schema", "disclaimer_present": "disclaimer",
             "non_diagnostic": "non-diagnostic", "numeric_grounding": "grounding"}


def load():
    return {
        "cases": json.loads((DEMO / "synthetic_cases.json").read_text()),
        "outputs": json.loads((DEMO / "model_outputs.json").read_text()),
        "eval": json.loads((DEMO / "evaluation_summary.json").read_text()),
        "failures": json.loads((DEMO / "selected_failure_cases.json").read_text()),
    }


CSS = r"""
:root{
  --paper:#eef2f3; --surface:#ffffff; --surface-2:#f5f8f9; --ink:#151d22;
  --muted:#5b6b73; --line:#d6e0e3; --accent:#0b5450; --accent-2:#0f8a80;
  --good:#2f8f5b; --warn:#b7791b; --crit:#c0453b;
  --mark:#fbe6b0; --mark-ink:#5a4406;
}
@media (prefers-color-scheme:dark){
  :root{ --paper:#0d1316; --surface:#141c21; --surface-2:#0f171b; --ink:#dde7ea;
    --muted:#8ba0a9; --line:#26333b; --accent:#39b3a8; --accent-2:#4fd0c4;
    --good:#57bd83; --warn:#d9a441; --crit:#e0685d; --mark:#5a4a12; --mark-ink:#ffe9ab; }
}
:root[data-theme="light"]{ --paper:#eef2f3; --surface:#ffffff; --surface-2:#f5f8f9; --ink:#151d22;
  --muted:#5b6b73; --line:#d6e0e3; --accent:#0b5450; --accent-2:#0f8a80;
  --good:#2f8f5b; --warn:#b7791b; --crit:#c0453b; --mark:#fbe6b0; --mark-ink:#5a4406; }
:root[data-theme="dark"]{ --paper:#0d1316; --surface:#141c21; --surface-2:#0f171b; --ink:#dde7ea;
  --muted:#8ba0a9; --line:#26333b; --accent:#39b3a8; --accent-2:#4fd0c4;
  --good:#57bd83; --warn:#d9a441; --crit:#e0685d; --mark:#5a4a12; --mark-ink:#ffe9ab; }
*{box-sizing:border-box}
.phml{
  --font-head:Georgia,"Iowan Old Style","Palatino Linotype",serif;
  --font-body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --font-mono:ui-monospace,"SFMono-Regular","Cascadia Code",Menlo,monospace;
  background:var(--paper); color:var(--ink); font-family:var(--font-body);
  line-height:1.5; margin:0; padding:0 0 64px; -webkit-font-smoothing:antialiased;
  font-variant-numeric:tabular-nums;
}
.phml .wrap{max-width:1120px; margin:0 auto; padding:0 20px}
.phml a{color:var(--accent-2)}
.phml .safety{position:sticky; top:0; z-index:20; background:var(--accent); color:#fff;
  font-size:.82rem; line-height:1.35; padding:8px 0; border-bottom:1px solid rgba(0,0,0,.15)}
.phml .safety .wrap{padding:0 20px}
.phml header{padding:32px 0 6px}
.phml .eyebrow{font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent-2); font-weight:700; margin:0 0 8px}
.phml h1{font-family:var(--font-head); font-weight:600; font-size:1.95rem; line-height:1.12;
  margin:0 0 10px; text-wrap:balance; letter-spacing:-.01em}
.phml .sub{max-width:74ch; color:var(--muted); font-size:.95rem; margin:0}
.phml section{margin-top:32px}
.phml h2{font-family:var(--font-head); font-weight:600; font-size:1.15rem; margin:0 0 4px;
  display:flex; align-items:baseline; gap:10px; flex-wrap:wrap}
.phml h2 .n{font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-family:var(--font-body)}
.phml .lede{color:var(--muted); font-size:.92rem; margin:0 0 16px; max-width:76ch}
/* findings */
.phml .findings{display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:18px}
.phml .find{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 18px}
.phml .find.win{border-left:4px solid var(--good)}
.phml .find.surprise{border-left:4px solid var(--accent-2)}
.phml .find .tag{font-size:.7rem; letter-spacing:.1em; text-transform:uppercase; font-weight:700; margin:0 0 6px}
.phml .find.win .tag{color:var(--good)}
.phml .find.surprise .tag{color:var(--accent-2)}
.phml .find h3{font-family:var(--font-head); font-weight:600; font-size:1.05rem; margin:0 0 6px}
.phml .find p{margin:0; font-size:.9rem; color:var(--ink)}
.phml .find p+p{margin-top:7px; color:var(--muted)}
/* controls */
.phml .controls{display:flex; flex-wrap:wrap; gap:16px; align-items:center; margin:2px 0 16px}
.phml .seg{display:inline-flex; border:1px solid var(--line); border-radius:999px; overflow:hidden; background:var(--surface)}
.phml .seg button{border:0; background:transparent; color:var(--muted); font:inherit; font-size:.86rem;
  padding:8px 16px; cursor:pointer}
.phml .seg button[aria-pressed="true"]{background:var(--accent); color:#fff; font-weight:600}
.phml .pills{display:flex; flex-wrap:wrap; gap:8px}
.phml .pill{border:1px solid var(--line); background:var(--surface); color:var(--ink); border-radius:999px;
  padding:6px 13px; font:inherit; font-size:.8rem; cursor:pointer; display:flex; gap:7px; align-items:center}
.phml .pill:hover{border-color:var(--accent-2)}
.phml .pill[aria-selected="true"]{border-color:var(--accent); background:color-mix(in srgb,var(--accent) 12%,transparent)}
.phml .pill .pid{font-family:var(--font-mono); font-size:.76rem}
.phml .pill .arch{color:var(--muted); font-size:.74rem}
.phml button:focus-visible,.phml .pill:focus-visible{outline:2px solid var(--accent-2); outline-offset:2px}
/* what changed strip */
.phml .changed{display:flex; flex-wrap:wrap; gap:10px; margin:0 0 14px}
.phml .delta{display:flex; align-items:center; gap:8px; background:var(--surface); border:1px solid var(--line);
  border-radius:10px; padding:8px 12px; font-size:.84rem}
.phml .delta .k{color:var(--muted)}
.phml .delta .v{font-weight:700; display:inline-flex; align-items:center; gap:6px}
.phml .delta .from{color:var(--crit)} .phml .delta .to{color:var(--good)}
.phml .delta .same{color:var(--muted)}
.phml .arrow{color:var(--muted)}
/* before/after hero */
.phml .ba{display:grid; grid-template-columns:1fr 1fr; gap:16px}
.phml .col{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px; display:flex; flex-direction:column}
.phml .col.before{border-top:3px solid var(--crit)}
.phml .col.after{border-top:3px solid var(--good)}
.phml .col .lab{display:flex; justify-content:space-between; align-items:baseline; margin:0 0 8px}
.phml .col .lab b{font-family:var(--font-head); font-size:1rem}
.phml .col .lab .st{font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; font-weight:700}
.phml .col.before .st{color:var(--crit)} .phml .col.after .st{color:var(--good)}
.phml .chips{display:flex; flex-wrap:wrap; gap:6px; margin:0 0 10px}
.phml .chip{font-size:.7rem; padding:3px 8px; border-radius:6px; font-weight:600}
.phml .chip small{opacity:.75}
.phml .chip.ok{background:color-mix(in srgb,var(--good) 15%,transparent); color:var(--good)}
.phml .chip.mid{background:color-mix(in srgb,var(--warn) 18%,transparent); color:var(--warn)}
.phml .chip.bad{background:color-mix(in srgb,var(--crit) 15%,transparent); color:var(--crit)}
.phml .otext{font-size:.82rem; white-space:pre-wrap; line-height:1.5; margin:0; max-height:460px; overflow:auto; color:var(--ink)}
.phml .otext mark{background:var(--mark); color:var(--mark-ink); padding:0 2px; border-radius:3px; font-weight:700}
.phml .otext .hd{font-weight:700; color:var(--accent-2)}
/* case (timeline+gold) */
.phml .caserow{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px}
.phml .card{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px}
.phml .card h4{margin:0 0 8px; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted)}
.phml .flow,.phml .gold{font-size:.82rem; white-space:pre-wrap; margin:0; max-height:280px; overflow:auto; color:var(--ink)}
.phml .flow{font-family:var(--font-mono)}
/* full data (collapsible) */
.phml details.full{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:2px 16px; margin-top:6px}
.phml details.full>summary{cursor:pointer; padding:14px 0; font-weight:600; font-family:var(--font-head)}
.phml .tablecard{overflow-x:auto; margin:2px 0 14px}
.phml table{border-collapse:collapse; width:100%; font-size:.88rem; min-width:640px}
.phml thead th{font-weight:600; text-align:right; color:var(--muted); font-size:.72rem; letter-spacing:.04em;
  text-transform:uppercase; padding:10px 12px; border-bottom:1px solid var(--line)}
.phml thead th:first-child{text-align:left}
.phml tbody td{padding:9px 12px; border-bottom:1px solid var(--line); text-align:right}
.phml tbody td:first-child{text-align:left; font-weight:600}
.phml tbody tr.grp td{border-bottom:2px solid var(--line)}
.phml tbody tr.gold td{color:var(--muted); font-style:italic}
.phml .cell{display:inline-block; min-width:3em; padding:2px 8px; border-radius:6px; font-weight:600}
.phml .s-good{background:color-mix(in srgb,var(--good) 16%,transparent); color:var(--good)}
.phml .s-warn{background:color-mix(in srgb,var(--warn) 18%,transparent); color:var(--warn)}
.phml .s-crit{background:color-mix(in srgb,var(--crit) 16%,transparent); color:var(--crit)}
/* limits + fixed list */
.phml ul.lim{margin:0; padding-left:18px; columns:2; column-gap:28px; font-size:.9rem; color:var(--muted)}
.phml ul.lim li{margin:0 0 8px; break-inside:avoid}
.phml .fixrow{display:flex; flex-wrap:wrap; gap:10px}
.phml .fix{background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--good);
  border-radius:10px; padding:10px 12px; font-size:.82rem; min-width:230px; flex:1}
.phml .fix .who{font-family:var(--font-mono); font-size:.76rem; color:var(--muted)}
.phml .fix .r{color:var(--ink); margin-top:4px}
.phml footer{margin-top:44px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:.82rem}
.phml .themebtn{position:fixed; right:14px; bottom:14px; z-index:30; background:var(--surface);
  border:1px solid var(--line); color:var(--ink); border-radius:999px; padding:8px 12px; font:inherit; font-size:.8rem; cursor:pointer}
@media (max-width:820px){
  .phml .ba,.phml .caserow,.phml .findings{grid-template-columns:1fr}
  .phml ul.lim{columns:1}
  .phml h1{font-size:1.55rem}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

BODY = """
<div class="safety" role="note"><div class="wrap"><b>Synthetic data · research demo.</b>
Synthetic patient timelines only. Not a diagnostic tool; not for real patient data or PHI.
Outputs may be wrong. Do not use for health decisions — consult a clinician.</div></div>

<div class="wrap">
  <header>
    <p class="eyebrow">Preventive Health Model Lab · a fine-tuning study platform</p>
    <h1>What does fine-tuning improve — and does the base model choice matter?</h1>
    <p class="sub">A small controlled platform. Same task, same synthetic data, identical QLoRA recipe on two
      base models — <b>Gemma&nbsp;3</b> and its medical sibling <b>MedGemma</b>. We measure each model
      <b>before and after</b> fine-tuning on the same held-out synthetic patients (n=6). Metrics are automatic:
      they measure form &amp; faithfulness, not clinical correctness.</p>
  </header>

  <section id="findings">
    <h2>Two findings</h2>
    <div class="findings">
      <div class="find win"><p class="tag">What fine-tuning did</p>
        <h3>It fixed the base models' real gaps</h3>
        <p>Out of the box, both models often skipped the safety disclaimer, slipped into diagnostic language,
          and cited numbers that weren't in the record. Fine-tuning reliably resolved all three — safety
          hard-fails went from 100% to 0% for both.</p></div>
      <div class="find surprise"><p class="tag">The part we didn't expect</p>
        <h3>Medical pretraining wasn't the deciding factor</h3>
        <p>We expected the medical base (MedGemma) to pull ahead after fine-tuning. It didn't — both landed
          near the ceiling.</p>
        <p>The read: for this task, fine-tuning's payoff rides on the base model's general capability more than
          on medical pretraining. A capable base adapts well regardless of domain — this says something useful
          about fine-tuning, and takes nothing away from medical models in general.</p></div>
    </div>
  </section>

  <section id="hero">
    <h2>See it happen <span class="n">one base model · before → after</span></h2>
    <p class="lede">Pick a model and a synthetic patient. Left is the base model's answer; right is the same
      model after fine-tuning. Numbers the model made up (not in the timeline) are <mark>highlighted</mark>.</p>
    <div class="controls">
      <div class="seg" id="modelseg" role="group" aria-label="base model">
        <button data-model="gemma3" aria-pressed="false" type="button">Gemma 3</button>
        <button data-model="medgemma" aria-pressed="true" type="button">MedGemma (medical)</button>
      </div>
      <div class="pills" id="pills" role="tablist"></div>
    </div>
    <div class="changed" id="changed"></div>
    <div class="ba">
      <div class="col before"><div class="lab"><b id="lb-before">base</b><span class="st">before fine-tuning</span></div>
        <div class="chips" id="ch-before"></div><div class="otext" id="tx-before"></div></div>
      <div class="col after"><div class="lab"><b id="lb-after">fine-tuned</b><span class="st">after fine-tuning</span></div>
        <div class="chips" id="ch-after"></div><div class="otext" id="tx-after"></div></div>
    </div>
    <div class="caserow">
      <div class="card"><h4>Synthetic timeline (the input both saw)</h4><pre class="flow" id="tl"></pre></div>
      <div class="card"><h4>Gold reference answer</h4><pre class="gold" id="gold"></pre></div>
    </div>
  </section>

  <section id="full">
    <h2>All four conditions <span class="n">the base pair, then the fine-tuned pair</span></h2>
    <p class="lede">Both bases start weak; both fine-tuned models land near the gold ceiling — and the medical
      base doesn't separate from the general one at either stage. "hard-fail" = missing disclaimer or diagnostic
      language; "made-up #" = clinical numbers not in the record.</p>
    <details class="full"><summary>Show the full comparison table</summary>
      <div class="tablecard"><table id="agg"></table></div>
    </details>
  </section>

  <section id="limits">
    <h2>Read before believing the numbers <span class="n">honest limits</span></h2>
    <ul class="lim">
      <li>Tiny test set (n=6) — small differences between the two fine-tuned models are within noise.</li>
      <li>Automatic metrics measure structure, safety framing, and numeric faithfulness — <b>not</b> clinical correctness. No clinician reviewed any output.</li>
      <li>Synthetic + template-derived gold, so fine-tuning largely learns the target format; this likely overstates real-world performance.</li>
      <li>Base MedGemma sometimes rambled (repeating the prompt) and hit the token cap, which lowers its base scores.</li>
      <li>The test split is missing one archetype (improving-after-intervention).</li>
      <li>No real clinical data — whether a medical base helps on real, messy EHR data is still open, and worth testing next.</li>
    </ul>
  </section>

  <section id="fixed">
    <h2>Where the base models slipped — all fixed after fine-tuning <span class="n">shown, not hidden</span></h2>
    <p class="lede">The messy parts are the point: these are the exact gaps the evaluator caught in the
      <em>base</em> models, every one of which fine-tuning closed.</p>
    <div class="fixrow" id="fixes"></div>
  </section>

  <footer>
    Precomputed, read-only, synthetic-only. Built from <code>demo_artifacts/</code> by
    <code>scripts/08_build_demo_page.py</code>. Full write-up: <code>reports/final_experiment_report.md</code>.
    Research/education only — not medical advice.
  </footer>
</div>
<button class="themebtn" id="themebtn" type="button">◐ theme</button>
"""

JS = r"""
const D = window.DEMO;
const CONDS = D.conditions, LBL = D.condLabel, DIMS = D.dims, DLBL = D.dimLabel, AGG = D.aggOrder;
const esc = s => String(s).replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const band = v => v>=0.999?'s-good':(v>=0.5?'s-warn':'s-crit');
const chipband = v => v>=0.999?'ok':(v>=0.5?'mid':'bad');
let STATE = {model:'medgemma', pid:null};

function renderOutput(text, ung){
  let h = esc(text);
  h = h.replace(/^([#>\-\*\d][^\n]{0,80})$/gm, (m,l)=> /(:|\d\.|#|summary|signal|evidence|missing|question|disclaimer|conclude)/i.test(l)&&l.length<80 ? `<span class="hd">${l}</span>` : m);
  (ung||[]).forEach(u=>{const raw=esc(u.raw||'').trim(); if(raw) h=h.split(raw).join('<mark title="not in the timeline">'+raw+'</mark>');});
  return h;
}
function chips(s){
  return DIMS.map(d=>{const v=s.dims[d]; return `<span class="chip ${chipband(v)}">${DLBL[d]} <small>${v.toFixed(2)}</small></span>`;}).join('')
    + (()=>{const n=(s.ungrounded_numbers||[]).length; return n?`<span class="chip bad">made-up # <small>${n}</small></span>`:`<span class="chip ok">made-up # <small>0</small></span>`;})();
}
function boolDelta(k, b, a){
  const bt=b>=0.999, at=a>=0.999;
  const from = bt?'<span class="to">yes</span>':'<span class="from">no</span>';
  const to = at?'<span class="to">yes</span>':'<span class="from">no</span>';
  return `<div class="delta"><span class="k">${k}</span><span class="v">${from}<span class="arrow">→</span>${to}</span></div>`;
}
function numDelta(k, b, a, goodLow){
  const cls=x=> (goodLow? (x===0?'to':'from') : (x>=0.999?'to':'from'));
  return `<div class="delta"><span class="k">${k}</span><span class="v"><span class="${cls(b)}">${b}</span><span class="arrow">→</span><span class="${cls(a)}">${a}</span></span></div>`;
}
function renderHero(){
  const m=STATE.model, pid=STATE.pid;
  const base=D.eval.per_case[pid][m+'_base'], ql=D.eval.per_case[pid][m+'_qlora'];
  const cs=D.cases.find(c=>c.patient_id===pid);
  document.getElementById('tl').textContent=cs.timeline;
  document.getElementById('gold').textContent=cs.gold;
  document.getElementById('lb-before').textContent=LBL[m+'_base'];
  document.getElementById('lb-after').textContent=LBL[m+'_qlora'];
  document.getElementById('ch-before').innerHTML=chips(base);
  document.getElementById('ch-after').innerHTML=chips(ql);
  document.getElementById('tx-before').innerHTML=renderOutput(D.outputs[pid][m+'_base'], base.ungrounded_numbers);
  document.getElementById('tx-after').innerHTML=renderOutput(D.outputs[pid][m+'_qlora'], ql.ungrounded_numbers);
  document.getElementById('changed').innerHTML =
      boolDelta('safety disclaimer', base.dims.disclaimer_present, ql.dims.disclaimer_present)
    + boolDelta('non-diagnostic', base.dims.non_diagnostic, ql.dims.non_diagnostic)
    + numDelta('made-up numbers', (base.ungrounded_numbers||[]).length, (ql.ungrounded_numbers||[]).length, true)
    + numDelta('7-section schema', +base.dims.schema_conformance.toFixed(2), +ql.dims.schema_conformance.toFixed(2), false);
  document.querySelectorAll('#modelseg button').forEach(b=>b.setAttribute('aria-pressed', b.dataset.model===m));
  document.querySelectorAll('#pills .pill').forEach(p=>p.setAttribute('aria-selected', p.dataset.pid===pid));
}
// default: the combo with the most dramatic base gap
function pickDefault(){
  let best=null, score=-1;
  D.cases.forEach(c=>['gemma3','medgemma'].forEach(m=>{
    const b=D.eval.per_case[c.patient_id][m+'_base'];
    const s=(b.hard_fail?2:0)+(b.ungrounded_numbers||[]).length+(b.dims.disclaimer_present<0.999?1:0)+(b.dims.non_diagnostic<0.999?1:0);
    if(s>score){score=s; best={model:m, pid:c.patient_id};}
  }));
  return best;
}
// pills
document.getElementById('pills').innerHTML = D.cases.map(c=>
  `<button class="pill" role="tab" data-pid="${c.patient_id}"><span class="pid">${c.patient_id.replace('SYNTHETIC-GEN-','#')}</span><span class="arch">${c.archetype}</span></button>`).join('');
document.querySelectorAll('#pills .pill').forEach(p=>p.addEventListener('click',()=>{STATE.pid=p.dataset.pid; renderHero();}));
document.querySelectorAll('#modelseg button').forEach(b=>b.addEventListener('click',()=>{STATE.model=b.dataset.model; renderHero();}));
// aggregate table
(function(){
  const head='<thead><tr><th>condition</th>'+DIMS.map(d=>`<th>${DLBL[d]}</th>`).join('')+'<th>overall</th><th>hard-fail</th><th>made-up #</th></tr></thead>';
  const body=AGG.map((c,i)=>{
    const a=D.eval.aggregate[c]; const gold=c==='gold_ceiling'?' gold':''; const grp=(c==='medgemma_base')?' grp':'';
    const cells=DIMS.map(d=>`<td><span class="cell ${band(a.dims[d])}">${a.dims[d].toFixed(2)}</span></td>`).join('');
    const hf=(a.hard_fail_rate*100).toFixed(0)+'%', hfc=a.hard_fail_rate===0?'s-good':'s-crit';
    const hc=a.n_hallucinated_numbers, hcc=hc===0?'s-good':(hc<=2?'s-warn':'s-crit');
    return `<tr class="${gold}${grp}"><td>${LBL[c]}</td>${cells}<td><span class="cell ${band(a.overall_auto_score)}">${a.overall_auto_score.toFixed(2)}</span></td><td><span class="cell ${hfc}">${hf}</span></td><td><span class="cell ${hcc}">${hc}</span></td></tr>`;
  }).join('');
  document.getElementById('agg').innerHTML=head+'<tbody>'+body+'</tbody>';
})();
// fixed-gaps list (base failures)
document.getElementById('fixes').innerHTML=(D.failures||[]).filter(f=>/_base$/.test(f.condition)).map(f=>{
  const reasons=(f.hard_fail_reasons||[]).join('; ')||'ok'; const nh=(f.hallucinated_numbers||[]).length;
  return `<div class="fix"><div class="who">${LBL[f.condition]||f.condition} · ${f.patient_id.replace('SYNTHETIC-GEN-','#')} · ${f.archetype}</div><div class="r">${esc(reasons)}${nh?` · ${nh} made-up number${nh>1?'s':''}`:''} <b style="color:var(--good)">→ fixed</b></div></div>`;
}).join('')||'<p class="lede">None.</p>';
// theme
document.getElementById('themebtn').addEventListener('click',()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next=cur==='dark'?'light':(cur==='light'?'dark':(matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark'));
  document.documentElement.setAttribute('data-theme',next);
});
const def=pickDefault(); STATE.model=def.model; STATE.pid=def.pid; renderHero();
"""


def main() -> int:
    d = load()
    data = {
        "conditions": CONDITIONS, "aggOrder": AGG_ORDER, "condLabel": COND_LABEL,
        "dims": DIMS, "dimLabel": DIM_LABEL,
        "cases": d["cases"], "outputs": d["outputs"], "eval": d["eval"], "failures": d["failures"],
    }
    blob = json.dumps(data, ensure_ascii=False)
    inner = (f"<style>{CSS}</style>\n<div class=\"phml\">{BODY}</div>\n"
             f"<script>window.DEMO={blob};\n{JS}</script>\n")
    (DEMO / "index.body.html").write_text(inner, encoding="utf-8")
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>Preventive Health Model Lab — a fine-tuning study</title>\n</head>\n<body>\n"
            + inner + "</body>\n</html>\n")
    (DEMO / "index.html").write_text(full, encoding="utf-8")
    print(f"wrote {DEMO/'index.html'} ({len(full)//1024} KB) and index.body.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
