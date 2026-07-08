#!/usr/bin/env python3
"""Build the self-contained static demo page from demo_artifacts/.

Reads the precomputed (synthetic-only) JSON in demo_artifacts/ and emits:
  demo_artifacts/index.html       full standalone document (deploy anywhere static)
  demo_artifacts/index.body.html  body content only (for embedding / the Artifact tool)

No network, no model, no live inference. All data is inlined. Read-only viewer.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "demo_artifacts"
CONDITIONS = ["gemma3_base", "gemma3_qlora", "medgemma_base", "medgemma_qlora"]
COND_LABEL = {
    "gemma3_base": "Gemma 3 · base",
    "gemma3_qlora": "Gemma 3 · QLoRA",
    "medgemma_base": "MedGemma · base",
    "medgemma_qlora": "MedGemma · QLoRA",
    "gold_ceiling": "Gold (ceiling)",
}
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
  :root{
    --paper:#0d1316; --surface:#141c21; --surface-2:#0f171b; --ink:#dde7ea;
    --muted:#8ba0a9; --line:#26333b; --accent:#39b3a8; --accent-2:#4fd0c4;
    --good:#57bd83; --warn:#d9a441; --crit:#e0685d;
    --mark:#5a4a12; --mark-ink:#ffe9ab;
  }
}
:root[data-theme="light"]{
  --paper:#eef2f3; --surface:#ffffff; --surface-2:#f5f8f9; --ink:#151d22;
  --muted:#5b6b73; --line:#d6e0e3; --accent:#0b5450; --accent-2:#0f8a80;
  --good:#2f8f5b; --warn:#b7791b; --crit:#c0453b; --mark:#fbe6b0; --mark-ink:#5a4406;
}
:root[data-theme="dark"]{
  --paper:#0d1316; --surface:#141c21; --surface-2:#0f171b; --ink:#dde7ea;
  --muted:#8ba0a9; --line:#26333b; --accent:#39b3a8; --accent-2:#4fd0c4;
  --good:#57bd83; --warn:#d9a441; --crit:#e0685d; --mark:#5a4a12; --mark-ink:#ffe9ab;
}
*{box-sizing:border-box}
.phml{
  --font-head:Georgia,"Iowan Old Style","Palatino Linotype",serif;
  --font-body:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --font-mono:ui-monospace,"SFMono-Regular","Cascadia Code",Menlo,monospace;
  background:var(--paper); color:var(--ink); font-family:var(--font-body);
  line-height:1.5; margin:0; padding:0 0 64px;
  -webkit-font-smoothing:antialiased; font-variant-numeric:tabular-nums;
}
.phml .wrap{max-width:1120px; margin:0 auto; padding:0 20px}
.phml a{color:var(--accent-2)}
/* safety banner — always visible */
.phml .safety{
  position:sticky; top:0; z-index:20; background:var(--accent); color:#fff;
  font-size:.82rem; line-height:1.35; padding:8px 20px; border-bottom:1px solid rgba(0,0,0,.15);
}
.phml .safety b{letter-spacing:.02em}
.phml .safety .wrap{max-width:1120px; margin:0 auto; padding:0}
/* header */
.phml header{padding:34px 0 10px}
.phml .eyebrow{font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent-2); font-weight:700; margin:0 0 8px}
.phml h1{font-family:var(--font-head); font-weight:600; font-size:2rem; line-height:1.12;
  margin:0 0 10px; text-wrap:balance; letter-spacing:-.01em}
.phml .thesis{max-width:70ch; color:var(--ink); font-size:1.02rem; margin:0 0 4px}
.phml .thesis strong{color:var(--accent-2)}
.phml .sub{color:var(--muted); font-size:.9rem; margin:6px 0 0}
.phml section{margin-top:34px}
.phml h2{font-family:var(--font-head); font-weight:600; font-size:1.15rem; margin:0 0 4px;
  display:flex; align-items:baseline; gap:10px}
.phml h2 .n{font-size:.7rem; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); font-family:var(--font-body)}
.phml .lede{color:var(--muted); font-size:.92rem; margin:0 0 16px; max-width:74ch}
/* result table */
.phml .tablecard{background:var(--surface); border:1px solid var(--line); border-radius:12px; overflow-x:auto}
.phml table{border-collapse:collapse; width:100%; font-size:.9rem; min-width:640px}
.phml thead th{font-family:var(--font-body); font-weight:600; text-align:right; color:var(--muted);
  font-size:.74rem; letter-spacing:.04em; text-transform:uppercase; padding:12px 14px; border-bottom:1px solid var(--line)}
.phml thead th:first-child{text-align:left}
.phml tbody td{padding:11px 14px; border-bottom:1px solid var(--line); text-align:right; font-variant-numeric:tabular-nums}
.phml tbody tr:last-child td{border-bottom:0}
.phml tbody td:first-child{text-align:left; font-weight:600}
.phml tbody tr.gold td{color:var(--muted); font-style:italic}
.phml tbody tr.win td:first-child::after{content:""}
.phml .cell{display:inline-block; min-width:3.2em; padding:2px 8px; border-radius:6px; font-weight:600}
.phml .s-good{background:color-mix(in srgb,var(--good) 16%,transparent); color:var(--good)}
.phml .s-warn{background:color-mix(in srgb,var(--warn) 18%,transparent); color:var(--warn)}
.phml .s-crit{background:color-mix(in srgb,var(--crit) 16%,transparent); color:var(--crit)}
.phml .takeaways{display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:16px}
.phml .tk{background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:10px; padding:14px 16px}
.phml .tk h3{margin:0 0 5px; font-size:.95rem; font-family:var(--font-head); font-weight:600}
.phml .tk p{margin:0; font-size:.88rem; color:var(--muted)}
.phml .tk.warnedge{border-left-color:var(--warn)}
/* case picker */
.phml .pills{display:flex; flex-wrap:wrap; gap:8px; margin:0 0 18px}
.phml .pill{border:1px solid var(--line); background:var(--surface); color:var(--ink);
  border-radius:999px; padding:7px 14px; font:inherit; font-size:.84rem; cursor:pointer;
  display:flex; gap:8px; align-items:center; transition:border-color .12s,background .12s}
.phml .pill:hover{border-color:var(--accent-2)}
.phml .pill[aria-selected="true"]{border-color:var(--accent); background:color-mix(in srgb,var(--accent) 12%,transparent)}
.phml .pill .arch{color:var(--muted); font-size:.76rem}
.phml .pill .pid{font-family:var(--font-mono); font-size:.78rem}
:root[data-theme="dark"] .pill[aria-selected="true"], .phml .pill:focus-visible{outline:2px solid var(--accent-2); outline-offset:2px}
/* selected case */
.phml .caserow{display:grid; grid-template-columns:1fr 1fr; gap:16px}
.phml .card{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px}
.phml .card h4{margin:0 0 8px; font-size:.72rem; letter-spacing:.1em; text-transform:uppercase; color:var(--muted)}
.phml .flow{font-family:var(--font-mono); font-size:.82rem; line-height:1.5; white-space:pre-wrap;
  margin:0; color:var(--ink); max-height:340px; overflow:auto}
.phml .gold{font-size:.86rem; white-space:pre-wrap; margin:0; max-height:340px; overflow:auto; color:var(--ink)}
.phml .outputs{display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px}
.phml .out{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:14px 16px; display:flex; flex-direction:column}
.phml .out.hardfail{border-top:3px solid var(--crit)}
.phml .out.clean{border-top:3px solid var(--good)}
.phml .out .cond{font-family:var(--font-head); font-weight:600; font-size:.98rem; margin:0 0 8px; display:flex; justify-content:space-between; align-items:center}
.phml .chips{display:flex; flex-wrap:wrap; gap:6px; margin:0 0 10px}
.phml .chip{font-size:.7rem; padding:3px 8px; border-radius:6px; font-weight:600; letter-spacing:.01em}
.phml .chip small{opacity:.75; font-weight:600}
.phml .chip.ok{background:color-mix(in srgb,var(--good) 15%,transparent); color:var(--good)}
.phml .chip.mid{background:color-mix(in srgb,var(--warn) 18%,transparent); color:var(--warn)}
.phml .chip.bad{background:color-mix(in srgb,var(--crit) 15%,transparent); color:var(--crit)}
.phml .halluc{margin-left:auto; font-size:.72rem; color:var(--crit); font-weight:600}
.phml .otext{font-size:.82rem; white-space:pre-wrap; line-height:1.5; margin:0; max-height:420px; overflow:auto; color:var(--ink)}
.phml .otext mark{background:var(--mark); color:var(--mark-ink); padding:0 2px; border-radius:3px; font-weight:700}
.phml .otext .hd{font-weight:700; color:var(--accent-2)}
/* limitations + failures */
.phml ul.lim{margin:0; padding-left:18px; columns:2; column-gap:28px; font-size:.9rem; color:var(--muted)}
.phml ul.lim li{margin:0 0 8px; break-inside:avoid}
.phml .failrow{display:flex; flex-wrap:wrap; gap:10px}
.phml .fail{background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--crit);
  border-radius:10px; padding:10px 12px; font-size:.82rem; min-width:230px; flex:1}
.phml .fail .who{font-family:var(--font-mono); font-size:.76rem; color:var(--muted)}
.phml .fail .r{color:var(--ink); margin-top:4px}
.phml footer{margin-top:44px; padding-top:16px; border-top:1px solid var(--line); color:var(--muted); font-size:.82rem}
.phml .toggle{position:fixed; right:14px; bottom:14px; z-index:30; background:var(--surface);
  border:1px solid var(--line); color:var(--ink); border-radius:999px; padding:8px 12px; font:inherit; font-size:.8rem; cursor:pointer}
@media (max-width:820px){
  .phml .caserow,.phml .outputs,.phml .takeaways{grid-template-columns:1fr}
  .phml ul.lim{columns:1}
  .phml h1{font-size:1.6rem}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

BODY = """
<div class="safety" role="note"><div class="wrap"><b>Synthetic data · research demo.</b>
Synthetic patient timelines only. Not a diagnostic tool; not for real patient data or PHI.
Outputs may be wrong. Do not use for health decisions — consult a clinician.</div></div>

<div class="wrap">
  <header>
    <p class="eyebrow">Preventive Health Model Lab · held-out synthetic test (n=6)</p>
    <h1>Does medical pretraining help a 4B model reason about preventive health?</h1>
    <p class="thesis">Fine-tuning helped both models a lot. But on this synthetic benchmark,
      the medical model <strong>did not beat</strong> its non-medical twin — an honest,
      slightly inconvenient result.</p>
    <p class="sub">Same base family (Gemma&nbsp;3 vs MedGemma), identical QLoRA recipe, same synthetic
      data &amp; split. Metrics are automatic — they measure form &amp; faithfulness, not clinical correctness.</p>
  </header>

  <section id="result">
    <h2>Result at a glance <span class="n">automatic scores, 0–1</span></h2>
    <p class="lede">Overall is the mean of four checks. "hard-fail" = missing safety disclaimer or diagnostic
      language. "halluc #" = clinical numbers cited that are not in the record. Gold answers set the ceiling.</p>
    <div class="tablecard"><table id="agg"></table></div>
    <div class="takeaways">
      <div class="tk"><h3>QLoRA worked — for both</h3><p>Both models went from a 100% safety hard-fail rate
        to 0%, and from overall ~0.61 to ~1.0. The biggest, most consistent effect was learning to follow the
        non-diagnostic, disclaimer-bearing format.</p></div>
      <div class="tk warnedge"><h3>Medical pretraining: no measurable edge</h3><p>After fine-tuning the
        general-purpose control matched (marginally beat) MedGemma. With n=6 and automatic metrics on synthetic
        data, the honest read is "no evidence of an advantage here", not "the control wins".</p></div>
    </div>
  </section>

  <section id="cases">
    <h2>Inspect a synthetic patient <span class="n">timeline · gold · 4 model outputs</span></h2>
    <p class="lede">Pick a case. You see the synthetic timeline, the reference answer, and each model's output
      before and after fine-tuning. Numbers the model made up (not in the timeline) are <mark>highlighted</mark>.</p>
    <div class="pills" id="pills" role="tablist"></div>
    <div class="caserow">
      <div class="card"><h4>Synthetic timeline (model input)</h4><pre class="flow" id="tl"></pre></div>
      <div class="card"><h4>Gold reference answer</h4><pre class="gold" id="gold"></pre></div>
    </div>
    <div class="outputs" id="outs"></div>
  </section>

  <section id="limits">
    <h2>Limitations <span class="n">read before believing the numbers</span></h2>
    <ul class="lim">
      <li>Tiny test set (n=6) — rank differences between the two fine-tuned models are within noise.</li>
      <li>Automatic metrics measure structure, safety framing, and numeric faithfulness — <b>not</b> clinical correctness. No clinician reviewed any output.</li>
      <li>Synthetic + template-derived gold, so fine-tuning largely learns the target format; this likely overstates real-world performance.</li>
      <li>Base MedGemma outputs partly degenerated (repeating the prompt) and hit the token cap, dragging its base scores down.</li>
      <li>The test split is missing one archetype (improving-after-intervention).</li>
      <li>No real clinical data. Whether the medical base helps on real, messy EHR data is unproven.</li>
    </ul>
  </section>

  <section id="failures">
    <h2>Failures, not hidden <span class="n">every hard-fail &amp; hallucination</span></h2>
    <p class="lede">Showing the messy parts is the point. These are the safety hard-fails and made-up numbers the
      evaluator caught — mostly from the <em>base</em> models before fine-tuning.</p>
    <div class="failrow" id="fails"></div>
  </section>

  <footer>
    Precomputed, read-only, synthetic-only. Built from <code>demo_artifacts/</code> by
    <code>scripts/08_build_demo_page.py</code>. Full analysis: <code>reports/final_experiment_report.md</code>.
    Research/education only — not medical advice.
  </footer>
</div>
<button class="toggle" id="themebtn" type="button">◐ theme</button>
"""

JS = r"""
const D = window.DEMO;
const CONDS = D.conditions, LBL = D.condLabel, DIMS = D.dims, DLBL = D.dimLabel;
const esc = s => s.replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const band = v => v>=0.999?'s-good':(v>=0.5?'s-warn':'s-crit');
const chipband = v => v>=0.999?'ok':(v>=0.5?'mid':'bad');

// aggregate table
(function(){
  const rows = [...CONDS,'gold_ceiling'];
  const head = '<thead><tr><th>condition</th>' + DIMS.map(d=>`<th>${DLBL[d]}</th>`).join('')
    + '<th>overall</th><th>hard-fail</th><th>halluc #</th></tr></thead>';
  const body = rows.map(c=>{
    const a = D.eval.aggregate[c]; const gold = c==='gold_ceiling'?' class="gold"':'';
    const cells = DIMS.map(d=>`<td><span class="cell ${band(a.dims[d])}">${a.dims[d].toFixed(2)}</span></td>`).join('');
    const hf = (a.hard_fail_rate*100).toFixed(0)+'%';
    const hfc = a.hard_fail_rate===0?'s-good':'s-crit';
    const hc = a.n_hallucinated_numbers; const hcc = hc===0?'s-good':(hc<=2?'s-warn':'s-crit');
    return `<tr${gold}><td>${LBL[c]}</td>${cells}`
      + `<td><span class="cell ${band(a.overall_auto_score)}">${a.overall_auto_score.toFixed(2)}</span></td>`
      + `<td><span class="cell ${hfc}">${hf}</span></td>`
      + `<td><span class="cell ${hcc}">${hc}</span></td></tr>`;
  }).join('');
  document.getElementById('agg').innerHTML = head + '<tbody>' + body + '</tbody>';
})();

// light markdown-ish render + highlight hallucinated raws
function renderOutput(text, ungrounded){
  let h = esc(text);
  // bold header-ish lines
  h = h.replace(/^([#>\-\*\d][^\n]{0,80})$/gm, (m,l)=> /(:|\d\.|#|summary|signal|evidence|missing|question|disclaimer|conclude)/i.test(l) && l.length<80 ? `<span class="hd">${l}</span>` : m);
  (ungrounded||[]).forEach(u=>{
    const raw = esc(String(u.raw||'')).trim();
    if(raw.length>=1){ h = h.split(raw).join('<mark title="not found in the timeline">'+raw+'</mark>'); }
  });
  return h;
}

function renderCase(pid){
  const cs = D.cases.find(c=>c.patient_id===pid);
  document.getElementById('tl').textContent = cs.timeline;
  document.getElementById('gold').textContent = cs.gold;
  const per = D.eval.per_case[pid];
  const out = D.outputs[pid];
  document.getElementById('outs').innerHTML = CONDS.map(c=>{
    const s = per[c]; const hardfail = s.hard_fail;
    const chips = DIMS.map(d=>{
      const v = s.dims[d]; const cl = chipband(v);
      return `<span class="chip ${cl}">${DLBL[d]} <small>${v.toFixed(2)}</small></span>`;
    }).join('');
    const nh = (s.ungrounded_numbers||[]).length;
    const hall = nh>0 ? `<span class="halluc">⚠ ${nh} made-up #</span>` : '';
    const body = renderOutput(out[c]||'', s.ungrounded_numbers);
    return `<div class="out ${hardfail?'hardfail':'clean'}">
      <p class="cond">${LBL[c]} ${hardfail?'<span class="halluc">hard-fail</span>':''}</p>
      <div class="chips">${chips}${hall}</div>
      <div class="otext">${body}</div></div>`;
  }).join('');
  document.querySelectorAll('.pill').forEach(p=>p.setAttribute('aria-selected', p.dataset.pid===pid));
}

// pills
(function(){
  document.getElementById('pills').innerHTML = D.cases.map((c,i)=>
    `<button class="pill" role="tab" data-pid="${c.patient_id}" aria-selected="${i===0}">
       <span class="pid">${c.patient_id.replace('SYNTHETIC-GEN-','#')}</span>
       <span class="arch">${c.archetype}</span></button>`).join('');
  document.querySelectorAll('.pill').forEach(p=>p.addEventListener('click', ()=>renderCase(p.dataset.pid)));
})();

// failures
(function(){
  document.getElementById('fails').innerHTML = (D.failures||[]).map(f=>{
    const reasons = (f.hard_fail_reasons||[]).join('; ') || 'ok';
    const nh = (f.hallucinated_numbers||[]).length;
    const nhtxt = nh>0 ? ` · ${nh} made-up number${nh>1?'s':''}` : '';
    return `<div class="fail"><div class="who">${LBL[f.condition]||f.condition} · ${f.patient_id.replace('SYNTHETIC-GEN-','#')} · ${f.archetype}</div>
      <div class="r">${esc(reasons)}${nhtxt}</div></div>`;
  }).join('') || '<p class="lede">No failures recorded.</p>';
})();

renderCase(D.cases[0].patient_id);

// theme toggle
(function(){
  const btn = document.getElementById('themebtn');
  btn.addEventListener('click', ()=>{
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur==='dark'?'light':(cur==='light'?'dark':(matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark'));
    document.documentElement.setAttribute('data-theme', next);
  });
})();
"""


def main() -> int:
    d = load()
    data = {
        "conditions": CONDITIONS,
        "condLabel": COND_LABEL,
        "dims": DIMS,
        "dimLabel": DIM_LABEL,
        "cases": d["cases"],
        "outputs": d["outputs"],
        "eval": d["eval"],
        "failures": d["failures"],
    }
    blob = json.dumps(data, ensure_ascii=False)
    inner = (f"<style>{CSS}</style>\n<div class=\"phml\">{BODY}</div>\n"
             f"<script>window.DEMO={blob};\n{JS}</script>\n")
    (DEMO / "index.body.html").write_text(inner, encoding="utf-8")
    full = ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            "<title>Preventive Health Model Lab — results</title>\n</head>\n<body>\n"
            + inner + "</body>\n</html>\n")
    (DEMO / "index.html").write_text(full, encoding="utf-8")
    print(f"wrote {DEMO/'index.html'} ({len(full)//1024} KB) and index.body.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
