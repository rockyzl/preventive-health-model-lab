#!/usr/bin/env python3
"""Build the self-contained static demo page from demo_artifacts/.

A DELIBERATELY SIMPLE, at-a-glance page: anyone should get the point in ~5s
without reading walls of text or scrolling. One headline, one chart, two
plain-English takeaways. The raw model outputs live in a collapsed "see a real
example" section for people who want depth.

Emits:
  demo_artifacts/index.html       full standalone document
  demo_artifacts/index.body.html  body content only (for embedding / Artifact)
"""
from __future__ import annotations

import json
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1] / "demo_artifacts"
COND_LABEL = {"gemma3_base": "Gemma 3", "gemma3_qlora": "Gemma 3",
              "medgemma_base": "MedGemma", "medgemma_qlora": "MedGemma"}


def load():
    return {
        "cases": json.loads((DEMO / "synthetic_cases.json").read_text()),
        "outputs": json.loads((DEMO / "model_outputs.json").read_text()),
        "eval": json.loads((DEMO / "evaluation_summary.json").read_text()),
    }


CSS = r"""
:root{
  --paper:#eef2f3; --surface:#ffffff; --ink:#141c21; --muted:#5b6b73; --line:#d6e0e3;
  --accent:#0b5450; --accent-2:#0f8a80; --good:#2f8f5b; --warn:#b7791b; --crit:#c0453b; --mark:#fbe6b0; --mark-ink:#5a4406;
}
@media (prefers-color-scheme:dark){:root{
  --paper:#0d1316; --surface:#141c21; --ink:#e4edf0; --muted:#8ba0a9; --line:#26333b;
  --accent:#39b3a8; --accent-2:#4fd0c4; --good:#57bd83; --warn:#d9a441; --crit:#e0685d; --mark:#5a4a12; --mark-ink:#ffe9ab;}}
:root[data-theme="light"]{--paper:#eef2f3;--surface:#fff;--ink:#141c21;--muted:#5b6b73;--line:#d6e0e3;--accent:#0b5450;--accent-2:#0f8a80;--good:#2f8f5b;--warn:#b7791b;--crit:#c0453b;--mark:#fbe6b0;--mark-ink:#5a4406;}
:root[data-theme="dark"]{--paper:#0d1316;--surface:#141c21;--ink:#e4edf0;--muted:#8ba0a9;--line:#26333b;--accent:#39b3a8;--accent-2:#4fd0c4;--good:#57bd83;--warn:#d9a441;--crit:#e0685d;--mark:#5a4a12;--mark-ink:#ffe9ab;}
*{box-sizing:border-box}
.phml{--fh:Georgia,"Iowan Old Style",serif; --fb:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; --fm:ui-monospace,"SFMono-Regular",Menlo,monospace;
  background:var(--paper); color:var(--ink); font-family:var(--fb); line-height:1.5; margin:0;
  -webkit-font-smoothing:antialiased; font-variant-numeric:tabular-nums;}
.phml .safety{background:var(--accent); color:#fff; font-size:.76rem; line-height:1.3; padding:6px 16px; text-align:center;}
.phml .wrap{max-width:820px; margin:0 auto; padding:22px 20px 30px;}
.phml h1{font-family:var(--fh); font-weight:600; font-size:1.5rem; line-height:1.15; margin:0 0 8px; text-wrap:balance;}
.phml .sub{color:var(--muted); font-size:.95rem; margin:0 0 22px; max-width:66ch;}
/* takeaways — the point, big and scannable */
.phml .takes{display:grid; gap:12px; margin:0 0 24px;}
.phml .take{display:flex; gap:12px; align-items:flex-start; background:var(--surface); border:1px solid var(--line);
  border-radius:12px; padding:14px 16px;}
.phml .take.win{border-left:4px solid var(--good);} .phml .take.surprise{border-left:4px solid var(--accent-2);}
.phml .take .ic{font-size:1.3rem; line-height:1.2; flex:none;}
.phml .take b{font-family:var(--fh); font-weight:600; font-size:1.05rem; display:block; margin-bottom:2px;}
.phml .take span{color:var(--muted); font-size:.9rem;}
/* chart */
.phml .chartcard{background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:16px 18px; margin:0 0 22px;}
.phml .chartcard h2{font-family:var(--fb); font-size:.82rem; letter-spacing:.04em; text-transform:uppercase; color:var(--muted); margin:0 0 4px; font-weight:700;}
.phml .chartcard .cap{color:var(--muted); font-size:.82rem; margin:0 0 14px;}
.phml .bars{display:grid; gap:9px;}
.phml .barrow{display:grid; grid-template-columns:118px 1fr; gap:12px; align-items:center;}
.phml .barrow .lbl{font-size:.82rem; text-align:right; color:var(--ink);}
.phml .barrow .lbl small{color:var(--muted); display:block; font-size:.72rem;}
.phml .track{position:relative; background:color-mix(in srgb,var(--muted) 12%,transparent); border-radius:6px; height:26px;}
.phml .fill{position:absolute; left:0; top:0; bottom:0; border-radius:6px; display:flex; align-items:center; justify-content:flex-end;
  padding-right:8px; color:#fff; font-size:.78rem; font-weight:700; min-width:2.6em;}
.phml .fill.base{background:var(--warn);} .phml .fill.tuned{background:var(--good);} .phml .fill.gold{background:var(--muted);}
.phml .legend{display:flex; gap:16px; margin-top:12px; font-size:.78rem; color:var(--muted); flex-wrap:wrap;}
.phml .legend i{display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:5px; vertical-align:-1px;}
.phml .sw-base{background:var(--warn);} .phml .sw-tuned{background:var(--good);}
/* fine print + links */
.phml .fine{color:var(--muted); font-size:.8rem; margin:0 0 4px;}
.phml .links{display:flex; gap:18px; flex-wrap:wrap; font-size:.88rem; margin:6px 0 0;}
.phml .links a{color:var(--accent-2);}
/* collapsed example */
.phml details.ex{margin-top:22px; background:var(--surface); border:1px solid var(--line); border-radius:12px; padding:2px 16px;}
.phml details.ex>summary{cursor:pointer; padding:14px 0; font-weight:600; font-family:var(--fh);}
.phml .ba{display:grid; grid-template-columns:1fr 1fr; gap:14px; padding-bottom:14px;}
.phml .col{border:1px solid var(--line); border-radius:10px; padding:12px 14px; min-width:0;}
.phml .col.before{border-top:3px solid var(--crit);} .phml .col.after{border-top:3px solid var(--good);}
.phml .col h3{margin:0 0 8px; font-size:.85rem;} .phml .col .st{font-size:.68rem; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); font-weight:700;}
.phml .otext{font-size:.8rem; line-height:1.5; max-height:340px; overflow:auto; color:var(--ink);}
.phml .otext .hd{font-weight:700; color:var(--accent-2); display:block; margin:8px 0 2px;}
.phml .otext strong{font-weight:700;} .phml .otext ul{margin:3px 0 3px 0; padding-left:18px;} .phml .otext li{margin:1px 0;}
.phml .otext p{margin:3px 0;} .phml .otext mark{background:var(--mark); color:var(--mark-ink); padding:0 2px; border-radius:3px; font-weight:700;}
.phml .otext .trow{font-family:var(--fm); font-size:.74rem; white-space:pre-wrap;}
.phml .themebtn{position:fixed; right:12px; bottom:12px; background:var(--surface); border:1px solid var(--line); color:var(--ink);
  border-radius:999px; padding:7px 11px; font:inherit; font-size:.78rem; cursor:pointer;}
@media (max-width:680px){.phml .ba{grid-template-columns:1fr;} .phml .barrow{grid-template-columns:96px 1fr;}}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
"""

BODY = """
<div class="safety">Synthetic data · research/education only · not a diagnostic tool · do not enter real health data</div>
<div class="wrap">
  <h1 id="h1">I fine-tuned two open medical AIs to reason about health trends — safely.</h1>
  <p class="sub">Same task, same synthetic data, same recipe on two models — Gemma&nbsp;3 and its medical
    sibling MedGemma. The question: what does fine-tuning fix, and does a "medical" model actually do better?</p>

  <div class="takes">
    <div class="take win"><span class="ic">✅</span><div>
      <b id="t1b">Fine-tuning fixed the safety problems.</b>
      <span id="t1s">Both models went from breaking the safety rules to following them every time.</span></div></div>
    <div class="take surprise"><span class="ic">🔍</span><div>
      <b>The medical model didn't win.</b>
      <span id="t2s">After fine-tuning it was tied with the general one — a capable base + fine-tuning is what mattered, not the medical pretraining.</span></div></div>
  </div>

  <div class="chartcard">
    <h2>Overall quality score, before vs after fine-tuning</h2>
    <p class="cap" id="chartcap">Higher is better (0–1). Automatic score: safe framing + correct structure + no made-up numbers.</p>
    <div class="bars" id="bars"></div>
    <div class="legend"><span><i class="sw-base"></i>before fine-tuning</span><span><i class="sw-tuned"></i>after fine-tuning</span></div>
  </div>

  <p class="fine" id="fine"></p>
  <p class="links">
    <a href="https://github.com/rockyzl/preventive-health-model-lab" target="_blank" rel="noopener">Code &amp; full write-up →</a>
    <a href="https://huggingface.co/rockyaaos/preventive-health-model-lab" target="_blank" rel="noopener">Hugging Face ↗</a>
  </p>

  <details class="ex">
    <summary>See a real before → after example (one synthetic patient)</summary>
    <div class="ba">
      <div class="col before"><h3 id="ex-lb">MedGemma</h3><span class="st">before fine-tuning</span><div class="otext" id="ex-before"></div></div>
      <div class="col after"><h3>same model</h3><span class="st">after fine-tuning</span><div class="otext" id="ex-after"></div></div>
    </div>
  </details>
</div>
<button class="themebtn" id="themebtn" type="button">◐ theme</button>
"""

JS = r"""
const D = window.DEMO, A = D.eval.aggregate;
const esc = s => String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const pct = x => Math.round(x*100);

// ---- chart: 4 bars grouped by model + gold ceiling ----
(function(){
  const rows = [
    {lbl:'Gemma 3',   sub:'general',  base:A.gemma3_base.overall_auto_score,  tuned:A.gemma3_qlora.overall_auto_score},
    {lbl:'MedGemma',  sub:'medical',  base:A.medgemma_base.overall_auto_score, tuned:A.medgemma_qlora.overall_auto_score},
  ];
  let html = '';
  rows.forEach(r=>{
    html += `<div class="barrow"><div class="lbl">${r.lbl}<small>${r.sub} · base</small></div>
      <div class="track"><div class="fill base" style="width:${Math.max(r.base*100,9)}%">${r.base.toFixed(2)}</div></div></div>`;
    html += `<div class="barrow"><div class="lbl">${r.lbl}<small>${r.sub} · fine-tuned</small></div>
      <div class="track"><div class="fill tuned" style="width:${Math.max(r.tuned*100,9)}%">${r.tuned.toFixed(2)}</div></div></div>`;
  });
  const gold = A.gold_ceiling.overall_auto_score;
  html += `<div class="barrow"><div class="lbl">Gold<small>ideal answer</small></div>
    <div class="track"><div class="fill gold" style="width:${gold*100}%">${gold.toFixed(2)}</div></div></div>`;
  document.getElementById('bars').innerHTML = html;
})();

// ---- takeaway numbers + fine print, pulled from the data ----
(function(){
  const n = A.gemma3_base.n;
  const baseHF = Math.round((A.gemma3_base.hard_fail_rate + A.medgemma_base.hard_fail_rate)/2*100);
  const tunedHF = Math.round((A.gemma3_qlora.hard_fail_rate + A.medgemma_qlora.hard_fail_rate)/2*100);
  document.getElementById('t1s').textContent =
    `Both models broke a safety rule (no disclaimer, or diagnosis-style wording) on ${baseHF}% of cases before fine-tuning, and ${tunedHF}% after.`;
  const baseMU = A.gemma3_base.n_hallucinated_numbers + A.medgemma_base.n_hallucinated_numbers;
  const tunedMU = A.gemma3_qlora.n_hallucinated_numbers + A.medgemma_qlora.n_hallucinated_numbers;
  document.getElementById('fine').textContent =
    `Measured on ${n} held-out synthetic patients with automatic checks (not clinical review). Made-up numbers dropped from ${baseMU} to ${tunedMU} across all outputs. Small synthetic test — a demonstration, not clinical evidence.`;
})();

// ---- mini-markdown for the collapsed example ----
function inline(s){
  return s.replace(/\*\*([^*]+)\*\*/g,'<strong>$1</strong>')
          .replace(/(^|[^*])\*(?!\*)([^*\n]+)\*(?!\*)/g,'$1<em>$2</em>');
}
function md(text, ung){
  let h = esc(text);
  (ung||[]).forEach(u=>{const raw=esc(String(u.raw||'')).trim(); if(raw) h=h.split(raw).join(''+raw+'');});
  const out=[]; let inList=false;
  for(const line of h.split('\n')){
    let m;
    if((m=line.match(/^\s*#{1,6}\s+(.*)$/))){ if(inList){out.push('</ul>');inList=false;} out.push('<span class="hd">'+inline(m[1])+'</span>'); }
    else if(/^\s*\|.*\|\s*$/.test(line)){ if(inList){out.push('</ul>');inList=false;} if(!/^\s*\|[\s\-:|]+\|\s*$/.test(line)) out.push('<div class="trow">'+inline(line)+'</div>'); }
    else if((m=line.match(/^\s*[\*\-]\s+(.*)$/))){ if(!inList){out.push('<ul>');inList=true;} out.push('<li>'+inline(m[1])+'</li>'); }
    else if(line.trim()===''){ if(inList){out.push('</ul>');inList=false;} }
    else { if(inList){out.push('</ul>');inList=false;} out.push('<p>'+inline(line)+'</p>'); }
  }
  if(inList) out.push('</ul>');
  return out.join('').split('').join('<mark title="not in the record">').split('').join('</mark>');
}
// pick the most illustrative example: biggest base failure
(function(){
  let best=null, sc=-1;
  D.cases.forEach(c=>['gemma3','medgemma'].forEach(mm=>{
    const b=D.eval.per_case[c.patient_id][mm+'_base'];
    const s=(b.hard_fail?2:0)+(b.ungrounded_numbers||[]).length+(b.dims.disclaimer_present<0.999?1:0);
    if(s>sc){sc=s; best={m:mm, pid:c.patient_id};}
  }));
  const {m,pid}=best;
  const lab = m==='medgemma' ? 'MedGemma' : 'Gemma 3';
  document.getElementById('ex-lb').textContent = lab;
  document.getElementById('ex-before').innerHTML = md(D.outputs[pid][m+'_base'], D.eval.per_case[pid][m+'_base'].ungrounded_numbers);
  document.getElementById('ex-after').innerHTML  = md(D.outputs[pid][m+'_qlora'], D.eval.per_case[pid][m+'_qlora'].ungrounded_numbers);
})();

document.getElementById('themebtn').addEventListener('click',()=>{
  const cur=document.documentElement.getAttribute('data-theme');
  const next=cur==='dark'?'light':(cur==='light'?'dark':(matchMedia('(prefers-color-scheme: dark)').matches?'light':'dark'));
  document.documentElement.setAttribute('data-theme',next);
});
"""


def main() -> int:
    d = load()
    data = {"cases": d["cases"], "outputs": d["outputs"], "eval": d["eval"]}
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
