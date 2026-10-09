"""Build validation/coding_tool.html, a local page for hand-coding the validation sample.

The page shows one device at a time: the sponsor's summary text, opened at the testing
section, with keys Y / N / U to record whether the summary reports testing on patients
or patient data. Codes autosave in the browser and export to a CSV that replaces
validation/hand_coded_sample.csv.

Blinding: the page never shows the classifier's output. The testing section is found by
its heading, not by the classifier's keywords, and nothing in the text is highlighted.

Run:  .venv/bin/python src/make_coding_tool.py   then open validation/coding_tool.html
"""

import json
import re

import pandas as pd

from common import INTERIM, ROOT

SAMPLE = ROOT / "validation" / "hand_coded_sample.csv"
OUT = ROOT / "validation" / "coding_tool.html"
TXT_DIR = INTERIM / "summaries"

SUMMARY_START = re.compile(r"(?:^|\f)[^\n\f]{0,30}?510\s*\(\s*k\s*\)\s*summary\b", re.I | re.M)
# A short line that names a testing section, e.g. "VII. PERFORMANCE DATA", "Clinical Testing:".
TESTING_HEADING = re.compile(
    r"(?im)^[^\n]{0,25}?\b(performance (data|testing|tests?|evaluation|assessment)"
    r"|clinical (testing|tests?|performance|stud(y|ies)|data|evidence|validation|evaluation)"
    r"|non-?clinical (testing|tests?|data|performance|stud(y|ies))|summary of (non-?)?clinical"
    r"|bench (testing|performance)|nonclinical/bench)[^\n]{0,50}$")


def is_heading(line: str) -> bool:
    line = line.strip()
    return len(line) <= 80 and not re.search(r"[a-z]{3,}\.$", line)


def prepare(text: str, pathway: str) -> tuple[str, int]:
    """Return (sponsor text, character offset of the testing section or -1)."""
    if pathway == "510(k)":
        m = SUMMARY_START.search(text)
        if m:
            text = text[m.start():]
    text = re.sub(r"\n{3,}", "\n\n", text.replace("\f", "\n\n――― page break ―――\n\n")).strip()
    for m in TESTING_HEADING.finditer(text):
        if is_heading(m.group(0)):
            return text, m.start()
    return text, -1


def main() -> None:
    sample = pd.read_csv(SAMPLE, dtype=str).fillna("")
    for col in ("hand_validated", "notes"):
        if col not in sample:
            sample[col] = ""
    devices = []
    for d in sample.itertuples(index=False):
        path = TXT_DIR / f"{d.submission_number.replace('/', '')}.txt"
        text, anchor = prepare(path.read_text(), d.pathway) if path.exists() else ("(no text extracted)", -1)
        devices.append({
            "id": d.submission_number, "pathway": d.pathway, "date": d.decision_date,
            "name": d.device_name, "company": d.company, "panel": d.panel, "url": d.url,
            "before": text[:anchor] if anchor > 0 else "",
            "text": text[anchor:] if anchor >= 0 else text,
            "anchored": anchor >= 0,
            "code": d.hand_validated, "notes": d.notes,
        })
    columns = [c for c in sample.columns if c not in ("hand_validated", "notes")]
    rows = sample[columns].to_dict(orient="records")
    html = TEMPLATE.replace("__DATA__", json.dumps(devices).replace("</", "<\\/")) \
                   .replace("__ROWS__", json.dumps(rows).replace("</", "<\\/")) \
                   .replace("__COLUMNS__", json.dumps(columns))
    OUT.write_text(html)
    found = sum(d["anchored"] for d in devices)
    print(f"wrote {OUT} ({len(devices)} devices, testing section located in {found})")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Validation Coding</title>
<style>
:root {
  --bg: #f7f7f5; --panel: #ffffff; --text: #1d1d1b; --muted: #6b6b66; --border: #e2e2dc;
  --yes: #1f7a4d; --no: #b3261e; --unsure: #9a6700; --accent: #2f5bd3; --mark: #fff3bf;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #161615; --panel: #1f1f1d; --text: #ecece8; --muted: #a3a39c; --border: #34342f;
    --yes: #4cc38a; --no: #ff8a80; --unsure: #f2c94c; --accent: #8ab4ff; --mark: #3a3420;
  }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
header { position: sticky; top: 0; z-index: 2; background: var(--panel); border-bottom: 1px solid var(--border);
  padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: center; }
header h1 { font-size: 15px; margin: 0; }
.progress { flex: 1; min-width: 160px; height: 6px; background: var(--border); border-radius: 3px; overflow: hidden; }
.progress div { height: 100%; background: var(--accent); width: 0; }
.counts { color: var(--muted); font-size: 13px; }
button, select { font: inherit; font-size: 13px; padding: 5px 10px; border-radius: 6px; cursor: pointer;
  border: 1px solid var(--border); background: var(--panel); color: var(--text); }
main { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; padding: 16px; max-width: 1280px; margin: 0 auto; }
@media (max-width: 860px) { main { grid-template-columns: 1fr; } aside { order: -1; } }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 16px; }
.meta h2 { font-size: 18px; margin: 0 0 4px; }
.meta p { margin: 0; color: var(--muted); font-size: 13px; }
.meta a { color: var(--accent); }
.doc { margin-top: 12px; white-space: pre-wrap; overflow-wrap: anywhere; font: 14px/1.6 ui-monospace, Menlo, monospace;
  max-height: calc(100vh - 230px); overflow-y: auto; border-top: 1px solid var(--border); padding-top: 12px; }
.doc .before { color: var(--muted); }
.anchor-note { font-size: 12px; color: var(--muted); margin: 8px 0 0; }
.show-before { margin-top: 8px; }
aside .card + .card { margin-top: 12px; }
.codes { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; }
.codes button { padding: 12px 0; font-size: 15px; font-weight: 600; }
.codes button kbd { display: block; font-size: 11px; font-weight: 400; color: var(--muted); }
.codes button.on[data-code="yes"] { background: var(--yes); color: #fff; border-color: var(--yes); }
.codes button.on[data-code="no"] { background: var(--no); color: #fff; border-color: var(--no); }
.codes button.on[data-code="unsure"] { background: var(--unsure); color: #fff; border-color: var(--unsure); }
.codes button.on kbd { color: inherit; }
textarea { width: 100%; min-height: 70px; margin-top: 10px; font: inherit; font-size: 13px; padding: 8px;
  border-radius: 6px; border: 1px solid var(--border); background: var(--bg); color: var(--text); }
.nav { display: flex; gap: 8px; margin-top: 10px; }
.nav button { flex: 1; }
.rule ol { margin: 6px 0 0; padding-left: 20px; font-size: 13px; }
.rule h3, .keys h3 { font-size: 13px; margin: 0; }
.rule .ans { font-weight: 600; }
.keys { font-size: 12px; color: var(--muted); }
.keys kbd { border: 1px solid var(--border); border-radius: 4px; padding: 0 4px; font-size: 11px; }
.saved { font-size: 12px; color: var(--muted); }
.saved.warn { color: var(--no); font-weight: 600; }
.links { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; }
.toast { position: fixed; bottom: 16px; left: 50%; transform: translateX(-50%); background: var(--text); color: var(--bg);
  padding: 8px 14px; border-radius: 6px; font-size: 13px; opacity: 0; transition: opacity .2s; pointer-events: none; }
.toast.show { opacity: 1; }
</style>
</head>
<body>
<header>
  <h1>Validation coding</h1>
  <div class="progress"><div id="bar"></div></div>
  <span class="counts" id="counts"></span>
  <select id="filter" title="Which devices to step through">
    <option value="all">All devices</option>
    <option value="todo">Not coded yet</option>
    <option value="unsure">Unsure only</option>
  </select>
  <span class="saved" id="saved"></span>
  <button id="linkFile" title="Pick a CSV file; every code is then written to it automatically">Auto-save to file…</button>
  <button id="import" title="Load codes from a previously exported CSV">Import CSV</button>
  <button id="export">Export CSV</button>
  <input type="file" id="importFile" accept=".csv,text/csv" hidden>
</header>
<main>
  <section class="card">
    <div class="meta">
      <h2 id="name"></h2>
      <p id="info"></p>
      <p class="links"><a id="pdf" target="_blank" rel="noopener">Open full PDF ↗</a>
        <button id="copy" title="Copy the text shown below (C)">Copy text</button></p>
    </div>
    <p class="anchor-note" id="anchorNote"></p>
    <button class="show-before" id="showBefore">Show text above the testing section</button>
    <div class="doc" id="doc"></div>
  </section>
  <aside>
    <div class="card">
      <div class="codes">
        <button data-code="yes">Yes<kbd>Y</kbd></button>
        <button data-code="no">No<kbd>N</kbd></button>
        <button data-code="unsure">Unsure<kbd>U</kbd></button>
      </div>
      <textarea id="notes" placeholder="Notes (optional): a few words quoted from the summary"></textarea>
      <div class="nav">
        <button id="prev">← Back</button>
        <button id="next">Next →</button>
      </div>
    </div>
    <div class="card rule">
      <h3>Does the summary report testing on patients or patient data?</h3>
      <ol>
        <li>Tested on <b>real patient data</b> (scans, signals, records, enrolled patients)? If not, the answer is <span class="ans">No</span>.</li>
        <li>Reports a <b>result</b> (sensitivity, accuracy, agreement, Dice, met acceptance criteria)? If not, the answer is <span class="ans">No</span>.</li>
        <li>Result is from <b>this</b> submission, not "unchanged from K######"? If so, the answer is <span class="ans">Yes</span>.</li>
      </ol>
      <p style="font-size:12px;color:var(--muted);margin:8px 0 0">Bench, phantom, simulated or software V&amp;V only: No.
      Clinicians only glance at "sample images" with no endpoint: No. Can't decide in a minute: Unsure.</p>
    </div>
    <div class="card keys">
      <h3>Keys</h3>
      <kbd>Y</kbd> <kbd>N</kbd> <kbd>U</kbd> code and go to next &nbsp;·&nbsp;
      <kbd>←</kbd> <kbd>→</kbd> back / next &nbsp;·&nbsp; <kbd>O</kbd> open PDF &nbsp;·&nbsp;
      <kbd>C</kbd> copy text &nbsp;·&nbsp; <kbd>Space</kbd> scroll text &nbsp;·&nbsp; <kbd>Esc</kbd> leave notes box.
      Codes save automatically in this browser after every key press. For a copy on disk,
      use "Auto-save to file…" (Chrome/Edge) or Export CSV now and then.
    </div>
  </aside>
</main>
<div class="toast" id="toast"></div>
<script>
const DEVICES = __DATA__;
const ROWS = __ROWS__;
const COLUMNS = __COLUMNS__;
const KEY = "validation-coding-v1";

let saved = {};
try { saved = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) { saved = {}; }
for (const d of DEVICES) {
  if (!saved[d.id]) saved[d.id] = { code: d.code || "", notes: d.notes || "" };
}
let i = 0;
try { i = Math.min(+localStorage.getItem(KEY + "-pos") || 0, DEVICES.length - 1); } catch (e) {}

const $ = id => document.getElementById(id);
let fileHandle = null, writing = Promise.resolve();
function stamp() { return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
function persist() {
  const el = $("saved");
  try {
    localStorage.setItem(KEY, JSON.stringify(saved));
    localStorage.setItem(KEY + "-pos", i);
    if (localStorage.getItem(KEY) !== JSON.stringify(saved)) throw new Error("read-back mismatch");
    el.className = "saved";
    el.textContent = `Saved in browser ${stamp()}` + (fileHandle ? ` + ${fileHandle.name}` : "");
  } catch (e) {
    el.className = "saved warn";
    el.textContent = "Browser is NOT saving. Use Export CSV now.";
  }
  if (fileHandle) writing = writing.then(writeFile).catch(err => {
    el.className = "saved warn"; el.textContent = "File save failed: " + err.message; fileHandle = null;
  });
}
async function writeFile() {
  const w = await fileHandle.createWritable();
  await w.write(buildCsv());
  await w.close();
}
function toast(msg) { const t = $("toast"); t.textContent = msg; t.classList.add("show"); setTimeout(() => t.classList.remove("show"), 1400); }

function visible(k) {
  const f = $("filter").value, c = saved[DEVICES[k].id].code;
  return f === "all" || (f === "todo" && !c) || (f === "unsure" && c === "unsure");
}
function step(dir) {
  for (let k = i + dir; k >= 0 && k < DEVICES.length; k += dir) {
    if (visible(k)) { i = k; render(); return true; }
  }
  return false;
}

function render() {
  const d = DEVICES[i], s = saved[d.id];
  $("name").textContent = `${i + 1}. ${d.name}`;
  $("info").textContent = `${d.id} · ${d.pathway} · ${d.date} · ${d.company} · ${d.panel}`;
  $("pdf").href = d.url || "#";
  $("anchorNote").textContent = d.anchored
    ? "Opened at the first testing-section heading. Scroll up via the button if you need context."
    : "No testing heading found: showing the whole summary.";
  $("showBefore").style.display = d.before ? "" : "none";
  $("showBefore").textContent = "Show text above the testing section";
  const doc = $("doc");
  doc.textContent = d.text;
  doc.scrollTop = 0;
  $("notes").value = s.notes;
  document.querySelectorAll(".codes button").forEach(b => b.classList.toggle("on", b.dataset.code === s.code));
  const vals = Object.values(saved);
  const done = vals.filter(v => v.code).length;
  const n = c => vals.filter(v => v.code === c).length;
  $("bar").style.width = (100 * done / DEVICES.length) + "%";
  $("counts").textContent = `${done}/${DEVICES.length} coded · yes ${n("yes")} · no ${n("no")} · unsure ${n("unsure")}`;
  persist();
}

function setCode(code) {
  saved[DEVICES[i].id].code = code;
  persist();
  if (!step(1)) { render(); toast($("filter").value === "all" ? "Last device. Export when done." : "No more in this filter."); }
}

$("showBefore").onclick = () => {
  const d = DEVICES[i], doc = $("doc");
  if (doc.querySelector(".before")) return;
  const span = document.createElement("span");
  span.className = "before";
  span.textContent = d.before + "\n\n――― testing section starts here ―――\n\n";
  doc.prepend(span);
  $("showBefore").textContent = "Context shown above (greyed)";
};
document.querySelectorAll(".codes button").forEach(b => b.onclick = () => setCode(b.dataset.code));
$("prev").onclick = () => step(-1) || toast("No earlier device in this filter.");
$("next").onclick = () => step(1) || toast("No later device in this filter.");
$("notes").oninput = e => { saved[DEVICES[i].id].notes = e.target.value; persist(); };
$("filter").onchange = () => { if (!visible(i)) step(1) || step(-1); render(); };

document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") { if (e.key === "Escape") e.target.blur(); return; }
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "y") setCode("yes");
  else if (k === "n") setCode("no");
  else if (k === "u") setCode("unsure");
  else if (k === "arrowright") { e.preventDefault(); $("next").click(); }
  else if (k === "arrowleft") { e.preventDefault(); $("prev").click(); }
  else if (k === "o") window.open(DEVICES[i].url, "_blank", "noopener");
  else if (k === "c") copyText();
  else if (k === " ") { e.preventDefault(); $("doc").scrollBy({ top: $("doc").clientHeight * 0.85 * (e.shiftKey ? -1 : 1) }); }
});

function buildCsv() {
  const cols = [...COLUMNS, "hand_validated", "notes"];
  const esc = v => { v = String(v ?? ""); return /[",\n]/.test(v) ? '"' + v.replace(/"/g, '""') + '"' : v; };
  const lines = [cols.join(",")];
  for (const r of ROWS) {
    const s = saved[r.submission_number] || {};
    lines.push(cols.map(c => esc(c === "hand_validated" ? s.code : c === "notes" ? s.notes : r[c])).join(","));
  }
  return lines.join("\n") + "\n";
}

function parseCsv(text) {
  const rows = []; let row = [], field = "", q = false;
  for (let k = 0; k < text.length; k++) {
    const ch = text[k];
    if (q) {
      if (ch === '"' && text[k + 1] === '"') { field += '"'; k++; }
      else if (ch === '"') q = false;
      else field += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { row.push(field); field = ""; }
    else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[k + 1] === "\n") k++;
      row.push(field); rows.push(row); row = []; field = "";
    } else field += ch;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  return rows.filter(r => r.some(x => x !== ""));
}

$("import").onclick = () => $("importFile").click();
$("importFile").onchange = async e => {
  const f = e.target.files[0]; if (!f) return;
  const rows = parseCsv(await f.text()), head = rows.shift() || [];
  const id = head.indexOf("submission_number"), code = head.indexOf("hand_validated"), notes = head.indexOf("notes");
  if (id < 0 || code < 0) { toast("That CSV has no submission_number / hand_validated columns."); return; }
  let n = 0;
  for (const r of rows) {
    if (saved[r[id]] && r[code]) { saved[r[id]].code = r[code]; saved[r[id]].notes = notes >= 0 ? r[notes] : ""; n++; }
  }
  render(); toast(`Imported ${n} codes`); e.target.value = "";
};

$("linkFile").onclick = async () => {
  if (!window.showSaveFilePicker) { toast("This browser can't auto-save to a file; use Chrome/Edge, or Export CSV."); return; }
  try {
    fileHandle = await window.showSaveFilePicker({ suggestedName: "hand_coded_sample.csv",
      types: [{ description: "CSV", accept: { "text/csv": [".csv"] } }] });
    persist(); toast("Every code is now also written to " + fileHandle.name);
  } catch (e) { if (e.name !== "AbortError") toast("Could not open file: " + e.message); }
};

async function copyText() {
  const text = `${DEVICES[i].id} | ${DEVICES[i].name}\n\n` + $("doc").textContent;
  try { await navigator.clipboard.writeText(text); }
  catch (e) {
    const ta = document.createElement("textarea");
    ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.appendChild(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  toast("Text copied");
}
$("copy").onclick = copyText;

window.addEventListener("beforeunload", persist);

$("export").onclick = () => {
  const blob = new Blob([buildCsv()], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "hand_coded_sample.csv";
  a.click();
  toast("Saved hand_coded_sample.csv to your downloads");
};

render();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
