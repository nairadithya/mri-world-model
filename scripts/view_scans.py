#!/usr/bin/env python3
"""Quick local NIfTI explorer over a scan tree (browser, client-side NiiVue).

Serves a small UI on 127.0.0.1 that renders NIfTI in the browser via NiiVue.
Nothing is uploaded: the NIfTI bytes go host -> your own browser, and NiiVue
renders them client-side. The only network fetch is the NiiVue JS bundle,
downloaded once into a local cache (override with --vendor).

Supported layouts (auto-detected):
  SAILOR derivatives : <root>/sub-XX/ses-YY/*.nii.gz   (e.g. sub-23/ses-05/T2)
  LUMIERE preprocess : <root>/Patient-XXX/week-*/*.nii.gz
  any tree with       <patient>/<session>/*.nii(.gz)

Base modalities (click = load) are recognized by name; everything else loads
as an overlay (masks, segmentations, -icor/-zscore variants).

Usage:
    python scripts/view_scans.py                      # SAILOR derivatives
    python scripts/view_scans.py --root data/lumiere_preprocessed
    python scripts/view_scans.py --root data/sailor_reprocessed_bet --port 8770
    # remote box: ssh -L 8765:127.0.0.1:8765 <host>, then browse localhost:8765
    # deep-link / scripting (repeat ov for multiple overlays):
    #   /?p=sub-01/ses-05/T1c.nii.gz&ov=sub-01/ses-05/EdemaMask-ONCO.nii.gz

Note: SAILOR is controlled-access. Keep this on localhost; do not port-forward
beyond a trusted tunnel or upload the files anywhere.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import posixpath
import threading
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SAILOR_ROOT = "data/sailor/sailor_ebrains_pseud/derivatives/mni2009c-n-s"
NIIVUE_VERSION = "0.69.0"
NIIVUE_URL = (f"https://cdn.jsdelivr.net/npm/@niivue/niivue@{NIIVUE_VERSION}"
              "/dist/niivue.umd.js")
BASE_NAMES = {"T1c", "CT1", "T1", "T2", "Flair", "FLAIR", "t1", "t2"}
OVERLAY_COLORMAPS = ["red", "green", "blue", "warm", "cool", "jet", "viridis"]


def _scan_name(name: str) -> str:
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


SAILOR_RANO = {1: "PD", 2: "SD", 3: "PR", 5: "CR"}
SAILOR_TREATMENT = {"CRT": "CRT", "TMZ": "TMZ", "no": "none", "unknown": "unknown"}


def _read_text(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def load_lumiere_meta(meta_dir: str | None):
    """LUMIERE sidecar tables: patient demographics, RANO, per-timepoint
    modality completeness (keyed by patient / (patient, timepoint))."""
    demo: dict[str, dict] = {}
    rano: dict[tuple, dict] = {}
    comp: dict[tuple, str] = {}
    if not meta_dir or not os.path.isdir(meta_dir):
        return demo, rano, comp

    def find(sub):
        import glob
        g = glob.glob(os.path.join(meta_dir, sub))
        return g[0] if g else None

    p = find("*emographics*.csv")
    if p:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                def g(k):
                    return (r.get(k) or "").strip()
                m = {}
                if g("Age at surgery (years)"):
                    m["Age"] = g("Age at surgery (years)") + " y"
                if g("Sex"):
                    m["Sex"] = g("Sex")
                if g("IDH (WT: wild type)"):
                    m["IDH"] = g("IDH (WT: wild type)")
                if g("MGMT qualitative"):
                    m["MGMT"] = g("MGMT qualitative")
                if g("Survival time (weeks)"):
                    m["Survival"] = g("Survival time (weeks)") + " wk"
                demo[r["Patient"]] = m
    p = find("*rating*.csv")
    if p:
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            rk = next((k for k in (rd.fieldnames or []) if k.startswith("Rating (")), None)
            ak = next((k for k in (rd.fieldnames or []) if k.startswith("Rating rationale")), None)
            for r in rd:
                m = {}
                if rk and (r.get(rk) or "").strip():
                    m["RANO"] = r[rk].strip()
                if ak and (r.get(ak) or "").strip():
                    m["Rationale"] = r[ak].strip()
                rano[(r["Patient"], (r.get("Date") or "").strip())] = m
    p = find("*completeness*.csv")
    if p:
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                have = [m for m in ("CT1", "T1", "T2", "FLAIR")
                        if (r.get(m) or "").strip()]
                comp[(r["Patient"], (r.get("Timepoint") or "").strip())] = ", ".join(have)
    return demo, rano, comp


def build_index(root: str, meta_dir: str | None = None) -> dict:
    demo, rano, comp = load_lumiere_meta(meta_dir)
    patients = []
    for pdir in sorted(os.listdir(root)):
        ppath = os.path.join(root, pdir)
        if not os.path.isdir(ppath) or pdir.startswith("."):
            continue
        pmeta: dict[str, str] = {}
        age = _read_text(os.path.join(ppath, "age-years.txt"))
        osurv = _read_text(os.path.join(ppath, "overall-survival-months.txt"))
        gaps = _read_text(os.path.join(ppath, "intervals-days.txt"))
        if age:
            pmeta["Age"] = f"{float(age):.1f} y"
        if osurv:
            pmeta["OS"] = f"{float(osurv):.1f} mo"
        if gaps:
            pmeta["Gaps"] = " / ".join(gaps.split()) + " d"
        pmeta.update(demo.get(pdir, {}))

        sessions = []
        subs = sorted(d for d in os.listdir(ppath)
                      if os.path.isdir(os.path.join(ppath, d)))
        # flat patient dir (no session level): treat the dir itself as a session
        if not subs:
            subs = ["."]
        for sdir in subs:
            spath = os.path.join(ppath, sdir)
            files = []
            for f in sorted(os.listdir(spath)):
                if not (f.endswith(".nii") or f.endswith(".nii.gz")):
                    continue
                stem = _scan_name(f)
                kind = "base" if stem in BASE_NAMES else "overlay"
                rel = os.path.relpath(os.path.join(spath, f), root)
                files.append({"name": stem, "rel": rel, "kind": kind})
            smeta: dict[str, str] = {}
            code = _read_text(os.path.join(spath, "RANO.txt"))
            if code:
                try:
                    smeta["RANO"] = SAILOR_RANO.get(int(code.split()[0]), code)
                except ValueError:
                    smeta["RANO"] = code
            treat = _read_text(os.path.join(spath, "treatment.txt"))
            if treat:
                smeta["Treatment"] = SAILOR_TREATMENT.get(treat.split()[0], treat)
            smeta.update(rano.get((pdir, sdir), {}))
            if (pdir, sdir) in comp:
                smeta["Modalities"] = comp[(pdir, sdir)]
            if files:
                sessions.append({"id": pdir if sdir == "." else sdir,
                                 "meta": smeta, "files": files})
        if sessions:
            patients.append({"id": pdir, "meta": pmeta, "sessions": sessions})
    return {"root": os.path.abspath(root), "patients": patients}


PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>Scan explorer</title>
<style>
:root{--bg:#11151c;--panel:#1a212b;--edge:#2b3542;--fg:#d7e0ea;--mut:#8b98a8;--acc:#4ea1ff}
*{box-sizing:border-box} html,body{margin:0;height:100%;background:var(--bg);color:var(--fg);
font:13px/1.4 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;overflow:hidden}
#app{display:grid;grid-template-columns:320px 1fr;height:100vh}
#side{background:var(--panel);border-right:1px solid var(--edge);display:flex;flex-direction:column;min-width:0}
#side h1{font-size:14px;margin:0;padding:12px 14px;border-bottom:1px solid var(--edge)}
#side h1 small{color:var(--mut);font-weight:400;display:block;font-size:11px;margin-top:2px;
white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#searchwrap{padding:8px 10px;border-bottom:1px solid var(--edge)}
#search{width:100%;background:var(--bg);border:1px solid var(--edge);color:var(--fg);
border-radius:6px;padding:6px 8px;outline:none}
#tree{overflow:auto;flex:1;padding:4px 6px 20px}
#meta{max-height:40%;overflow:auto;border-top:1px solid var(--edge);padding:9px 11px;font-size:12px}
.mhead{color:var(--mut);text-transform:uppercase;letter-spacing:.06em;font-size:10px;margin-bottom:6px}
.msec{margin-bottom:9px}.mtitle{color:var(--acc);font-weight:600;margin-bottom:2px}
.mrow{display:flex;justify-content:space-between;gap:10px;padding:1px 0}
.mrow span{color:var(--mut)}
.mrow b{font-weight:500;text-align:right;max-width:64%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mut{color:var(--mut)}
details{margin:1px 0} details>summary{cursor:pointer;padding:4px 6px;border-radius:5px;color:var(--mut)}
details>summary:hover{background:#202a36}
summary::marker{color:var(--mut)}
.file{display:flex;align-items:center;gap:6px;padding:3px 8px;margin-left:14px;border-radius:5px;
cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.file:hover{background:#223046}.file.base{color:#cfe6ff}.file.ov{color:#a9b4c0}
.dot{width:7px;height:7px;border-radius:50%;flex:0 0 auto}
.dot.base{background:var(--acc)}.dot.ov{background:#c0603f}
#main{display:flex;flex-direction:column;min-width:0}
#bar{display:flex;align-items:center;gap:14px;padding:8px 12px;border-bottom:1px solid var(--edge);
background:var(--panel);flex-wrap:wrap}
#bar label{color:var(--mut);display:flex;align-items:center;gap:6px}
select,input[type=range]{background:var(--bg);color:var(--fg);border:1px solid var(--edge);border-radius:5px}
button{background:#243040;color:var(--fg);border:1px solid var(--edge);border-radius:6px;padding:5px 10px;cursor:pointer}
button:hover{background:#2c3a4c}
#viewer{flex:1;min-height:0;position:relative;background:#05070a}
#gl{position:absolute;inset:0;width:100%;height:100%;display:block}
#hud{position:absolute;left:8px;bottom:8px;color:var(--mut);font-size:11px;pointer-events:none;
background:rgba(10,13,18,.6);padding:3px 7px;border-radius:5px}
.pill{color:var(--mut);font-size:11px;border:1px solid var(--edge);border-radius:999px;padding:2px 8px}
</style></head><body>
<div id="app">
 <div id="side">
  <h1>Scan explorer<small id="rootlabel"></small></h1>
  <div id="searchwrap"><input id="search" placeholder="filter patients / sessions / files…"></div>
  <div id="tree"></div>
  <div id="meta"><div class="mhead">Metadata</div>
   <div id="metabody"><div class="mut">select a scan</div></div></div>
 </div>
 <div id="main">
  <div id="bar">
   <label>colormap <select id="cmap">__CMAPS__</select></label>
   <label>overlay opacity <input id="op" type="range" min="0" max="1" step="0.05" value="0.45"></label>
   <button id="clear">clear overlays</button>
   <button id="reset">reset view</button>
   <span class="pill" id="status">load a scan</span>
  </div>
  <div id="viewer"><canvas id="gl"></canvas><div id="hud"></div></div>
 </div>
</div>
<script src="/vendor/niivue.umd.js"></script>
<script>
window.__INDEX__ = @@INDEX@@;
const nv = new niivue.Niivue({backColor:[0.03,0.05,0.08,1], show3Dcrosshair:true,
  isColorbar:false, isRadiologicalConvention:false});
const ready=nv.attachTo('gl');   // async: volumes added before it resolves get wiped
const state={base:null, overlays:[]};
const $=id=>document.getElementById(id);
$('rootlabel').textContent=window.__INDEX__.root;

function itemUrl(rel){return '/nii/'+rel.split('/').map(encodeURIComponent).join('/');}
function styleOverlays(){
  nv.volumes.forEach((vol,i)=>{if(i===0)return;
    vol.colormap=$('cmap').value; vol.opacity=parseFloat($('op').value);});
  nv.updateGLVolume();
}
function clearVolumes(){while(nv.volumes.length)nv.removeVolume(nv.volumes[nv.volumes.length-1]);}
async function refresh(){
  const list=[];
  if(state.base) list.push({url:itemUrl(state.base.rel),name:state.base.name,colormap:'gray',opacity:1});
  for(const o of state.overlays)
    list.push({url:itemUrl(o.rel),name:o.name,colormap:$('cmap').value,opacity:parseFloat($('op').value)});
  // NiiVue 0.69: loadVolumes APPENDS, and extra option fields break its
  // getFileExt(name||url) — so clear, load by URL only, then style.
  clearVolumes();
  await nv.loadVolumes(list.map(v=>({url:v.url})));
  nv.volumes.forEach((vol,i)=>{const s=list[i]; if(s){
    vol.name=s.name; vol.colormap=s.colormap; vol.opacity=s.opacity;}});
  nv.updateGLVolume();
  $('status').textContent=(state.base?state.base.name:'—')+
    (state.overlays.length?('  +'+state.overlays.length+' overlay'+(state.overlays.length>1?'s':'')):'');
  $('hud').textContent=state.base?('base '+state.base.name):'';
}
let queue=Promise.resolve();
const run=fn=>{queue=queue.then(fn,fn);return queue;};   // serialize refreshes
function pick(rel,name,kind){
  const parts=rel.split('/');
  if(parts.length>=2)showMeta(parts[0],parts[1],name);
  if(kind==='base'){state.base={rel,name};state.overlays=[];}
  else{state.overlays=state.overlays.filter(o=>o.rel!==rel);state.overlays.push({rel,name});}
  return run(refresh);
}
$('clear').onclick=()=>{state.overlays=[];run(refresh);};
$('reset').onclick=()=>{nv.setSliceType(niivue.SLICE_TYPE.MULTIPLANAR);};
$('cmap').onchange=()=>styleOverlays();
$('op').oninput=()=>styleOverlays();

const tree=$('tree');
function render(filter){
  tree.innerHTML=''; const f=(filter||'').toLowerCase();
  for(const p of window.__INDEX__.patients){
    const ses=p.sessions.map(s=>({s,fs:s.files.filter(x=>
      !f||p.id.toLowerCase().includes(f)||s.id.toLowerCase().includes(f)||x.name.toLowerCase().includes(f))}))
      .filter(x=>x.fs.length);
    if(!ses.length) continue;
    const d=document.createElement('details');
    const sum=document.createElement('summary');
    const n=ses.reduce((a,x)=>a+x.fs.length,0);
    const pm=p.meta||{};
    const ptag=[pm.Age,pm.OS,pm.Survival].filter(Boolean).join(' · ');
    sum.textContent=p.id+(ptag?' · '+ptag:'')+'  ('+ses.length+' ses, '+n+' files)';
    sum.onclick=()=>showMeta(p.id);
    d.appendChild(sum);
    if(f) d.open=true;
    for(const {s,fs} of ses){
      const sd=document.createElement('details'); sd.open=!!f;
      const ss=document.createElement('summary');
      const sm=s.meta||{};
      const stag=[sm.RANO,sm.Treatment].filter(Boolean).join(' · ');
      ss.textContent=s.id+(stag?' · '+stag:'');
      ss.onclick=()=>showMeta(p.id,s.id);
      sd.appendChild(ss);
      for(const x of fs){
        const el=document.createElement('div');
        el.className='file '+(x.kind==='base'?'base':'ov');
        el.title=x.rel;
        el.innerHTML='<span class="dot '+x.kind+'"></span>'+x.name;
        el.onclick=()=>pick(x.rel,x.name,x.kind);
        sd.appendChild(el);
      }
      d.appendChild(sd);
    }
    tree.appendChild(d);
  }
}

function metaRows(o){
  const e=Object.entries(o||{});
  return e.length?e.map(([k,v])=>{
    const t=String(v==null?'':v).replace(/"/g,'&quot;');
    return '<div class="mrow"><span>'+k+'</span><b title="'+t+'">'+t+'</b></div>';
  }).join(''):'<div class="mut">none</div>';
}
function showMeta(pid,sid,file){
  const p=window.__INDEX__.patients.find(x=>x.id===pid)||{};
  const s=(p.sessions||[]).find(x=>x.id===sid)||{};
  let h='<div class="msec"><div class="mtitle">'+pid+'</div>'+metaRows(p.meta)+'</div>';
  if(sid) h+='<div class="msec"><div class="mtitle">'+sid+'</div>'+metaRows(s.meta)+'</div>';
  if(file) h+='<div class="msec"><div class="mtitle">file</div>'+
    '<div class="mrow"><span>name</span><b>'+file+'</b></div></div>';
  $('metabody').innerHTML=h;
}
render('');
$('search').oninput=e=>render(e.target.value);

// deep-link / scripting: ?p=<rel>[&ov=<rel>,<rel>]  (e.g. ?p=sub-23/ses-05/T2.nii.gz)
function findByRel(rel){for(const p of window.__INDEX__.patients)
  for(const s of p.sessions)for(const f of s.files)if(f.rel===rel)return f;return null;}
ready.then(()=>{
  nv.setSliceType(niivue.SLICE_TYPE.MULTIPLANAR);
  const q=new URLSearchParams(location.search);
  if(q.get('p')){const f=findByRel(q.get('p'));if(f)pick(f.rel,f.name,f.kind);}
  const ovs=[];for(const v of q.getAll('ov'))ovs.push(...v.split(','));
  for(const r of ovs.filter(Boolean)){const f=findByRel(r);if(f)pick(f.rel,f.name,'overlay');}
});
</script></body></html>"""


def make_handler(root: str, vendor: str, meta_dir: str | None = None):
    root = os.path.abspath(root)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body: bytes, ctype="text/plain"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                idx = build_index(root, meta_dir)
                page = PAGE.replace("@@INDEX@@", json.dumps(idx))
                page = page.replace("__CMAPS__",
                                    "".join(f'<option>{c}</option>'
                                            for c in OVERLAY_COLORMAPS))
                return self._send(200, page.encode(), "text/html; charset=utf-8")
            if path == "/favicon.ico":
                return self._send(204, b"", "image/x-icon")
            if path == "/vendor/niivue.umd.js":
                try:
                    with open(vendor, "rb") as f:
                        return self._send(200, f.read(), "application/javascript")
                except OSError:
                    return self._send(500, b"vendor missing", "text/plain")
            if path.startswith("/nii/"):
                rel = posixpath.normpath(urllib.parse.unquote(path[len("/nii/"):]))
                if rel.startswith("..") or rel.startswith("/"):
                    return self._send(403, b"forbidden")
                full = os.path.abspath(os.path.join(root, rel))
                if os.path.commonpath([full, root]) != root:
                    return self._send(403, b"forbidden")
                if not os.path.isfile(full):
                    return self._send(404, b"not found")
                ctype = "application/gzip" if full.endswith(".gz") else "application/octet-stream"
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(os.path.getsize(full)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                with open(full, "rb") as f:
                    while True:
                        chunk = f.read(1 << 20)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                return
            return self._send(404, b"not found")

    return Handler


def ensure_vendor(path: str) -> bool:
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        print(f"fetching NiiVue {NIIVUE_VERSION} -> {path}")
        urllib.request.urlretrieve(NIIVUE_URL, path)
        return os.path.getsize(path) > 100_000
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not fetch NiiVue ({e}); the page will try the CDN")
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=SAILOR_ROOT)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--vendor", default=os.path.expanduser(
        "~/.cache/world-model-viewer/niivue.umd.js"))
    ap.add_argument("--meta-dir", default="data/lumiere_meta",
                    help="LUMIERE sidecar CSVs to join onto LUMIERE patients "
                         "(demographics/RANO/completeness)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"root not found: {args.root}")
        return 2
    ok = ensure_vendor(args.vendor)
    if not ok:
        print("(continuing; page will fall back to the CDN)")
    idx = build_index(args.root, args.meta_dir)
    n_files = sum(len(s["files"]) for p in idx["patients"] for s in p["sessions"])
    n_meta = sum(1 for p in idx["patients"] if p["meta"])
    print(f"root : {os.path.abspath(args.root)}")
    print(f"index: {len(idx['patients'])} patients, {n_files} scans, "
          f"{n_meta} with metadata")

    handler = make_handler(args.root, args.vendor, args.meta_dir)
    httpd = ThreadingHTTPServer((args.host, args.port), handler)
    httpd.daemon_threads = True
    url = f"http://{args.host}:{args.port}/"
    print(f"serve: {url}   (Ctrl-C to stop)")
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
