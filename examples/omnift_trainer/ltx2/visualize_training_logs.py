# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a self-contained HTML dashboard from verl console training logs."""

import argparse
import csv
import html
import json
import math
import re
from pathlib import Path

ANSI_RE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
STEP_RE = re.compile(r"(?:^|\s)step:(\d+)\s+-\s+(.*)")


def reward_metrics(name: str) -> tuple[str, ...]:
    """Return train/validation mean, std, min, and max metric names for one reward."""
    return tuple(
        f"{split}/reward/{name}/{statistic}"
        for split in ("train", "val")
        for statistic in ("mean", "std", "min", "max")
    )


PANELS = (
    ("Reward 总分", reward_metrics("sum")),
    ("VideoAlign", reward_metrics("video_align")),
    ("HPSv3", reward_metrics("hpsv3")),
    ("AudioBox", reward_metrics("audiobox")),
    ("CLAP", reward_metrics("clap")),
    ("DeSync", reward_metrics("desync")),
    ("Actor Loss", ("actor/video/policy_loss", "actor/audio/policy_loss", "actor/total_loss")),
    ("Policy 偏移", ("actor/video/old_deviate", "actor/audio/old_deviate")),
    ("Reference KL", ("actor/video/ref_kl_loss", "actor/audio/ref_kl_loss")),
    ("Reward Probability", ("actor/video/reward_prob_mean", "actor/audio/reward_prob_mean")),
    ("优化器", ("actor/grad_norm",)),
    ("耗时（秒）", ("timing_s/gen", "timing_s/reward", "timing_s/update_actor", "timing_s/step")),
)

HTML_TEMPLATE = r"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--panel:#fff;--ink:#1f2937;--muted:#687386;--line:#e4e7ec;--accent:#654ff0}
[data-theme=dark]{--bg:#111318;--panel:#1b1e25;--ink:#edf0f7;--muted:#9ca6b7;--line:#303540}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px Inter,system-ui,sans-serif}
header{height:62px;position:sticky;top:0;z-index:4;display:flex;align-items:center;gap:14px;padding:0 22px;
background:var(--panel);border-bottom:1px solid var(--line)}header h1{font-size:18px;margin:0}header .meta{color:var(--muted)}
button,input{font:inherit;color:inherit}.icon,.primary{border:1px solid var(--line);background:var(--panel);border-radius:7px;
padding:7px 10px;cursor:pointer}.primary{color:#fff;background:var(--accent);border-color:var(--accent)}
.layout{display:grid;grid-template-columns:245px 1fr;min-height:calc(100vh - 62px)}aside{padding:20px 16px;border-right:1px solid var(--line);
background:var(--panel)}aside h2{font-size:12px;text-transform:uppercase;color:var(--muted);letter-spacing:.08em;margin:20px 0 10px}
aside h2:first-child{margin-top:0}.run{display:flex;align-items:flex-start;gap:8px;margin:9px 0}.run span{overflow-wrap:anywhere}
.control{width:100%;accent-color:var(--accent)}.value{float:right;color:var(--accent);font-weight:650}
.add{display:flex;gap:6px}.add input{min-width:0;width:100%;padding:7px;border:1px solid var(--line);border-radius:7px;
background:var(--bg)}aside .wide{width:100%;margin-top:9px}main{padding:22px;min-width:0}.summary{display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-bottom:18px}.summary article,.chart{background:var(--panel);
border:1px solid var(--line);border-radius:10px}.summary article{padding:14px}.summary h3{font-size:14px;margin:0 0 12px}
dl{display:grid;grid-template-columns:1fr auto;gap:6px;margin:0}dt{color:var(--muted)}dd{margin:0;font-variant-numeric:tabular-nums}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(510px,1fr));gap:14px}.chart{min-width:0;padding:14px}
.chart-head{display:flex;align-items:center;justify-content:space-between;gap:8px}.chart h2{font-size:15px;margin:0}.actions{display:flex;gap:5px}
.actions button{padding:3px 7px;border:0;background:transparent;color:var(--muted);cursor:pointer}.canvas-wrap{position:relative}
canvas{display:block;width:100%;height:295px;cursor:crosshair}.legend{display:flex;flex-wrap:wrap;gap:5px 12px;min-height:24px}
.legend button{display:flex;align-items:center;gap:5px;padding:2px;border:0;background:none;color:var(--muted);font-size:11px;
cursor:pointer}.legend button.off{opacity:.35}.swatch{width:14px;height:3px;border-radius:2px}.tip{position:fixed;z-index:9;display:none;
pointer-events:none;background:#151923ee;color:#fff;border:1px solid #414858;border-radius:7px;padding:9px 11px;
font-size:12px;box-shadow:0 8px 24px #0004;max-width:420px}.tip b{display:block;margin-bottom:5px}.tip-row{display:grid;
grid-template-columns:10px minmax(110px,1fr) auto;gap:7px;align-items:center;margin:3px 0}.dot{width:8px;height:8px;border-radius:50%}
.empty{padding:60px;text-align:center;color:var(--muted)}@media(max-width:800px){.layout{grid-template-columns:1fr}
aside{border-right:0;border-bottom:1px solid var(--line)}main{padding:12px}.charts{grid-template-columns:1fr}}
</style></head><body>
<header><h1 id="title"></h1><span class="meta" id="meta"></span><span style="flex:1"></span>
<button class="icon" id="theme">深色</button></header>
<div class="layout"><aside><h2>Runs</h2><div id="runs"></div><h2>平滑</h2>
<label>移动平均 <span class="value" id="smooth-value"></span></label><input class="control" id="smooth" type="range" min="1" max="30">
<h2>添加图表</h2><div class="add"><input id="metric" list="metric-list" placeholder="搜索任意 metric">
<datalist id="metric-list"></datalist><button class="primary" id="add">添加</button></div>
<button class="icon wide" id="reset">重置全部缩放</button></aside>
<main><div class="summary" id="summary"></div><div class="charts" id="charts"></div></main></div>
<div class="tip" id="tip"></div><script id="payload" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById("payload").textContent);
const palette=["#654ff0","#e54b4b","#17a673","#e59b27","#1597bb","#d14d9f","#708090","#85a832"];
const state={runs:new Set(D.runs.map((_,i)=>i)),smooth:D.smooth,hidden:new Set()};
const charts=[],tip=document.getElementById("tip");
function esc(s){return String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));}
function fmt(v){if(v===undefined)return "—";let a=Math.abs(v);return a&&(a<1e-3||a>=1e4)?v.toExponential(3):v.toFixed(4).replace(/\.?0+$/,"");}
function latest(rows,key){for(let i=rows.length-1;i>=0;i--)if(key in rows[i])return rows[i][key];}
function color(ri,mi,count){return palette[(ri*count+mi)%palette.length];}
function points(run,metric){let raw=run.records.filter(r=>metric in r).map(r=>[r.step,r[metric]]),out=[],values=[];
  raw.forEach(p=>{values.push(p[1]);let tail=values.slice(-state.smooth);out.push([p[0],tail.reduce((a,b)=>a+b,0)/tail.length]);});return out;}
function updateSummary(){let root=document.getElementById("summary");root.innerHTML="";D.runs.forEach((run,ri)=>{
  if(!state.runs.has(ri))return;let hours=run.records.reduce((s,r)=>s+(r["timing_s/step"]||0),0)/3600;
  let a=document.createElement("article");a.innerHTML=`<h3>${esc(run.label)}</h3><dl><dt>最后 step</dt><dd>${run.records.at(-1).step}</dd>
  <dt>Train reward</dt><dd>${fmt(latest(run.records,"train/reward/sum/mean"))}</dd><dt>Val reward</dt>
  <dd>${fmt(latest(run.records,"val/reward/sum/mean"))}</dd><dt>Actor loss</dt>
  <dd>${fmt(latest(run.records,"actor/total_loss"))}</dd><dt>累计 step 时间</dt><dd>${hours.toFixed(2)} h</dd></dl>`;root.append(a);});}
function getSeries(chart){let out=[];D.runs.forEach((run,ri)=>{if(!state.runs.has(ri))return;chart.panel.metrics.forEach((metric,mi)=>{
  let key=`${chart.id}:${ri}:${metric}`,pts=points(run,metric);if(pts.length&&!state.hidden.has(key))
  out.push({key,pts,color:color(ri,mi,chart.panel.metrics.length),label:`${run.label} · ${metric}`,ri});});});return out;}
function themeColors(){let s=getComputedStyle(document.body);return {ink:s.getPropertyValue("--ink"),muted:s.getPropertyValue("--muted"),line:s.getPropertyValue("--line")};}
function draw(chart){if(chart.closed)return;let c=chart.canvas,w=c.clientWidth,h=c.clientHeight,dpr=devicePixelRatio||1;
  c.width=Math.round(w*dpr);c.height=Math.round(h*dpr);let x=c.getContext("2d");x.setTransform(dpr,0,0,dpr,0,0);x.clearRect(0,0,w,h);
  let series=getSeries(chart),all=series.flatMap(s=>s.pts),p={l:58,r:w-15,t:17,b:h-33};if(!all.length){x.fillStyle=themeColors().muted;
  x.fillText("没有可显示的数据",p.l,80);chart.plot=null;return;}let full=[Math.min(...all.map(v=>v[0])),Math.max(...all.map(v=>v[0]))];
  if(full[0]===full[1])full[1]++;let xr=chart.range||full,shown=all.filter(v=>v[0]>=xr[0]&&v[0]<=xr[1]);if(!shown.length)shown=all;
  let ymin=Math.min(...shown.map(v=>v[1])),ymax=Math.max(...shown.map(v=>v[1]));if(ymin===ymax){let pad=Math.max(Math.abs(ymin)*.05,1e-9);ymin-=pad;ymax+=pad;}
  else{let pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;}let sx=v=>p.l+(v-xr[0])/(xr[1]-xr[0])*(p.r-p.l);
  let sy=v=>p.t+(ymax-v)/(ymax-ymin)*(p.b-p.t),tc=themeColors();x.font="11px system-ui";x.strokeStyle=tc.line;x.fillStyle=tc.muted;
  for(let i=0;i<5;i++){let q=i/4,yy=p.t+q*(p.b-p.t),val=ymax-q*(ymax-ymin);x.beginPath();x.moveTo(p.l,yy);x.lineTo(p.r,yy);
  x.stroke();x.textAlign="right";x.fillText(fmt(val),p.l-8,yy+4);x.textAlign="center";let step=xr[0]+q*(xr[1]-xr[0]);x.fillText(fmt(step),p.l+q*(p.r-p.l),p.b+20);}
  series.forEach(s=>{x.strokeStyle=s.color;x.lineWidth=2;x.setLineDash(s.ri?[7,4]:[]);x.beginPath();let started=false;s.pts.forEach(v=>{
  if(v[0]<xr[0]||v[0]>xr[1])return;let xx=sx(v[0]),yy=sy(v[1]);started?x.lineTo(xx,yy):x.moveTo(xx,yy);started=true;});x.stroke();});x.setLineDash([]);
  chart.plot={p,xr,full,ymin,ymax,sx,sy,series};if(chart.hover!==null)drawHover(chart,x);}
function drawHover(chart,x){let o=chart.plot,px=Math.max(o.p.l,Math.min(o.p.r,chart.hover)),target=o.xr[0]+(px-o.p.l)/(o.p.r-o.p.l)*(o.xr[1]-o.xr[0]);
  x.strokeStyle="#8b95a7";x.lineWidth=1;x.beginPath();x.moveTo(px,o.p.t);x.lineTo(px,o.p.b);x.stroke();let rows=[];
  o.series.forEach(s=>{let near=s.pts.reduce((a,b)=>Math.abs(b[0]-target)<Math.abs(a[0]-target)?b:a);rows.push([s,near]);x.fillStyle=s.color;
  x.beginPath();x.arc(o.sx(near[0]),o.sy(near[1]),3.5,0,Math.PI*2);x.fill();});if(!rows.length)return;
  tip.innerHTML=`<b>step ${fmt(rows[0][1][0])}</b>`+rows.map(([s,v])=>`<div class="tip-row"><i class="dot" style="background:${s.color}"></i>
  <span>${esc(s.label)}</span><strong>${fmt(v[1])}</strong></div>`).join("");tip.style.display="block";}
function makeLegend(chart,root){D.runs.forEach((run,ri)=>chart.panel.metrics.forEach((metric,mi)=>{if(!run.records.some(r=>metric in r))return;
  let key=`${chart.id}:${ri}:${metric}`,b=document.createElement("button");b.innerHTML=`<i class="swatch" style="background:${color(ri,mi,chart.panel.metrics.length)}"></i>
  ${esc(run.label)} · ${esc(metric)}`;b.onclick=()=>{state.hidden.has(key)?state.hidden.delete(key):state.hidden.add(key);b.classList.toggle("off");draw(chart);};root.append(b);}));}
function addChart(panel){let id=charts.length,box=document.createElement("section");box.className="chart";box.innerHTML=`<div class="chart-head"><h2></h2>
  <div class="actions"><button title="下载 PNG">下载</button><button title="移除此图">×</button></div></div><div class="canvas-wrap"><canvas></canvas></div><div class="legend"></div>`;
  box.querySelector("h2").textContent=panel.title;document.getElementById("charts").append(box);let chart={id,panel,box,canvas:box.querySelector("canvas"),
  range:null,hover:null,drag:null,closed:false};charts.push(chart);makeLegend(chart,box.querySelector(".legend"));let buttons=box.querySelectorAll(".actions button");
  buttons[0].onclick=()=>{let a=document.createElement("a");a.download=panel.title.replace(/\W+/g,"_")+".png";a.href=chart.canvas.toDataURL();a.click();};
  buttons[1].onclick=()=>{chart.closed=true;box.remove();};chart.canvas.onwheel=e=>{e.preventDefault();if(!chart.plot)return;let o=chart.plot,
  px=e.offsetX,ratio=Math.max(0,Math.min(1,(px-o.p.l)/(o.p.r-o.p.l))),span=(o.xr[1]-o.xr[0])*Math.exp(e.deltaY*.001),max=o.full[1]-o.full[0];
  span=Math.max(Math.min(span,max),Math.min(1,max));let center=o.xr[0]+ratio*(o.xr[1]-o.xr[0]),lo=center-ratio*span;
  lo=Math.max(o.full[0],Math.min(lo,o.full[1]-span));chart.range=[lo,lo+span];draw(chart);};
  chart.canvas.onmousedown=e=>{if(chart.plot)chart.drag={x:e.offsetX,range:[...chart.plot.xr]};};chart.canvas.onmouseup=()=>chart.drag=null;
  chart.canvas.onmousemove=e=>{if(chart.drag&&chart.plot){let span=chart.drag.range[1]-chart.drag.range[0],delta=(chart.drag.x-e.offsetX)*span/
  (chart.plot.p.r-chart.plot.p.l),max=chart.plot.full[1]-chart.plot.full[0],lo=Math.max(chart.plot.full[0],Math.min(chart.drag.range[0]+delta,
  chart.plot.full[1]-span));chart.range=max===span?null:[lo,lo+span];}else chart.hover=e.offsetX;draw(chart);let r=e.target.getBoundingClientRect();
  tip.style.left=Math.min(innerWidth-tip.offsetWidth-12,r.left+e.offsetX+14)+"px";tip.style.top=Math.min(innerHeight-tip.offsetHeight-12,r.top+e.offsetY+14)+"px";};
  chart.canvas.onmouseleave=()=>{if(!chart.drag){chart.hover=null;tip.style.display="none";draw(chart);}};chart.canvas.ondblclick=()=>{chart.range=null;draw(chart);};
  new ResizeObserver(()=>draw(chart)).observe(chart.canvas);draw(chart);}
function renderAll(){updateSummary();charts.forEach(draw);}document.getElementById("title").textContent=D.title;
window.addEventListener("mouseup",()=>charts.forEach(chart=>chart.drag=null));
document.getElementById("meta").textContent=`${D.runs.length} runs · ${D.metrics.length} metrics · offline`;
let runs=document.getElementById("runs");D.runs.forEach((run,i)=>{let label=document.createElement("label");label.className="run";
  label.innerHTML=`<input type="checkbox" checked><span>${esc(run.label)}<small><br>${run.records.length} records</small></span>`;
  label.querySelector("input").onchange=e=>{e.target.checked?state.runs.add(i):state.runs.delete(i);renderAll();};runs.append(label);});
let smooth=document.getElementById("smooth"),smoothValue=document.getElementById("smooth-value");smooth.value=state.smooth;
function setSmooth(){state.smooth=+smooth.value;smoothValue.textContent=state.smooth;renderAll();}smooth.oninput=setSmooth;setSmooth();
let list=document.getElementById("metric-list");D.metrics.forEach(metric=>{let option=document.createElement("option");option.value=metric;list.append(option);});
document.getElementById("add").onclick=()=>{let input=document.getElementById("metric"),metric=input.value.trim();if(!D.metrics.includes(metric)){
  input.setCustomValidity("请选择日志中存在的 metric");input.reportValidity();return;}input.setCustomValidity("");addChart({title:metric,metrics:[metric]});input.value="";};
document.getElementById("metric").onkeydown=e=>{if(e.key==="Enter")document.getElementById("add").click();};
document.getElementById("reset").onclick=()=>{charts.forEach(c=>c.range=null);renderAll();};document.getElementById("theme").onclick=e=>{
  let dark=document.documentElement.dataset.theme!=="dark";document.documentElement.dataset.theme=dark?"dark":"";e.target.textContent=dark?"浅色":"深色";renderAll();};
D.panels.forEach(addChart);updateSummary();
</script></body></html>"""


def parse_log(path: Path) -> list[dict[str, float]]:
    """Parse numeric metrics from console ``step:N - key:value`` records."""
    by_step: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw_line in stream:
            line = ANSI_RE.sub("", raw_line.replace("\r", ""))
            match = STEP_RE.search(line)
            if not match:
                continue
            step = int(match.group(1))
            record = by_step.setdefault(step, {"step": float(step)})
            for item in match.group(2).split(" - "):
                key, separator, value = item.partition(":")
                if not separator or not key.strip():
                    continue
                try:
                    number = float(value.strip())
                except ValueError:
                    continue
                if math.isfinite(number):
                    record[key.strip()] = number
    return [by_step[step] for step in sorted(by_step)]


def discover_logs(inputs: list[Path]) -> list[Path]:
    """Expand input files and directories into an ordered unique log list."""
    paths: list[Path] = []
    for item in inputs:
        if item.is_file():
            paths.append(item.resolve())
        elif item.is_dir():
            paths.extend(path.resolve() for path in sorted(item.rglob("*.log")))
        else:
            raise FileNotFoundError(f"Log path does not exist: {item}")
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise ValueError("No .log files found")
    return unique


def infer_label(path: Path) -> str:
    """Return a compact run label for the common ``logs/date/0.log`` layout."""
    if path.parent.parent.name == "logs":
        return f"{path.parent.parent.parent.name}/{path.parent.name}"
    return f"{path.parent.name}/{path.name}"


def write_csv(run: dict, output_dir: Path) -> Path:
    """Write all parsed metrics for one run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["step"] + sorted({key for row in run["records"] for key in row if key != "step"})
    path = output_dir / f"{run['index'] + 1:02d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', run['label'])}.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(run["records"])
    return path


def render_dashboard(title: str, runs: list[dict], window: int) -> str:
    """Build a self-contained interactive dashboard document."""
    metrics = sorted({key for run in runs for row in run["records"] for key in row if key != "step"})
    payload = {
        "title": title,
        "smooth": window,
        "metrics": metrics,
        "panels": [{"title": name, "metrics": list(panel_metrics)} for name, panel_metrics in PANELS],
        "runs": [
            {"label": run["label"], "path": str(run["path"]), "records": run["records"]} for run in runs
        ],
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__TITLE__", html.escape(title)).replace("__DATA__", data)


def main() -> None:
    """Parse command-line arguments and write HTML plus CSV outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Log files or directories (directories are recursive)")
    parser.add_argument("-o", "--output", type=Path, default=Path("training_dashboard.html"))
    parser.add_argument("--label", action="append", dest="labels", help="Run label; repeat once per discovered log")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window (default: 1)")
    parser.add_argument("--title", default="OmniNFT 训练结果")
    parser.add_argument("--no-csv", action="store_true", help="Do not export parsed metrics as CSV")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be at least 1")
    paths = discover_logs(args.inputs)
    if args.labels and len(args.labels) != len(paths):
        parser.error(f"received {len(args.labels)} labels for {len(paths)} logs")
    runs = []
    for index, path in enumerate(paths):
        records = parse_log(path)
        if not records:
            parser.error(f"no step metrics found in {path}")
        label = args.labels[index] if args.labels else infer_label(path)
        runs.append({"index": index, "label": label, "path": path, "records": records})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_dashboard(args.title, runs, args.smooth), encoding="utf-8")
    print(f"HTML: {args.output.resolve()}")
    if not args.no_csv:
        csv_dir = args.output.parent / f"{args.output.stem}_csv"
        for run in runs:
            print(f"CSV:  {write_csv(run, csv_dir).resolve()}")
    for run in runs:
        print(f"{run['label']}: {len(run['records'])} steps, {len(run['records'][-1]) - 1} latest metrics")


if __name__ == "__main__":
    main()
