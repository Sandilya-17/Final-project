import json, pandas as pd
d = pd.read_csv("label_annotation.csv", encoding="utf-8-sig")
data = json.dumps([{"id": int(i), "text": str(t)} for i, t in zip(d["id"], d["text"])], ensure_ascii=False)
html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Label annotation</title>
<style>
body{font-family:-apple-system,system-ui,sans-serif;max-width:900px;margin:32px auto;padding:0 16px;background:#fafafa;color:#111}
#s{font-size:30px;line-height:1.7;padding:28px;background:#fff;border:1px solid #ddd;border-radius:10px;min-height:120px;font-family:"Noto Sans Telugu","Telugu Sangam MN",sans-serif}
button{font-size:18px;padding:12px 22px;margin:8px 8px 0 0;border-radius:8px;border:1px solid #888;background:#fff;cursor:pointer}
.p{background:#d9f2d9}.n{background:#f7d9d9}.u{background:#e6e6e6}
#bar{height:8px;background:#ddd;border-radius:4px;margin:12px 0}#fill{height:8px;background:#3a7;border-radius:4px;width:0}
small{color:#555}
</style></head><body>
<h2>Telugu sentence labels</h2>
<div id="c"></div><div id="bar"><div id="fill"></div></div>
<div id="s"></div>
<div><button class="p" onclick="a('p')">p &nbsp;Positive</button><button class="n" onclick="a('n')">n &nbsp;Negative</button><button class="u" onclick="a('u')">u &nbsp;Neutral</button><button onclick="back()">b &nbsp;Back</button><button onclick="save()">Download CSV</button></div>
<p><small>Keys: p / n / u label the sentence and go to the next one, b goes back. Judge only how the sentence reads.
Download any time; you can stop and continue (answers are remembered in this browser).</small></p>
<script>
const D=__DATA__; let i=0, A={};
try{A=JSON.parse(localStorage.getItem("ann")||"{}");}catch(e){}
for(i=0;i<D.length&&A[D[i].id];i++);
function show(){document.getElementById("c").textContent=(i<D.length?"Sentence "+(i+1)+" of "+D.length:"All "+D.length+" done - click Download CSV")+" ("+Object.keys(A).length+" labelled)";
document.getElementById("fill").style.width=(100*Object.keys(A).length/D.length)+"%";
document.getElementById("s").textContent=i<D.length?D[i].text:"Done. Click Download CSV, then move the file into the project folder.";}
function a(l){if(i>=D.length)return;A[D[i].id]=l;try{localStorage.setItem("ann",JSON.stringify(A));}catch(e){}i++;show();}
function back(){if(i>0){i--;delete A[D[i].id];try{localStorage.setItem("ann",JSON.stringify(A));}catch(e){}show();}}
function save(){let t="id,my_label\\n"+D.filter(x=>A[x.id]).map(x=>x.id+","+A[x.id]).join("\\n")+"\\n";
const u=URL.createObjectURL(new Blob([t],{type:"text/csv"}));const l=document.createElement("a");l.href=u;l.download="my_labels.csv";l.click();}
document.addEventListener("keydown",e=>{const k=e.key.toLowerCase();if(k=="p"||k=="n"||k=="u")a(k);else if(k=="b")back();});
show();
</script></body></html>"""
open("annotate.html", "w", encoding="utf-8").write(html.replace("__DATA__", data))
print("Wrote annotate.html with", len(d), "sentences")
