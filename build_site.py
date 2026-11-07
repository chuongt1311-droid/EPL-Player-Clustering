"""Builds docs/index.html: the interactive explorer (defenders, midfielders, attackers; every season 2013/14 to 2026/27 so far).
Called by the last section of the notebook. Plotly is loaded from a CDN, everything else is inline."""
import json
from pathlib import Path

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Premier League player types, 2013 to 2026</title>
<meta name="description" content="Interactive map of Premier League defenders, midfielders and attackers grouped by how they pass (k-means on nine pass types).">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#f7f1e3;--ink:#1d1b18;--mute:#8a8170;--line:#d9d0b9;--garnet:#8b1e2d}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Georgia,'DejaVu Serif',serif}
header{padding:16px 24px 4px}h1{margin:0;font-size:26px}.sub{margin:6px 0 0;color:var(--mute);font-size:14px;max-width:1000px;line-height:1.45}.note{color:var(--garnet);font-style:italic;font-size:14px;margin-top:4px}
.bar{display:flex;flex-wrap:wrap;gap:10px 16px;align-items:center;padding:10px 24px;margin-top:8px;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
.bar label{font-size:13px;color:var(--mute)}select,input,button{font-family:inherit;font-size:14px;background:#fffaf0;border:1px solid var(--line);border-radius:6px;padding:6px 8px;color:var(--ink)}button{cursor:pointer}button.primary{background:var(--garnet);color:#fff;border-color:var(--garnet)}
main{display:grid;grid-template-columns:minmax(0,1fr) 330px}#map{height:calc(100vh - 230px);min-height:560px}
aside{padding:14px 18px;border-left:1px solid var(--line)}aside h2{font-size:18px;margin:0}.meta{color:var(--mute);font-size:13px;margin:2px 0 8px}.stat{font-size:13px;margin-bottom:8px}
.row{display:grid;grid-template-columns:104px 1fr 38px;align-items:center;gap:6px;font-size:12px;margin:3px 0}.track{position:relative;height:10px;background:#ece4cf;border-radius:3px}.fill{height:10px;border-radius:3px}.tick{position:absolute;top:-2px;width:2px;height:14px;background:var(--ink)}
.hint{color:var(--mute);font-size:12px;line-height:1.5;margin-top:10px}
.groups{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;padding:12px 24px;border-top:1px solid var(--line)}.g{border-left:10px solid;padding-left:10px}.g b{font-size:15px}.g div{font-size:12.5px;color:var(--mute);margin-top:2px}.g div.n{color:var(--ink)}
.time{padding:6px 24px 4px;border-top:1px solid var(--line)}.time h3{font-size:15px;margin:8px 0 0}#over{height:260px}
details{padding:8px 24px 22px;border-top:1px solid var(--line);font-size:13px;line-height:1.55;color:var(--mute)}summary{cursor:pointer;color:var(--ink);font-size:14px}details b{color:var(--ink)}
@media(max-width:900px){main{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="sub"></div><div class="note" id="note"></div></header>
<div class="bar"><label>Position <select id="grp"></select></label><label>Season <select id="season"></select></label><label>Names <select id="names"><option value="key">Key players</option><option value="all">All players</option><option value="none">None</option></select></label>
<label>Follow a player <input id="find" list="plist" placeholder="e.g. Ben White" autocomplete="off"></label><datalist id="plist"></datalist><button id="clear">Clear</button><button class="primary" id="save">Save image (PNG)</button></div>
<main><div id="map"></div><aside><h2 id="pn">Hover over a dot</h2><div class="meta" id="pm">Details and the full pass mix appear here. Click a dot to keep it.</div><div id="card"></div>
<div class="hint">Bars = share of this player's open-play passes in each of the nine pass types; the black tick = the average for his type. Dot size = value created by passes (xT per 90, x100). The map is the same for every season of a position, so a player who changes style moves across it: use "Follow a player" to see the path.</div></aside></main>
<div class="groups" id="groups"></div>
<div class="time"><h3 id="overt"></h3><div id="over"></div></div>
<details><summary>What are the nine pass types, and how was this made?</summary>
<p id="ptlist"></p>
<p><b>Method.</b> 4.5 million open-play passes (2013/14 to 25 Oct 2026) were clustered with k-means into nine types using where each pass starts, how far it goes toward goal and sideways, and whether it was lofted or a cross. Each player-season is then his share of passes in each type. Players are grouped within position with k-means on those shares; the number of groups was chosen by silhouette and a bootstrap stability check. Names are the author's reading of each group's most-used passes.</p>
<p><b>Caveats.</b> Types form a continuum, not islands (silhouette about 0.2). Position is the one a player most often started in. Pass xT (value created by completed passes) ignores carrying, defending and off-ball work; xT before 2024/25 uses a grid fitted on the same data, so compare within a season. The 2026/27 season is only 8 matches per team old: players with 200+ passes are shown (500+ in full seasons), so groups may still move. Data: WhoScored event data.</p></details>
<script>
const D = __DATA__; const BG='#f7f1e3', INK='#1d1b18', MUTE='#8a8170', PT=D.pt, PC=D.ptcol;
const gd=document.getElementById('map'); const ov=document.getElementById('over'); let G=null, season=D.seasons[D.seasons.length-2], pinned=null, follow=null, hooked=false;
const lab=s=>s.replace('-','/')+(s===D.partial?' so far':'');
const diam=xt=>1.25*Math.sqrt(40+330*Math.min(Math.max(xt==null?0:xt,0),12)/12);
const gsel=document.getElementById('grp'), ssel=document.getElementById('season');
Object.keys(D.groups).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=D.groups[k].label;gsel.appendChild(o)});
D.seasons.slice().reverse().forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=lab(s);ssel.appendChild(o)}); ssel.value=season;
document.getElementById('ptlist').innerHTML=PT.map((n,i)=>`<b style="color:${PC[i]}">${n}</b>: ${D.ptdesc[i]}`).join(' &nbsp;·&nbsp; ');
function seasonPlayers(){ return G.players.filter(p=>p.se===season); }
function setGroup(k){ G=D.groups[k]; pinned=null; follow=null; document.getElementById('find').value='';
  const dl=document.getElementById('plist'); dl.innerHTML=''; [...new Set(G.players.map(p=>p.n))].sort().forEach(n=>{const o=document.createElement('option');o.value=n;dl.appendChild(o)});
  card(null); render(); }
function render(){
  const ps=seasonPlayers(); const mode=document.getElementById('names').value;
  document.getElementById('title').textContent=G.title+', '+lab(season);
  document.getElementById('sub').textContent=`k-means on each ${G.who}'s mix of the nine pass types: ${ps.length} players in ${lab(season)} (${season===D.partial?D.minp_partial:D.minp}+ passes). The map is fixed across seasons, so dots are comparable. Dot size = value created by passes (xT per 90). Hover for details.`;
  document.getElementById('note').textContent = season===D.partial ? 'This season is only 8 matches per team old (to 25 Oct 2026): players with 200+ passes, so groups may still move.' : (season<'2024-25'?'xT before 2024/25 uses a grid fitted on the same data: compare within a season, not across.':'');
  const tr=[]; G.types.forEach(t=>{ const idx=ps.map((p,i)=>i).filter(i=>ps[i].c===t.id);
    tr.push({type:'scatter',mode:mode==='none'?'markers':'markers+text',name:t.name,x:idx.map(i=>ps[i].x),y:idx.map(i=>ps[i].y),customdata:idx.map(i=>ps[i].id),
      text:idx.map(i=>mode==='all'||(mode==='key'&&ps[i].k)?ps[i].s:''),textposition:'middle right',textfont:{size:11,color:INK,family:'Georgia'},
      marker:{size:idx.map(i=>diam(ps[i].xt)),color:t.color,opacity:follow?0.3:0.8,line:{color:BG,width:1}},
      hovertext:idx.map(i=>{const p=ps[i];return `<b>${p.n}</b><br>${p.t||''}${p.t?' · ':''}${p.pos}<br>${t.name}<br>xT/90 ${p.xt==null?'n/a':p.xt} · ${p.p.toLocaleString()} passes`}),hoverinfo:'text',
      hoverlabel:{bgcolor:'#fffaf0',bordercolor:t.color,font:{family:'Georgia',size:13,color:INK}}}); });
  if(follow){ const tp=G.players.filter(p=>p.n===follow).sort((a,b)=>a.se<b.se?-1:1);
    tr.push({type:'scatter',mode:'lines+markers+text',x:tp.map(p=>p.x),y:tp.map(p=>p.y),text:tp.map(p=>p.se.slice(2).replace('-','/')),textposition:'top center',textfont:{size:11,color:'#8b1e2d',family:'Georgia'},
      line:{color:'#8b1e2d',width:2.2},marker:{size:tp.map(p=>p.se===season?diam(p.xt)+8:9),color:tp.map(p=>p.se===season?'rgba(0,0,0,0)':'#8b1e2d'),line:{color:'#8b1e2d',width:tp.map(p=>p.se===season?3:1)}},hoverinfo:'skip',showlegend:false}); }
  const xs=G.players.map(p=>p.x), ys=G.players.map(p=>p.y), xr=[Math.min(...xs)-0.06,Math.max(...xs)+0.06], yr=[Math.min(...ys)-0.05,Math.max(...ys)+0.05];
  const A=G.axes, ann=[{xref:'paper',yref:'paper',x:0,y:-0.075,xanchor:'left',showarrow:false,text:'← '+A.x[0],font:{size:12,color:MUTE}},{xref:'paper',yref:'paper',x:1,y:-0.075,xanchor:'right',showarrow:false,text:A.x[1]+' →',font:{size:12,color:MUTE}},
    {xref:'paper',yref:'paper',x:-0.035,y:1,yanchor:'top',textangle:-90,showarrow:false,text:'↑ '+A.y[1],font:{size:12,color:MUTE}},{xref:'paper',yref:'paper',x:-0.035,y:0,yanchor:'bottom',textangle:-90,showarrow:false,text:'↓ '+A.y[0],font:{size:12,color:MUTE}}];
  Plotly.react(gd,tr,{paper_bgcolor:BG,plot_bgcolor:BG,font:{family:'Georgia, serif',color:INK},margin:{l:62,r:20,t:8,b:62},dragmode:'pan',showlegend:false,hovermode:'closest',hoverdistance:25,
    xaxis:{range:xr,showgrid:false,zeroline:false,showticklabels:false},yaxis:{range:yr,showgrid:false,zeroline:false,showticklabels:false},annotations:ann},
    {responsive:true,scrollZoom:true,displaylogo:false,modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d'],toImageButtonOptions:{format:'png',filename:'player_types',width:2400,height:1900,scale:1}});
  if(!hooked){ hooked=true;
    gd.on('plotly_hover',ev=>{ const pt=ev.points[0]; if(pt.customdata!==undefined && pinned===null) card(pt.customdata); });
    gd.on('plotly_unhover',()=>{ if(pinned===null) card(null); });
    gd.on('plotly_click',ev=>{ const pt=ev.points[0]; if(pt.customdata!==undefined){ pinned=pt.customdata; card(pinned);} }); }
  const cnt={}; ps.forEach(p=>cnt[p.c]=(cnt[p.c]||0)+1);
  document.getElementById('groups').innerHTML=G.types.map(t=>{const m=ps.filter(p=>p.c===t.id), xt=m.filter(p=>p.xt!=null).map(p=>p.xt), a=xt.length?(xt.reduce((u,v)=>u+v,0)/xt.length).toFixed(1):'n/a';
    return `<div class="g" style="border-color:${t.color}"><b style="color:${t.color}">${t.name}</b><div class="n">${m.length} players in ${lab(season)} · pass xT/90 ${a}</div><div>${t.desc}</div><div>Plays most: ${t.sig}</div></div>`}).join('');
  drawOver(); }
function drawOver(){ const ss=D.seasons; document.getElementById('overt').textContent='How the types changed: share of '+G.label.toLowerCase()+' in each type, by season';
  const tr=G.types.map(t=>({type:'bar',name:t.name,x:ss.map(s=>s.slice(2).replace('-','/')+(s===D.partial?'*':'')),y:ss.map(s=>100*(G.share[s]?G.share[s][t.id]:0)),marker:{color:t.color},hovertemplate:'%{x}: %{y:.0f}%<extra>'+t.name+'</extra>'}));
  Plotly.react(ov,tr,{barmode:'stack',paper_bgcolor:BG,plot_bgcolor:BG,font:{family:'Georgia, serif',color:INK,size:12},margin:{l:42,r:10,t:6,b:36},yaxis:{ticksuffix:'%',range:[0,100],gridcolor:'#e4dcc6'},xaxis:{tickangle:-45},legend:{orientation:'h',y:-0.32,font:{size:11}}},{displayModeBar:false,responsive:true}); }
function card(id){ const el=document.getElementById('card'); if(id===null){ document.getElementById('pn').textContent='Hover over a dot'; document.getElementById('pm').textContent='Details and the full pass mix appear here. Click a dot to keep it.'; el.innerHTML=''; return; }
  const p=G.players.find(q=>q.id===id); if(!p) return; const t=G.types.find(x=>x.id===p.c);
  document.getElementById('pn').textContent=p.n; document.getElementById('pm').innerHTML=`${lab(p.se)} · ${p.t?p.t+' · ':''}${p.pos}`;
  const order=[...Array(9).keys()].sort((a,b)=>p.sh[b]-p.sh[a]);
  el.innerHTML=`<div class="stat"><span style="color:${t.color}">●</span> <b style="color:${t.color}">${t.name}</b><br>Pass xT/90: <b>${p.xt==null?'n/a (under the minutes needed)':p.xt}</b> · ${p.p.toLocaleString()} passes · ${p.m.toLocaleString()} minutes</div>`+
   order.map(j=>`<div class="row"><span>${PT[j]}</span><div class="track"><div class="fill" style="width:${Math.min(p.sh[j]*2.2,100)}%;background:${PC[j]}"></div><div class="tick" style="left:${Math.min(t.mix[j]*2.2,100)}%"></div></div><span>${p.sh[j].toFixed(0)}%</span></div>`).join(''); }
gsel.onchange=e=>setGroup(e.target.value); ssel.onchange=e=>{season=e.target.value;pinned=null;card(null);render()};
document.getElementById('names').onchange=()=>render();
document.getElementById('find').onchange=e=>{ const q=e.target.value.trim().toLowerCase(); if(!q) return; const all=[...new Set(G.players.map(p=>p.n))]; let n=all.find(x=>x.toLowerCase()===q)||all.find(x=>x.toLowerCase().includes(q));
  if(n){ follow=n; const tp=G.players.filter(p=>p.n===n).sort((a,b)=>a.se<b.se?-1:1); if(!tp.some(p=>p.se===season)){season=tp[tp.length-1].se; ssel.value=season;} const cur=G.players.find(p=>p.n===n&&p.se===season); pinned=cur.id; card(pinned); render(); } };
document.getElementById('clear').onclick=()=>{follow=null;pinned=null;document.getElementById('find').value='';card(null);render()};
document.getElementById('save').onclick=()=>Plotly.downloadImage(gd,{format:'png',filename:'player_types',width:2400,height:1900,scale:1});
setGroup(Object.keys(D.groups)[0]);
</script></body></html>"""


def build_site(payload: dict, out_path) -> None:
    html = TEMPLATE.replace('__DATA__', json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    Path(out_path).write_text(html, encoding='utf-8')
