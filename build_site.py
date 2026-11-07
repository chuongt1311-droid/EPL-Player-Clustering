"""Builds docs/index.html: the interactive explorer (defenders, midfielders, attackers; every season 2013/14 to 2026/27 so far).
Called by the last section of the notebook. Plotly is loaded from a CDN, everything else is inline."""
import base64
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
.sim{margin-top:14px;border-top:1px solid var(--line);padding-top:8px}.sim h3{font-size:15px;margin:0 0 4px}.sim .ctl{display:flex;flex-wrap:wrap;gap:6px 10px;align-items:center;font-size:12px;color:var(--mute);margin-bottom:6px}.sim select{font-size:12px;padding:3px 5px}
.nb{padding:6px 8px;margin:3px -8px;border-radius:6px;cursor:pointer;font-size:13px;line-height:1.35}.nb:hover{background:#efe6cf}.nb .m{color:var(--mute);font-size:12px}.nb .w{color:var(--garnet)}
@media(max-width:900px){main{grid-template-columns:1fr}aside{border-left:0;border-top:1px solid var(--line)}}
</style></head><body>
<header><h1 id="title"></h1><div class="sub" id="sub"></div><div class="note" id="note"></div></header>
<div class="bar"><label>Position <select id="grp"></select></label><label>Season <select id="season"></select></label><label>Carry level <select id="carry"><option value="all">All</option><option value="2">High carriers</option><option value="1">Mid carriers</option><option value="0">Low carriers</option></select></label><label>Dot size <select id="dsize"><option value="pass">Pass xT/90</option><option value="carry">Carry xT/90</option><option value="both">Pass + carry xT/90</option></select></label><label>Detail <select id="lvl"><option value="c">Types (stable)</option><option value="c2">Finer styles (fuzzier)</option></select></label><label>Names <select id="names"><option value="key">Key players</option><option value="all">All players</option><option value="none">None</option></select></label>
<label>Follow a player <input id="find" list="plist" placeholder="e.g. Ben White" autocomplete="off"></label><datalist id="plist"></datalist><button id="clear">Clear</button><label>Credit <input id="credit" value="@chuongt1311-droid" style="width:150px"></label><button class="primary" id="save">Save image (PNG)</button></div>
<main><div id="map"></div><aside><h2 id="pn">Hover over a dot</h2><div class="meta" id="pm">Details and the full pass mix appear here. Click a dot to keep it.</div><div id="card"></div>
<div class="hint">Bars = share of this player's open-play passes in each of the nine pass types; the black tick = the average for his type. Dot size = value created (xT per 90, x100; passes by default, or carries, or both). The map is the same for every season of a position, so a player who changes style moves across it: use "Follow a player" to see the path. Click a dot to keep it: the <b>Plays like</b> list below it names the eight players with the closest mix, and the season each one matched in. Click a name to jump there.</div></aside></main>
<div class="groups" id="groups"></div>
<div class="time"><h3 id="overt"></h3><div id="over"></div></div>
<details><summary>What are the nine pass types, and how was this made?</summary>
<p id="ptlist"></p>
<p><b>Method.</b> 4.5 million open-play passes (2013/14 to 25 Oct 2026) were clustered with k-means into nine types using where each pass starts, how far it goes toward goal and sideways, and whether it was lofted or a cross. Each player-season is then his share of passes in each type. Players are grouped within position with k-means on those shares, with a separate pass cutoff and number of groups for each position, chosen by split-half reliability, silhouette and stability checks. <b>Types</b> (4, 3 and 4 groups) are the stable level; <b>styles</b> (7, 5 and 6 groups) are finer but their borders wobble, so treat them as neighbourhoods on a continuum. Names are the author's reading of each group's most-used passes.</p>
<p><b>Plays like.</b> For the selected player-season the page computes the Hellinger distance (Euclidean distance between the square roots of the nine shares) to every other player-season of every position, keeps each other player's closest season, and lists the eight nearest. The noise scale is the typical distance between a player's own two half-seasons, rescaled to two full seasons: matches closer than that cannot be told apart from sampling noise. At 500+ passes the player's own other half-season was the very nearest only about one time in eight (median rank about 14 of 3,300), so read the list as a shortlist.</p>
<p><b>Carrying.</b> A carry is the ball moving with a player between two events, inferred from event data (no speed, space or pressure); a take-on is an attempt to beat an opponent. Per 90 minutes the page shows carries, progressive carries (10 m or more closer to goal), take-ons and carry xT, each with its percentile among players of the same position and season, plus a three-level carry tier (thirds of a composite of carries, progressive carries and take-ons within position) and a dribbler tag (top third of take-ons per carry). Carries do <b>not</b> make the k-means groups more stable (tested: stability unchanged), so the types and styles above use passes only. They do help to tell players apart, which is why they sit in the Plays like list. The tier is fuzzy: a player is in the same third on his odd and even matches about 70 to 80% of the time. Carry xT before 2024/25 uses a grid fitted on the same data.</p>
<p><b>Caveats.</b> Types form a continuum, not islands (silhouette about 0.2). Position is the one a player most often started in. Pass xT (value created by completed passes) ignores carrying, defending and off-ball work; xT before 2024/25 uses a grid fitted on the same data, so compare within a season. The 2026/27 season is only 8 matches per team old: players with 200+ passes are shown (full seasons use a cutoff of 150 for defenders and 250 for midfielders and attackers), so groups may still move. Data: WhoScored event data.</p></details>
<script>
const D = __DATA__; const BG='#f7f1e3', INK='#1d1b18', MUTE='#8a8170', PT=D.pt, PC=D.ptcol;
const gd=document.getElementById('map'); const ov=document.getElementById('over'); let G=null, lvl='c', season=D.seasons[D.seasons.length-2], pinned=null, follow=null, hooked=false;
const lab=s=>s.replace('-','/')+(s===D.partial?' so far':'');
const xtv=p=>dsize==='pass'?p.xt:dsize==='carry'?(p.cm?p.cm[3]:null):((p.xt==null&&!p.cm)?null:(p.xt||0)+(p.cm?p.cm[3]:0));
const diam=p=>{const mx=dsize==='both'?16:12, xt=xtv(p); return 1.25*Math.sqrt(40+330*Math.min(Math.max(xt==null?0:xt,0),mx)/mx);};
const okC=p=>carryF==='all'||p.ct===+carryF;
const gsel=document.getElementById('grp'), ssel=document.getElementById('season');
Object.keys(D.groups).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=D.groups[k].label;gsel.appendChild(o)});
D.seasons.slice().reverse().forEach(s=>{const o=document.createElement('option');o.value=s;o.textContent=lab(s);ssel.appendChild(o)}); ssel.value=season;
document.getElementById('ptlist').innerHTML=PT.map((n,i)=>`<b style="color:${PC[i]}">${n}</b>: ${D.ptdesc[i]}`).join(' &nbsp;·&nbsp; ');
const NB=D.nb||8; let nCarry=true, carryF='all', dsize='pass', nPool='all', nRole=false, curId=null, GK=null, NBL=[], ALLP=null;
function allp(){ if(ALLP) return ALLP; ALLP=[]; Object.keys(D.groups).forEach(g=>D.groups[g].players.forEach(p=>ALLP.push({g,p,v:p.sh.map(x=>Math.sqrt(x/100))}))); return ALLP; }
function neighbours(p){ const q=p.sh.map(x=>Math.sqrt(x/100)), best={};
  allp().forEach(o=>{ const r=o.p; if(r.n===p.n) return; if(nPool==='same'&&r.se!==p.se) return; if(nRole&&o.g!==GK) return;
    let s=0; for(let i=0;i<9;i++){ const d=o.v[i]-q[i]; s+=d*d; } const c=nCarry&&p.cz&&r.cz; if(c){ for(let i=0;i<4;i++){ const d=(r.cz[i]-p.cz[i])*D.cw; s+=d*d; } } const d=Math.sqrt(s); if(!best[r.n]||d<best[r.n].d) best[r.n]={o,d,c}; });
  return Object.values(best).sort((a,b)=>a.d-b.d).slice(0,NB); }
function similarHtml(p){ NBL=neighbours(p);
  const rows=NBL.map((x,i)=>{ const r=x.o.p, diff=r.sh.map((v,j)=>v-p.sh[j]), top=[...Array(9).keys()].sort((a,b)=>Math.abs(diff[b])-Math.abs(diff[a])).slice(0,2);
    const why=top.map(j=>`${PT[j]} ${diff[j]>=0?'+':'\u2212'}${Math.abs(diff[j]).toFixed(0)}`).join(', ');
    return `<div class="nb" data-i="${i}"><b>${r.n}</b> <span class="m">${lab(r.se)} \u00b7 ${r.pos}</span><br><span class="m">distance ${x.d.toFixed(3)}${x.d<=(x.c?D.noisec:D.noise)?' <span class="w">(within noise)</span>':''}${nCarry&&!x.c?' (passes only)':''} \u00b7 differs by: ${why} pts</span></div>`; }).join('');
  return `<div class="sim"><h3>Plays like</h3><div class="ctl"><select id="npool"><option value="all"${nPool==='all'?' selected':''}>Any season</option><option value="same"${nPool==='same'?' selected':''}>Same season only</option></select><label><input type="checkbox" id="nrole"${nRole?' checked':''}> same position group only</label><label><input type="checkbox" id="ncarry"${nCarry?' checked':''}> include carrying</label></div>${rows}<div class="hint">Distance = Hellinger distance between the two pass mixes (one row per player, his closest season). Below ${nCarry?D.noisec:D.noise} is indistinguishable from sampling noise. Carrying (carries, progressive carries, take-ons and carry xT per 90) is added at a low weight: it roughly doubles how often a player's own other half-season is found, from 9% to 17%. A shortlist, not a twin: tested at 500+ passes, a player's own other half-season was the very nearest only about one time in eight.</div></div>`; }
function carryHtml(p){ const nm=['Carries','Progressive','Take-ons','Carry xT'], TN=['low','mid','high'];
  if(!p.cm) return `<div class="sim"><h3>Carrying</h3><div class="hint">Not enough minutes for carry measures (450 needed).</div></div>`;
  const rows=nm.map((n,j)=>`<div class="row" title="${p.cp[j]}th percentile among ${G.who}s this season"><span>${n}</span><div class="track"><div class="fill" style="width:${p.cp[j]}%;background:#5b6b8a"></div></div><span>${p.cm[j]}</span></div>`).join('');
  const sty=TY().find(x=>x.id===p[lvl]).name; const tag=`${sty} \u00b7 <b>${TN[p.ct]} carrier</b>${p.dt===2?' \u00b7 <b>dribbler</b>':''}`;
  return `<div class="sim"><h3>Carrying</h3><div class="meta" style="margin:0 0 4px">per 90 minutes; bar = percentile among ${G.who}s in ${lab(p.se)}</div>${rows}<div class="stat" style="margin-top:6px">${tag}</div><div class="hint" style="margin-top:2px">Take-ons per carry ${p.tpc} (the dribbler index; top third among ${G.who}s = dribbler).${p.se===D.partial?' Provisional: 8 matches.':''} Carries are inferred from event data: where the ball went with the player, not how fast or under what pressure.</div></div>`; }
function jumpTo(g,p){ gsel.value=g; setGroup(g); season=p.se; ssel.value=season; pinned=p.id; card(p.id); render(); }
function renderExport(){
  const ps=seasonPlayers(), ty=TY(), k=ty.length, S=2, PTs=1.389, FONT='Georgia, "DejaVu Serif", serif', GARNET='#8b1e2d';
  const cols=Math.min(k,4), rows=Math.ceil(k/cols), W=1200, L=95, R=1150, partial=season===D.partial, T=partial?200:178, B=T+640, H=B+78+rows*100+120;
  const cv=document.createElement('canvas'); cv.width=W*S; cv.height=H*S; const c=cv.getContext('2d'); c.scale(S,S); c.fillStyle=BG; c.fillRect(0,0,W,H);
  const txt=(s,x,y,size,color,o={})=>{ c.font=`${o.style||''} ${o.weight||''} ${size}px ${FONT}`; c.fillStyle=color; c.textAlign=o.align||'left'; c.textBaseline=o.base||'alphabetic'; c.fillText(s,x,y); };
  const who=G.label.toLowerCase(), credit=(document.getElementById('credit').value||'').trim();
  txt(`K-means clustering: ${who} (${lvl==='c'?'types':'styles'})`,L,64,31,INK,{weight:'bold'});
  txt(`Premier League ${lab(season)}`,L,96,18,MUTE);
  if(credit) txt(`Chart by ${credit}`,L,120,13.5,MUTE,{style:'italic'});
  if(partial) txt('Only 8 matches per team old (to 25 Oct 2026): players with 200+ passes, so groups may still move.',L,credit?144:120,13,GARNET,{style:'italic'});
  const xs=ps.map(p=>p.x), ys=ps.map(p=>p.y), xlo=Math.min(...xs)-0.08, xhi=Math.max(...xs)+0.08, ylo=Math.min(...ys)-0.06, yhi=Math.max(...ys)+0.06;
  const X=x=>L+(x-xlo)/(xhi-xlo)*(R-L), Y=y=>B-(y-ylo)/(yhi-ylo)*(B-T), mx=dsize==='both'?16:12, sArea=v=>40+330*Math.min(Math.max(v==null?0:v,0),mx)/mx, rad=v=>Math.sqrt(sArea(v))/2*PTs;
  const arrow=(x0,y0,x1,y1)=>{ c.strokeStyle=MUTE; c.fillStyle=MUTE; c.lineWidth=1.1; c.beginPath(); c.moveTo(x0,y0); c.lineTo(x1,y1); c.stroke(); const a=Math.atan2(y1-y0,x1-x0); [[x1,y1,a],[x0,y0,a+Math.PI]].forEach(([x,y,t])=>{ c.beginPath(); c.moveTo(x,y); c.lineTo(x-9*Math.cos(t-0.33),y-9*Math.sin(t-0.33)); c.lineTo(x-9*Math.cos(t+0.33),y-9*Math.sin(t+0.33)); c.closePath(); c.fill(); }); };
  arrow(L,B,R,B); arrow(L,B,L,T);
  const A=G.axes; txt(A.x[0],L,B+30,13.5,MUTE); txt(A.x[1],R,B+30,13.5,MUTE,{align:'right'});
  c.save(); c.translate(L-18,T); c.rotate(-Math.PI/2); txt(A.y[1],0,0,13.5,MUTE,{align:'right'}); c.restore(); c.save(); c.translate(L-18,B); c.rotate(-Math.PI/2); txt(A.y[0],0,0,13.5,MUTE); c.restore();
  const col=p=>ty.find(t=>t.id===p[lvl]).color, vis=p=>okC(p), dots=ps.map(p=>({p,x:X(p.x),y:Y(p.y),r:rad(xtv(p))}));
  const order=dots.map((d,i)=>i).sort((a,b)=>((dots[b].p.k?1e6:0)+dots[b].p.p)-((dots[a].p.k?1e6:0)+dots[a].p.p)), placed=[], labels=[], FS=11.6, ang=[0,25,-25,50,-50,90,-90,130,-130,160,-160,180,70,-70,110,-110], rings=[2,8,19,33,48];
  c.font=`${FS}px ${FONT}`;
  for(const i of order){ const d=dots[i]; if(!vis(d.p)) continue; const w=c.measureText(d.p.s).width, h=FS*1.1; let done=false;
    for(let ri=0;ri<rings.length&&!done;ri++){ const r=d.r+rings[ri]*PTs;
      for(const a of ang){ const dx=Math.cos(a*Math.PI/180), dy=-Math.sin(a*Math.PI/180), ha=dx>0.3?'l':(dx<-0.3?'r':'c'), va=Math.abs(dy)<0.5?'c':(dy<0?'b':'t'), tx=d.x+r*dx, tyy=d.y+r*dy;
        const x0=ha==='l'?tx:(ha==='r'?tx-w:tx-w/2), y0=va==='c'?tyy-h/2:(va==='b'?tyy-h:tyy), bx=x0-w*0.02, by=y0-h*0.06, bw=w*1.04, bh=h*1.12, box={x0:bx,y0:by,x1:bx+bw,y1:by+bh};
        if(box.x0<L-70||box.x1>R+14||box.y0<T-6||box.y1>B+2) continue;
        if(placed.some(o=>box.x0<o.x1&&box.x1>o.x0&&box.y0<o.y1&&box.y1>o.y0)) continue;
        if(dots.some((o,j)=>{ if(j===i) return false; const cx=Math.min(Math.max(o.x,box.x0),box.x1), cy=Math.min(Math.max(o.y,box.y0),box.y1); return (cx-o.x)**2+(cy-o.y)**2<(o.r*0.9)**2; })) continue;
        placed.push(box); labels.push({i,box,x:x0,y:y0,ri}); done=true; break; } } }
  dots.forEach(d=>{ c.globalAlpha=vis(d.p)?0.78:0.1; c.fillStyle=col(d.p); c.strokeStyle=BG; c.lineWidth=1.2; c.beginPath(); c.arc(d.x,d.y,d.r,0,2*Math.PI); c.fill(); c.stroke(); }); c.globalAlpha=1;
  labels.forEach(l=>{ const d=dots[l.i]; if(l.ri>=1){ const cx=Math.min(Math.max(d.x,l.box.x0),l.box.x1), cy=Math.min(Math.max(d.y,l.box.y0),l.box.y1), dist=Math.hypot(d.x-cx,d.y-cy)||1, ux=(d.x-cx)/dist, uy=(d.y-cy)/dist, ex=d.x-ux*(d.r+1), ey=d.y-uy*(d.r+1);
      c.strokeStyle=MUTE; c.fillStyle=MUTE; c.lineWidth=0.85; c.beginPath(); c.moveTo(cx,cy); c.lineTo(ex,ey); c.stroke(); const t=Math.atan2(uy,ux); c.beginPath(); c.moveTo(ex,ey); c.lineTo(ex-5.5*Math.cos(t-0.4),ey-5.5*Math.sin(t-0.4)); c.lineTo(ex-5.5*Math.cos(t+0.4),ey-5.5*Math.sin(t+0.4)); c.closePath(); c.fill(); }
    c.font=`${FS}px ${FONT}`; c.textAlign='left'; c.textBaseline='top'; c.lineWidth=3.4; c.strokeStyle=BG; c.lineJoin='round'; c.strokeText(d.p.s,l.x,l.y); c.fillStyle=INK; c.fillText(d.p.s,l.x,l.y); });
  const mu=[...Array(9).keys()].map(j=>ps.reduce((a,p)=>a+p.sh[j],0)/ps.length), sd=mu.map((m,j)=>Math.sqrt(ps.reduce((a,p)=>a+(p.sh[j]-m)**2,0)/ps.length)||1), cw=(R-L)/cols, y0c=B+78;
  ty.forEach((t,j)=>{ const x0=L+(j%cols)*cw, y=y0c+Math.floor(j/cols)*100, m=ps.filter(p=>p[lvl]===t.id), xv=m.filter(p=>p.xt!=null).map(p=>p.xt), a=xv.length?(xv.reduce((u,v)=>u+v,0)/xv.length).toFixed(1):'n/a';
    c.fillStyle=t.color; c.fillRect(x0,y+2,14,80); let nf=16.5; c.font=`bold ${nf}px ${FONT}`; while(c.measureText(t.name).width>cw-38&&nf>11){ nf-=0.5; c.font=`bold ${nf}px ${FONT}`; } txt(t.name,x0+24,y+17,nf,t.color,{weight:'bold'}); txt(`${m.length} players · pass xT/90 ${a}`,x0+24,y+38,12.8,INK);
    if(m.length){ const mm=[...Array(9).keys()].map(q=>m.reduce((u,p)=>u+p.sh[q],0)/m.length), top=[...Array(9).keys()].sort((u,v)=>(mm[v]-mu[v])/sd[v]-(mm[u]-mu[u])/sd[u]).slice(0,3).sort((u,v)=>mm[v]-mm[u]);
      const words=('Plays most: '+top.map(q=>`${PT[q]} ${mm[q].toFixed(0)}%`).join(' · ')).split(' '); let line='', yy=y+57; c.font=`12px ${FONT}`;
      for(const wd of words){ const tr=line?line+' '+wd:wd; if(c.measureText(tr).width>cw-40&&line){ txt(line,x0+24,yy,12,MUTE); yy+=15; line=wd; } else line=tr; } if(line) txt(line,x0+24,yy,12,MUTE); } });
  const vals=dsize==='both'?[2,8,14]:[2,6,10]; vals.forEach((v,q)=>{ const cx=L+16+q*95; c.fillStyle='#b9b09a'; c.beginPath(); c.arc(cx,H-88,rad(v),0,2*Math.PI); c.fill(); txt(String(v),cx+rad(v)+8,H-88,12.5,MUTE,{base:'middle'}); });
  txt('Closer dots pass alike. Colour = k-means group. Names: as many as fit, most-used players first; an arrow marks a name placed away from its dot.',L,H-54,12,MUTE);
  txt('Dot size: '+(dsize==='pass'?'passing threat per 90 minutes (xT, x100)':dsize==='carry'?'carrying threat per 90 minutes (carry xT, x100)':'passing + carrying threat per 90 minutes (xT, x100)'),L,H-28,12.5,MUTE); txt('Data: WhoScored',R,H-28,12.5,MUTE,{align:'right'});
  return cv; }
async function exportPng(){
  const cv=renderExport();
  if(D.logo){ await new Promise(res=>{ const im=new Image(); im.onload=()=>{ const h=84, w=im.width/im.height*h; cv.getContext('2d').drawImage(im,1150-w,34,w,h); res(); }; im.onerror=res; im.src=D.logo; }); }
  cv.toBlob(b=>{ const a=document.createElement('a'); a.href=URL.createObjectURL(b); a.download=`kmeans_${G.label.toLowerCase()}_${lvl==='c'?'types':'styles'}_${season}.png`; a.click(); setTimeout(()=>URL.revokeObjectURL(a.href),3000); },'image/png'); }
function seasonPlayers(){ return G.players.filter(p=>p.se===season); }
const TY=()=>lvl==='c'?G.types:G.styles, SHR=()=>lvl==='c'?G.share:G.share2;
function setGroup(k){ G=D.groups[k]; GK=k; pinned=null; follow=null; document.getElementById('find').value='';
  const dl=document.getElementById('plist'); dl.innerHTML=''; [...new Set(G.players.map(p=>p.n))].sort().forEach(n=>{const o=document.createElement('option');o.value=n;dl.appendChild(o)});
  card(null); render(); }
function render(){
  const ps=seasonPlayers(); const mode=document.getElementById('names').value;
  document.getElementById('title').textContent=(lvl==='c'?G.title:G.title2)+', '+lab(season);
  document.getElementById('sub').textContent=`k-means on each ${G.who}'s mix of the nine pass types: ${ps.length} players in ${lab(season)} (${season===D.partial?D.minp_partial:G.minp}+ passes). The map is fixed across seasons, so dots are comparable. Dot size = value created by passes (xT per 90). Hover for details.`;
  const sb=G.stab[lvl]; const base = season===D.partial ? 'This season is only 8 matches per team old (to 25 Oct 2026): players with 200+ passes, so groups may still move.' : (season<'2024-25'?'xT before 2024/25 uses a grid fitted on the same data: compare within a season, not across.':'');
  document.getElementById('note').textContent = (lvl==='c2' ? `Styles are fuzzy neighbourhoods, not hard categories: ${sb.k} groups, resample stability ${sb.boot}, and a player's odd-match and even-match mixes agree on his style only ${sb.half} (1 = always). ` : `${sb.k} groups, resample stability ${sb.boot}. `) + base;
  const tr=[]; TY().forEach(t=>{ const idx=ps.map((p,i)=>i).filter(i=>ps[i][lvl]===t.id);
    tr.push({type:'scatter',mode:mode==='none'?'markers':'markers+text',name:t.name,x:idx.map(i=>ps[i].x),y:idx.map(i=>ps[i].y),customdata:idx.map(i=>ps[i].id),
      text:idx.map(i=>mode==='all'||(mode==='key'&&ps[i].k)?ps[i].s:''),textposition:'middle right',textfont:{size:11,color:INK,family:'Georgia'},
      marker:{size:idx.map(i=>diam(ps[i])),color:t.color,opacity:idx.map(i=>okC(ps[i])?(follow?0.3:0.8):0.1),line:{color:BG,width:1}},
      hovertext:idx.map(i=>{const p=ps[i];return `<b>${p.n}</b><br>${p.t||''}${p.t?' · ':''}${p.pos}<br>${t.name}<br>xT/90 ${p.xt==null?'n/a':p.xt} · ${p.p.toLocaleString()} passes${p.cm?`<br>carries: ${p.cm[1]} progressive, ${p.cm[2]} take-ons per 90`:''}`}),hoverinfo:'text',
      hoverlabel:{bgcolor:'#fffaf0',bordercolor:t.color,font:{family:'Georgia',size:13,color:INK}}}); });
  if(follow){ const tp=G.players.filter(p=>p.n===follow).sort((a,b)=>a.se<b.se?-1:1);
    tr.push({type:'scatter',mode:'lines+markers+text',x:tp.map(p=>p.x),y:tp.map(p=>p.y),text:tp.map(p=>p.se.slice(2).replace('-','/')),textposition:'top center',textfont:{size:11,color:'#8b1e2d',family:'Georgia'},
      line:{color:'#8b1e2d',width:2.2},marker:{size:tp.map(p=>p.se===season?diam(p)+8:9),color:tp.map(p=>p.se===season?'rgba(0,0,0,0)':'#8b1e2d'),line:{color:'#8b1e2d',width:tp.map(p=>p.se===season?3:1)}},hoverinfo:'skip',showlegend:false}); }
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
    document.getElementById('groups').innerHTML=TY().map(t=>{const m=ps.filter(p=>p[lvl]===t.id), xt=m.filter(p=>p.xt!=null).map(p=>p.xt), a=xt.length?(xt.reduce((u,v)=>u+v,0)/xt.length).toFixed(1):'n/a';
    return `<div class="g" style="border-color:${t.color}"><b style="color:${t.color}">${t.name}</b><div class="n">${m.length} players in ${lab(season)} · pass xT/90 ${a}</div><div>${t.desc}</div><div>Plays most: ${t.sig}</div></div>`}).join('');
  drawOver(); }
function drawOver(){ const ss=D.seasons; document.getElementById('overt').textContent='How the types changed: share of '+G.label.toLowerCase()+' in each type, by season';
  const tr=TY().map(t=>({type:'bar',name:t.name,x:ss.map(s=>s.slice(2).replace('-','/')+(s===D.partial?'*':'')),y:ss.map(s=>100*(SHR()[s]?SHR()[s][t.id]:0)),marker:{color:t.color},hovertemplate:'%{x}: %{y:.0f}%<extra>'+t.name+'</extra>'}));
  Plotly.react(ov,tr,{barmode:'stack',paper_bgcolor:BG,plot_bgcolor:BG,font:{family:'Georgia, serif',color:INK,size:12},margin:{l:42,r:10,t:6,b:36},yaxis:{ticksuffix:'%',range:[0,100],gridcolor:'#e4dcc6'},xaxis:{tickangle:-45},legend:{orientation:'h',y:-0.32,font:{size:11}}},{displayModeBar:false,responsive:true}); }
function card(id){ const el=document.getElementById('card'); curId=id; if(id===null){ document.getElementById('pn').textContent='Hover over a dot'; document.getElementById('pm').textContent='Details and the full pass mix appear here. Click a dot to keep it.'; el.innerHTML=''; return; }
  const p=G.players.find(q=>q.id===id); if(!p) return; const t=TY().find(x=>x.id===p[lvl]);
  document.getElementById('pn').textContent=p.n; document.getElementById('pm').innerHTML=`${lab(p.se)} · ${p.t?p.t+' · ':''}${p.pos}`;
  const order=[...Array(9).keys()].sort((a,b)=>p.sh[b]-p.sh[a]);
  el.innerHTML=`<div class="stat"><span style="color:${t.color}">●</span> <b style="color:${t.color}">${t.name}</b><br>Pass xT/90: <b>${p.xt==null?'n/a (under the minutes needed)':p.xt}</b> · ${p.p.toLocaleString()} passes · ${p.m.toLocaleString()} minutes</div>`+
   order.map(j=>`<div class="row"><span>${PT[j]}</span><div class="track"><div class="fill" style="width:${Math.min(p.sh[j]*2.2,100)}%;background:${PC[j]}"></div><div class="tick" style="left:${Math.min(t.mix[j]*2.2,100)}%"></div></div><span>${p.sh[j].toFixed(0)}%</span></div>`).join('')+carryHtml(p)+similarHtml(p); }
gsel.onchange=e=>setGroup(e.target.value); ssel.onchange=e=>{season=e.target.value;pinned=null;card(null);render()};
document.getElementById('names').onchange=()=>render();
document.getElementById('card').addEventListener('click',e=>{ const row=e.target.closest('.nb'); if(!row) return; const x=NBL[+row.dataset.i]; if(x) jumpTo(x.o.g,x.o.p); });
document.getElementById('card').addEventListener('change',e=>{ if(e.target.id==='npool') nPool=e.target.value; else if(e.target.id==='nrole') nRole=e.target.checked; else if(e.target.id==='ncarry') nCarry=e.target.checked; else return; if(curId!==null) card(curId); });
document.getElementById('carry').onchange=e=>{carryF=e.target.value;render()};
document.getElementById('dsize').onchange=e=>{dsize=e.target.value;render()};
document.getElementById('lvl').onchange=e=>{lvl=e.target.value;pinned=null;card(null);render()};
document.getElementById('find').onchange=e=>{ const q=e.target.value.trim().toLowerCase(); if(!q) return; const all=[...new Set(G.players.map(p=>p.n))]; let n=all.find(x=>x.toLowerCase()===q)||all.find(x=>x.toLowerCase().includes(q));
  if(n){ follow=n; const tp=G.players.filter(p=>p.n===n).sort((a,b)=>a.se<b.se?-1:1); if(!tp.some(p=>p.se===season)){season=tp[tp.length-1].se; ssel.value=season;} const cur=G.players.find(p=>p.n===n&&p.se===season); pinned=cur.id; card(pinned); render(); } };
document.getElementById('clear').onclick=()=>{follow=null;pinned=null;document.getElementById('find').value='';card(null);render()};
document.getElementById('save').onclick=()=>exportPng();
setGroup(Object.keys(D.groups)[0]);
</script></body></html>"""


def build_site(payload: dict, out_path) -> None:
    logo = Path(__file__).resolve().parent / 'images' / 'premier_league_logo.png'        # optional: drop your own copy here (not included in the repo)
    if logo.exists():
        payload = dict(payload, logo='data:image/png;base64,' + base64.b64encode(logo.read_bytes()).decode())
    html = TEMPLATE.replace('__DATA__', json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
    Path(out_path).write_text(html, encoding='utf-8')
