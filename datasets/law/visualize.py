"""Render law claim graphs as a standalone, interactive HTML file.

Usage: python datasets/law/visualize.py [input.json] [-o output.html]
Defaults to example.json and example.html alongside this script. Multi-source /
multi-target relationships are drawn as all source-to-target pairs, retaining
the relationship ID, type, and analysis on every connection.
Add --serve to enable decision simulation and budget-based seed selection.
No third-party packages or internet connection are required.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from .decisions import GeneralThresholdModel
    from .analysis import greedy_selection, random_search
except ImportError:
    from decisions import GeneralThresholdModel
    from analysis import greedy_selection, random_search
from pathlib import Path


def render_html(data: dict | list, *, model_api: bool = False) -> str:
    """Validate graph references and embed records safely in the HTML viewer."""
    cases = data if isinstance(data, list) else [data]
    if not cases:
        raise ValueError("Input must contain at least one case")
    for case in cases:
        graph = case["information"]
        ids = [node["id"] for node in graph["nodes"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Node IDs must be unique within each case")
        for node in graph["nodes"]:
            for field in ("id", "justice", "claim"):
                if not isinstance(node[field], str):
                    raise ValueError(f"Node {field} must be a string")
        for edge in graph["edges"]:
            if not isinstance(edge["type"], str):
                raise ValueError("Edge type must be a string")
            for field in ("sources", "targets"):
                if not isinstance(edge[field], list) or not edge[field]:
                    raise ValueError(f"Edge {field} must be a nonempty list")
                for endpoint in edge[field]:
                    if endpoint not in ids:
                        raise ValueError(f"Unknown node reference: {endpoint}")
    # Escape HTML-sensitive characters so claims cannot terminate the data script.
    payload = json.dumps(cases, ensure_ascii=True).replace("<", "\\u003c").replace(
        ">", "\\u003e").replace("&", "\\u0026")
    return HTML.replace("__GRAPH_DATA__", payload).replace("__MODEL_API__", json.dumps(model_api))


HTML = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Law claim graphs</title>
<style>
*{box-sizing:border-box}body{margin:0;height:100vh;display:flex;flex-direction:column;font:15px system-ui,sans-serif;color:#1e293b;background:#f8fafc}
header{padding:20px 28px;background:white;border-bottom:1px solid #e2e8f0}h1{margin:0 0 12px;font-size:24px}
button,select,input{font:inherit;padding:7px 12px;border:1px solid #cbd5e1;border-radius:6px;background:white;color:inherit}#distance{width:70px}.model-controls{margin-top:12px}.model-controls input{width:85px}#model-results{margin-bottom:22px;font-size:14px;line-height:1.6}#model-results table{width:100%;border-collapse:collapse}#model-results td,#model-results th{text-align:left;border-bottom:1px solid #e2e8f0;padding:5px 2px}#model-status{font-size:13px}
button{cursor:pointer}.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
header{flex-shrink:0}main{display:flex;flex:1;min-height:0}aside{padding:22px;overflow:auto;background:white;flex-shrink:0}
#claims-panel{width:290px;border-right:1px solid #e2e8f0}#legend-panel{width:210px;border-left:1px solid #e2e8f0}
.claim-group{margin:22px 0}.claim-group h3{font-size:14px;margin:0 0 12px;display:flex;align-items:center;gap:8px}.claim-list{list-style:none;margin:0;padding:0}.claim-list li{border-bottom:1px solid #e2e8f0;padding:0 0 12px;margin:0 0 12px;font-size:14px;line-height:1.5;overflow-wrap:anywhere}.claim-id{display:block;font-weight:650;font-size:12px;color:#64748b;margin-bottom:4px}
h2{font-size:15px;margin:0 0 12px}.legend{display:flex;align-items:center;gap:10px;margin:10px 0}.swatch{width:16px;height:16px;border-radius:50%}
aside p{font-size:13px;line-height:1.6;color:#64748b}#graph{flex:1;min-width:0;touch-action:none;cursor:grab}
.node{cursor:pointer}.node text{fill:white;font-size:13px;font-weight:650;pointer-events:none}.node circle{stroke:white;stroke-width:2}.node:focus{outline:none}.node:focus-visible circle{stroke:#0f172a;stroke-width:4;stroke-dasharray:3 2}
.edge{fill:none;stroke:#64748b;stroke-width:2;pointer-events:none}.hit{fill:none;stroke:transparent;stroke-width:14;pointer-events:stroke;cursor:help}
#tip{display:none;position:fixed;max-width:380px;white-space:pre-wrap;pointer-events:none;background:#0f172a;color:white;padding:14px 16px;border-radius:8px;line-height:1.5;font-size:14px;box-shadow:0 8px 24px #0003;z-index:5}
@media(max-width:900px){#claims-panel{width:210px}#legend-panel{width:160px}aside{padding:12px}header{padding:12px}}
@media(max-width:650px){main{flex-wrap:wrap;overflow:auto}#claims-panel{width:60%;max-height:220px;order:1}#legend-panel{width:40%;max-height:220px;order:2}#graph{order:3;flex-basis:100%;height:65vh;min-height:350px}}
</style></head>
<body><header><h1 id="heading">Law claim graphs</h1><div class="controls">
<label>Case <select id="cases"></select></label><button id="fit">Fit view</button><button id="layout">Reset layout</button>
<button id="in" aria-label="Zoom in">+</button><button id="out" aria-label="Zoom out">−</button>
<label for="distance">Hop distance k <input id="distance" type="number" min="0" step="1" value="1"></label>
<button id="clear">Clear selection</button><span id="selection" role="status" aria-live="polite"></span>
<span id="count"></span></div>
<div class="controls model-controls">
<label>Budget <input id="budget" type="number" min="0" step="1" value="3"></label>
<label>Method <select id="method"><option value="greedy">Greedy</option><option value="random_search">Random search</option></select></label>
<label>Simulations <input id="simulations" type="number" min="1" max="10000" step="1" value="200"></label>
<label>Trials <input id="trials" type="number" min="1" max="10000" step="1" value="100"></label>
<label>Random seed <input id="random-seed" type="number" step="1" value="7"></label>
<label>Justice cutoff <input id="justice-cutoff" type="number" min="0" max="1" step="0.05" value="0.5"></label>
<button id="simulate">Simulate selection</button><button id="optimize">Find influential claims</button>
<span id="model-status" role="status" aria-live="polite"></span></div></header>
<main><aside id="claims-panel" aria-label="Selected claims"><h2>Model results</h2><div id="model-results" aria-live="polite"></div><h2>Selected claims</h2><div id="selected-claims"></div></aside>
<svg id="graph" role="img" aria-label="Interactive claim graph">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#64748b"/></marker></defs>
<g id="viewport"></g></svg>
<aside id="legend-panel" aria-label="Graph legends"><h2>Justices</h2><div id="justices"></div><h2>Relationships</h2><div id="types"></div></aside>
</main><div id="tip" role="tooltip"></div>
<script id="data" type="application/json">__GRAPH_DATA__</script>
<script>
"use strict";
const cases=JSON.parse(document.getElementById('data').textContent);
const $=id=>document.getElementById(id), svg=$('graph'), view=$('viewport'), tip=$('tip');
const ns='http://www.w3.org/2000/svg';
const palette=['#2563eb','#d97706','#059669','#9333ea','#dc2626','#0891b2','#be185d','#4f46e5'];
const known={support:'',dispute:'9 5',challenge:'2 5',alternative:'12 4 2 4',dependence:'12 4 2 4 2 4'};
let nodes=[],links=[],box,drag=null,selected=new Set(),hovered=null;
const modelAPI=__MODEL_API__;
let modelResult=null, modelVersion=0, modelBusy=false;
function el(tag,attrs,parent){const e=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);parent.append(e);return e;}
function show(event,text){tip.textContent=text;tip.style.display='block';const r=event.currentTarget.getBoundingClientRect();const x=event.clientX??r.x+r.width/2,y=event.clientY??r.y+r.height/2;tip.style.left=Math.max(8,Math.min(x+16,innerWidth-tip.offsetWidth-8))+'px';tip.style.top=Math.max(8,Math.min(y+16,innerHeight-tip.offsetHeight-8))+'px';}
function hover(e,text){e.setAttribute('tabindex','0');e.setAttribute('aria-label',text);e.addEventListener('pointermove',event=>show(event,text));e.addEventListener('pointerleave',()=>tip.style.display='none');e.addEventListener('focus',event=>show(event,text));e.addEventListener('blur',()=>tip.style.display='none');}
// Circular justice clusters; optimize cluster and node order using relationships.
function layoutGraph(nodes, links) {
    const groups = [], byJustice = new Map(), neighbors = new Map(nodes.map(n => [n, new Set()]));
    for (const n of nodes) {
        if (!byJustice.has(n.justice)) {
            const group = {justice:n.justice, nodes:[], weight:0};
            byJustice.set(n.justice, group); groups.push(group);
        }
        byJustice.get(n.justice).nodes.push(n);
    }
    for (const {source, target} of links) {
        neighbors.get(source).add(target); neighbors.get(target).add(source);
        if (source.justice !== target.justice) {
            byJustice.get(source.justice).weight++;
            byJustice.get(target.justice).weight++;
        }
    }
    for (const group of groups) {
        const count = group.nodes.length;
        // At least 85 units between neighboring nodes on each ring.
        group.ringRadius = count === 1 ? 0 : Math.max(65, 85/(2*Math.sin(Math.PI/count)));
        group.radius = group.ringRadius + 55;
        group.isolated = group.nodes.filter(n => !neighbors.get(n).size);
    }
    if (!groups.length) return {groups:[], isolated:[]};
    // The justice with the most cross-group relationships anchors the center.
    const hub = groups.reduce((best, g) => g.weight > best.weight ||
        (g.weight === best.weight && g.nodes.length > best.nodes.length) ? g : best, groups[0]);
    const satellites = groups.filter(g => g !== hub);
    const largestSatellite = Math.max(0, ...satellites.map(g => g.radius));
    const orbit = Math.max(hub.radius + largestSatellite + 110,
        satellites.length > 1 ? (2*largestSatellite+90)/(2*Math.sin(Math.PI/satellites.length)) : 0);
    hub.x = 0; hub.y = 0;
    function placeGroups() {
        satellites.forEach((g, i) => {
            const angle = -Math.PI/2 + i*2*Math.PI/satellites.length;
            g.x = orbit*Math.cos(angle); g.y = orbit*Math.sin(angle);
        });
    }
    const external = links.filter(l => l.source.justice !== l.target.justice);
    function groupCost() {
        return external.reduce((sum, l) => {
            const a = byJustice.get(l.source.justice), b = byJustice.get(l.target.justice);
            return sum + Math.hypot(a.x-b.x, a.y-b.y);
        }, 0);
    }
    placeGroups(); let best = groupCost();
    for (let pass = 0; pass < 6; pass++) {
        let improved = false;
        for (let i = 0; i < satellites.length; i++) for (let j = i+1; j < satellites.length; j++) {
            [satellites[i],satellites[j]] = [satellites[j],satellites[i]]; placeGroups();
            const cost = groupCost();
            if (cost < best-1e-9) { best=cost; improved=true; }
            else { [satellites[i],satellites[j]] = [satellites[j],satellites[i]]; placeGroups(); }
        }
        if (!improved) break;
    }
    const offsetX = 60-Math.min(...groups.map(g => g.x-g.radius));
    const offsetY = 90-Math.min(...groups.map(g => g.y-g.radius));
    for (const group of groups) { group.x+=offsetX; group.y+=offsetY; }
    function placeNodes(group) {
        group.nodes.forEach((n, i) => {
            const angle = -Math.PI/2 + i*2*Math.PI/group.nodes.length;
            n.x = group.x+group.ringRadius*Math.cos(angle);
            n.y = group.y+group.ringRadius*Math.sin(angle);
        });
    }
    groups.forEach(placeNodes);
    // Prefer short links, few crossings, and lines that avoid unrelated nodes.
    // Limit crossing evaluation on very dense graphs to keep layout responsive.
    const scoredLinks = links.slice(0, 150);
    function orientation(a,b,c) { return (b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x); }
    function score() {
        let cost = scoredLinks.reduce((sum,l) => sum+Math.hypot(l.source.x-l.target.x,l.source.y-l.target.y)/85, 0);
        for (let i=0; i<scoredLinks.length; i++) {
            const a=scoredLinks[i].source, b=scoredLinks[i].target;
            const dx=b.x-a.x, dy=b.y-a.y, lengthSquared=dx*dx+dy*dy;
            for (const n of nodes) if (n!==a && n!==b && lengthSquared) {
                const t=((n.x-a.x)*dx+(n.y-a.y)*dy)/lengthSquared;
                if (t>0 && t<1 && Math.hypot(n.x-a.x-t*dx,n.y-a.y-t*dy)<40) cost+=80;
            }
            for (let j=i+1; j<scoredLinks.length; j++) {
                const c=scoredLinks[j].source, d=scoredLinks[j].target;
                if (a===c || a===d || b===c || b===d) continue;
                if (orientation(a,b,c)*orientation(a,b,d)<0 && orientation(c,d,a)*orientation(c,d,b)<0) cost+=40;
            }
        }
        return cost;
    }
    let current = score();
    for (let pass=0; pass<6; pass++) {
        let improved=false;
        for (const group of groups) for (let i=0; i<group.nodes.length; i++) {
            const limit = group.nodes.length > 30 ? Math.min(group.nodes.length,i+2) : group.nodes.length;
            for (let j=i+1; j<limit; j++) {
                const list=group.nodes;
                [list[i],list[j]]=[list[j],list[i]]; placeNodes(group);
                const candidate=score();
                if (candidate<current-1e-9) { current=candidate; improved=true; }
                else { [list[i],list[j]]=[list[j],list[i]]; placeNodes(group); }
            }
        }
        if (!improved) break;
    }
    return {groups, isolated:nodes.filter(n => !neighbors.get(n).size)};
}
// Multi-source BFS gives the union of undirected k-hop neighborhoods and
// measures each node's distance to its nearest selected node.
function selectedNeighborhood(nodes, links, selected, k) {
    const adjacency = new Map(nodes.map(n => [n, new Set()]));
    for (const {source, target} of links) {
        adjacency.get(source).add(target); adjacency.get(target).add(source);
    }
    const distances = new Map([...selected].map(n => [n, 0]));
    const queue = [...selected];
    for (let i = 0; i < queue.length; i++) {
        const current = queue[i], depth = distances.get(current);
        if (depth >= k) continue;
        for (const next of adjacency.get(current)) if (!distances.has(next)) {
            distances.set(next, depth + 1); queue.push(next);
        }
    }
    return distances;
}
function shade(color, distance, k) {
    // Keep each justice's hue, mixing white into more distant nodes.
    const white = k ? .72 * distance / k : 0;
    const rgb = color.match(/\d+/g).map(Number);
    return `rgb(${rgb.map(channel => Math.round(channel + (255-channel)*white)).join(',')})`;
}
function highlight(node=hovered) {
    const active = selected.size > 0;
    const anchors = active ? selected : new Set(node ? [node] : []);
    const k = active ? Number($('distance').value) : 1;
    const matches = selectedNeighborhood(nodes, links, anchors, k);
    for (const n of nodes) {
        const circle = n.element.querySelector('circle'), text = n.element.querySelector('text');
        const isSelected = selected.has(n), matched = matches.has(n);
        const adopted=modelResult?.active_nodes.includes(n.id);
        n.element.style.opacity = modelResult ? (adopted ? 1 : .18) : (!anchors.size || matched || isSelected ? 1 : .18);
        circle.setAttribute('fill', active && matched ? shade(n.shadeColor, matches.get(n), k) : n.color);
        circle.style.stroke = isSelected ? '#0f172a' : '';
        circle.style.strokeWidth = isSelected ? 4 : (adopted ? 3 : '');
        if (adopted && !isSelected) circle.style.stroke='#059669';
        text.style.fill = active && matched && k && matches.get(n)/k > .45 ? '#0f172a' : 'white';
        n.element.setAttribute('aria-pressed', String(isSelected));
    }
    for (const l of links) {
        const visible = !anchors.size || (matches.has(l.source) && matches.has(l.target));
        l.line.style.opacity = visible ? 1 : .1;
        l.line.style.strokeWidth = anchors.size && visible ? 3 : 2;
    }
    $('selection').textContent = active
        ? `${selected.size} selected · ${matches.size} within ${k} hops of any selection`
        : 'Click nodes to select';
    $('clear').disabled = !active;
}
function renderSelectedClaims() {
    const panel = $('selected-claims');
    panel.replaceChildren();
    const groups = new Map();
    for (const node of nodes) if (selected.has(node)) {
        if (!groups.has(node.justice)) groups.set(node.justice, []);
        groups.get(node.justice).push(node);
    }
    if (!groups.size) {
        const empty = document.createElement('p');
        empty.textContent = 'Click nodes to select claims.';
        panel.append(empty);
    }
    for (const [justice, claims] of groups) {
        const section = document.createElement('section'); section.className = 'claim-group';
        const heading = document.createElement('h3');
        const swatch = document.createElement('span'); swatch.className = 'swatch';
        swatch.style.background = claims[0].color;
        heading.append(swatch, document.createTextNode(`${justice} (${claims.length})`));
        const list = document.createElement('ul'); list.className = 'claim-list';
        for (const claim of claims) {
            const item = document.createElement('li');
            const id = document.createElement('span'); id.className = 'claim-id'; id.textContent = claim.id;
            item.append(id, document.createTextNode(claim.claim)); list.append(item);
        }
        section.append(heading, list); panel.append(section);
    }
}
function toggleSelection(node) {
    invalidateModel();
    if (selected.has(node)) selected.delete(node); else selected.add(node);
    highlight(); renderSelectedClaims();
}
function resetLayout() {
    const result = layoutGraph(nodes, links);
    view.querySelector('.justice-groups')?.remove();
    const background = el('g', {class:'justice-groups', 'pointer-events':'none'}, view);
    view.insertBefore(background, view.firstChild);
    for (const group of result.groups) {
        const color = group.nodes[0].color;
        el('circle', {cx:group.x, cy:group.y, r:group.radius,
            fill:color, 'fill-opacity':.045, stroke:color, 'stroke-opacity':.25}, background);
        const label = el('text', {x:group.x, y:group.y-group.radius-15, fill:color,
            'text-anchor':'middle', 'font-size':17, 'font-weight':650}, background);
        label.textContent = group.justice;
        const caption = el('text', {x:group.x, y:group.y+6,
            'text-anchor':'middle', fill:'#64748b', 'font-size':12}, background);
        if (group.nodes.length > 1)
            caption.textContent = `${group.nodes.length} claims${group.isolated.length ? ' · '+group.isolated.length+' unconnected' : ''}`;
    }
    update(); fit();
}
function update(){for(const n of nodes)n.element.setAttribute('transform',`translate(${n.x},${n.y})`);for(const l of links){const a=l.source,b=l.target,dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||1;const x1=a.x+dx/d*26,y1=a.y+dy/d*26,x2=b.x-dx/d*29,y2=b.y-dy/d*29;const bend=l.bend;const path=a===b?`M ${a.x-18} ${a.y-18} C ${a.x-80} ${a.y-100},${a.x+80} ${a.y-100},${a.x+18} ${a.y-18}`:`M ${x1} ${y1} Q ${(x1+x2)/2-dy/d*bend} ${(y1+y2)/2+dx/d*bend} ${x2} ${y2}`;l.line.setAttribute('d',path);l.hit.setAttribute('d',path);}}
function applyBox(){svg.setAttribute('viewBox',`${box.x} ${box.y} ${box.w} ${box.h}`);}
function fit(){const r=svg.getBoundingClientRect(),ratio=r.width/r.height||1;const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y);for(const circle of view.querySelectorAll('.justice-groups circle')){const x=Number(circle.getAttribute('cx')),y=Number(circle.getAttribute('cy')),r=Number(circle.getAttribute('r'));xs.push(x-r,x+r);ys.push(y-r-30,y+r);}const left=Math.min(0,...xs)-100,top=Math.min(0,...ys)-100;const contentWidth=Math.max(200,...xs)-left+100,contentHeight=Math.max(200,...ys)-top+100;let w=contentWidth,h=contentHeight;if(w/h<ratio)w=h*ratio;else h=w/ratio;box={x:left-(w-contentWidth)/2,y:top-(h-contentHeight)/2,w,h};applyBox();}
function zoom(factor,point){const p=point??{x:box.x+box.w/2,y:box.y+box.h/2};box={x:p.x+(box.x-p.x)*factor,y:p.y+(box.y-p.y)*factor,w:box.w*factor,h:box.h*factor};applyBox();}
function point(event){return new DOMPoint(event.clientX,event.clientY).matrixTransform(svg.getScreenCTM().inverse());}
function render(index){invalidateModel();selected.clear();hovered=null;drag=null;const c=cases[index],g=c.information;view.replaceChildren();$('justices').replaceChildren();$('types').replaceChildren();tip.style.display='none';$('heading').textContent=c.name??c.case_name??'Law claim graph';
const justices=[...new Set(g.nodes.map(n=>n.justice))],types=[...new Set(g.edges.map(e=>e.type))],colors={},styles={};
justices.forEach((j,i)=>{colors[j]=i<palette.length?palette[i]:`hsl(${i*137.508%360} 65% 38%)`;const row=document.createElement('div');row.className='legend';const dot=document.createElement('span');dot.className='swatch';dot.style.background=colors[j];row.append(dot,document.createTextNode(j));$('justices').append(row);});
types.forEach((t,i)=>{styles[t]=known[t]??`${14+i*2} 5 2 5`;const row=document.createElement('div');row.className='legend';const icon=el('svg',{width:42,height:16},row);el('line',{x1:0,y1:8,x2:42,y2:8,stroke:'#64748b','stroke-width':2,'stroke-dasharray':styles[t]},icon);row.append(document.createTextNode(t));$('types').append(row);});
nodes=g.nodes.map(n=>({...n,color:colors[n.justice],x:0,y:0}));const byId=new Map(nodes.map(n=>[n.id,n]));links=[];
const edgeLayer=el('g',{},view),nodeLayer=el('g',{},view),pairs=new Map();
for(const e of g.edges)for(const s of e.sources)for(const t of e.targets){const key=JSON.stringify([s,t].sort()),order=pairs.get(key)??0;pairs.set(key,order+1);const line=el('path',{class:'edge','stroke-dasharray':styles[e.type],'marker-end':'url(#arrow)'},edgeLayer),hit=el('path',{class:'hit'},edgeLayer);hover(hit,`${e.id??'Relationship'} · ${e.type}\n${s} → ${t}\nSources: ${e.sources.join(', ')}\nTargets: ${e.targets.join(', ')}${e.analysis?'\n\n'+e.analysis:''}`);links.push({source:byId.get(s),target:byId.get(t),line,hit,bend:order*35});}
for (const n of nodes) {
    n.element = el('g', {class:'node', role:'button', 'aria-pressed':'false'}, nodeLayer);
    const circle = el('circle', {r:25, fill:colors[n.justice]}, n.element);
    n.shadeColor = getComputedStyle(circle).fill;
    el('text', {'text-anchor':'middle', 'dominant-baseline':'central'}, n.element).textContent = n.id;
    hover(n.element, `${n.id} · ${n.justice}${n.position?' · '+n.position:''}\n\n${n.claim}`);
    n.element.addEventListener('pointerenter', () => { hovered=n; highlight(); });
    n.element.addEventListener('pointerleave', () => { hovered=null; highlight(); });
    n.element.addEventListener('focus', () => { hovered=n; highlight(); });
    n.element.addEventListener('blur', () => { hovered=null; highlight(); });
    n.element.addEventListener('keydown', event => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault(); if (!event.repeat) toggleSelection(n);
        }
    });
    n.element.addEventListener('pointerdown', event => {
        if (event.button !== 0) return;
        event.stopPropagation();
        const p=point(event);
        drag={node:n, clientX:event.clientX, clientY:event.clientY, moved:false,
            offsetX:n.x-p.x, offsetY:n.y-p.y};
        n.element.focus(); svg.setPointerCapture(event.pointerId); tip.style.display='none';
    });
}
$('budget').max=nodes.length;
$('budget').value=Math.min(Number($('budget').value),nodes.length);
$('distance').max = Math.max(0, nodes.length-1);
$('distance').value = Math.min(Number($('distance').value), Number($('distance').max));
$('count').textContent=`${nodes.length} claims · ${g.edges.length} relationships · ${links.length} connections`;resetLayout();highlight();renderSelectedClaims();}
cases.forEach((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=c.name??c.case_name??`Case ${i+1}`;$('cases').append(o);});
$('cases').addEventListener('change',e=>render(Number(e.target.value)));$('fit').onclick=fit;$('layout').onclick=resetLayout;$('in').onclick=()=>zoom(.8);$('out').onclick=()=>zoom(1.25);
svg.addEventListener('wheel',event=>{event.preventDefault();zoom(event.deltaY>0?1.1:1/1.1,point(event));},{passive:false});
svg.addEventListener('pointerdown',event=>{if(event.button!==0)return;drag={start:point(event),box:{...box}};svg.setPointerCapture(event.pointerId);});
svg.addEventListener('pointermove',event=>{if(!drag)return;const p=point(event);if(drag.node){if(Math.hypot(event.clientX-drag.clientX,event.clientY-drag.clientY)>4)drag.moved=true;if(drag.moved){drag.node.x=p.x+drag.offsetX;drag.node.y=p.y+drag.offsetY;update();}}else{box.x-=p.x-drag.start.x;box.y-=p.y-drag.start.y;applyBox();}});
svg.addEventListener('pointerup',()=>{if(drag?.node&&!drag.moved)toggleSelection(drag.node);drag=null;});svg.addEventListener('pointercancel',()=>drag=null);svg.addEventListener('lostpointercapture',()=>drag=null);
$('distance').addEventListener('input', () => {
    const input=$('distance');
    if (!input.value || !Number.isFinite(input.valueAsNumber)) return;
    input.value=Math.max(0,Math.min(Number(input.max),Math.floor(input.valueAsNumber)));
    highlight();
});
$('distance').addEventListener('change', () => {
    if (!$('distance').value) $('distance').value=1;
    $('distance').dispatchEvent(new Event('input'));
});
$('clear').onclick=()=>{invalidateModel();selected.clear();hovered=null;highlight();renderSelectedClaims();};
function invalidateModel() {
    modelVersion++; modelResult=null;
    $('model-results').replaceChildren();
    $('model-status').textContent=modelAPI ? '' : 'To enable model controls, run visualize.py --serve and open the printed URL.';
}
function modelNumber(id, integer=false) {
    const input=$(id), value=input.valueAsNumber;
    if (!input.value || !input.checkValidity() || !Number.isFinite(value) || (integer && !Number.isInteger(value)))
        throw new Error(`Enter a valid ${input.parentElement.textContent.trim()}.`);
    return value;
}
function resultParagraph(text) {
    const p=document.createElement('p'); p.textContent=text; $('model-results').append(p);
}
function showModelResult(result, solution) {
    $('model-results').replaceChildren();
    if(solution) resultParagraph(`${solution.method==='greedy'?'Greedy':'Random search'} selected ${solution.selected_nodes.join(', ') || 'no claims'}. Estimated spread: ${solution.expected_spread.toFixed(2)} of ${nodes.length} claims (standard error ${solution.standard_error===null?'unavailable':solution.standard_error.toFixed(2)}; ${solution.simulations} simulations).`);
    resultParagraph(`This simulation activated ${result.spread} of ${nodes.length} claims in ${result.rounds.length-1} diffusion rounds. Green outlines mark adopted claims; black outlines mark seeds.`);
    const table=document.createElement('table');
    const header=document.createElement('tr');
    for(const label of ['Justice','Decision','Active']) {const th=document.createElement('th');th.scope='col';th.textContent=label;header.append(th);}
    table.append(header);
    for(const [name,vote] of Object.entries(result.justices)) {
        const row=document.createElement('tr');
        for(const text of [name,vote.decision,`${vote.active_count}/${vote.total_count}`]) {const td=document.createElement('td');td.textContent=text;row.append(td);}
        table.append(row);
    }
    $('model-results').append(table);
    resultParagraph('Model assumptions: directed support edges; influence equals the fraction of active incoming neighbors. Votes use the selected justice cutoff. These are simulated adoption decisions.');
    if(solution) resultParagraph('The objective is total active claims. These sampled heuristics do not guarantee optimality; the greedy approximation guarantee requires submodular influence and accurate spread estimates.');
}
async function runModel(action) {
    if(!modelAPI || modelBusy) return;
    const version=modelVersion;
    try {
        const request={action,case_index:Number($('cases').value),selected_nodes:[...selected].map(n=>n.id),
            random_seed:modelNumber('random-seed',true),justice_threshold:modelNumber('justice-cutoff')};
        if(action==='optimize') Object.assign(request,{budget:modelNumber('budget',true),method:$('method').value,
            simulations:modelNumber('simulations',true),trials:modelNumber('trials',true)});
        modelBusy=true; $('simulate').disabled=true; $('optimize').disabled=true;
        $('model-status').textContent=action==='optimize'?'Finding influential claims…':'Simulating…';
        const response=await fetch('/api/model',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
        const payload=await response.json();
        if(version!==modelVersion) return;
        if(!response.ok) throw new Error(payload.error || 'Model request failed');
        if(payload.solution) {
            const ids=new Set(payload.solution.selected_nodes);
            selected=new Set(nodes.filter(n=>ids.has(n.id))); renderSelectedClaims();
        }
        modelResult=payload.result; showModelResult(payload.result,payload.solution); highlight();
        $('model-status').textContent='Model results ready.';
    } catch(error) {if(version===modelVersion) $('model-status').textContent=error.message;}
    finally {modelBusy=false; $('simulate').disabled=!modelAPI; $('optimize').disabled=!modelAPI;}
}
$('simulate').onclick=()=>runModel('simulate');
$('optimize').onclick=()=>runModel('optimize');
for(const id of ['budget','method','simulations','trials','random-seed','justice-cutoff'])
    $(id).addEventListener('change',()=>{invalidateModel();highlight();});
$('simulate').disabled=!modelAPI; $('optimize').disabled=!modelAPI;
$('method').addEventListener('change',()=>{$('trials').disabled=$('method').value!=='random_search';});
$('trials').disabled=true;
render(0);
</script></body></html>'''



def model_request(cases: list[dict], request: dict) -> dict:
    """Dispatch the browser's request to the Python model and heuristics."""
    index = request["case_index"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(cases):
        raise ValueError("Invalid case index")
    seed = request.get("random_seed", 7)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Random seed must be an integer")
    model = GeneralThresholdModel(cases[index])
    options = {"random_seed": seed, "justice_threshold": request.get("justice_threshold", 0.5)}
    if request["action"] == "simulate":
        return {"result": model.simulate(request["selected_nodes"], **options)}
    if request["action"] != "optimize":
        raise ValueError("Unknown model action")
    methods = {"greedy": greedy_selection, "random_search": random_search}
    method = request.get("method", "greedy")
    if method not in methods:
        raise ValueError("Unknown selection method")
    simulations = request.get("simulations", 200)
    trials = request.get("trials", 100)
    for name, value in (("simulations", simulations), ("trials", trials)):
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10000:
            raise ValueError(f"{name} must be an integer between 1 and 10000")
    kwargs = {"simulations": simulations, "random_seed": seed}
    if method == "random_search":
        kwargs["trials"] = trials
    solution = methods[method](model, request["budget"], **kwargs)
    return {"solution": solution, "result": model.simulate(solution["selected_nodes"], **options)}


def create_server(data: dict | list, port: int = 8000) -> ThreadingHTTPServer:
    """Serve the viewer and model API locally, without third-party dependencies."""
    html = render_html(data, model_api=True).encode("utf-8")
    cases = data if isinstance(data, list) else [data]

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            host = f"127.0.0.1:{self.server.server_port}"
            return (self.headers.get("Host") == host and
                    self.headers.get("Origin", f"http://{host}") == f"http://{host}")

        def do_GET(self):
            if not self.allowed():
                self.respond(403, b"Forbidden", "text/plain")
            elif self.path == "/":
                self.respond(200, html, "text/html; charset=utf-8")
            else:
                self.respond(404, b"Not found", "text/plain")

        def do_POST(self):
            if not self.allowed():
                self.respond(403, b"Forbidden", "text/plain")
                return
            if self.path != "/api/model":
                self.respond(404, b"Not found", "text/plain")
                return
            try:
                if self.headers.get("Content-Type") != "application/json":
                    raise ValueError("Expected application/json")
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 1000000:
                    raise ValueError("Invalid request length")
                request = json.loads(self.rfile.read(length))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")
                payload = model_request(cases, request)
                status = 200
            except (ValueError, KeyError, TypeError, OverflowError) as exc:
                payload, status = {"error": str(exc)}, 400
            self.respond(status, json.dumps(payload).encode("utf-8"), "application/json")

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).with_name("example.json"))
    parser.add_argument("-o", "--output", type=Path, help="Output HTML (defaults to input stem.html)")
    parser.add_argument("--serve", action="store_true", help="Serve locally with interactive Python model controls")
    parser.add_argument("--port", type=int, default=8000, help="Local server port (default: 8000)")
    args = parser.parse_args(argv)
    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        if args.serve:
            with create_server(data, args.port) as server:
                print(f"Open http://127.0.0.1:{server.server_port}/ (Ctrl+C to stop)", flush=True)
                try:
                    server.serve_forever()
                except KeyboardInterrupt:
                    pass
            return 0
        html = render_html(data)
        output = args.output or args.input.with_suffix(".html")
        if output.resolve() == args.input.resolve():
            raise ValueError("Output must differ from input")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(html, encoding="utf-8")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    print(f"Saved {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
