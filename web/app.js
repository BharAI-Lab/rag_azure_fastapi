// Elements
const form = document.getElementById("chat-form");
const input = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const clearBtn = document.getElementById("clear-btn");
const chat = document.getElementById("chat-window");
const jumpBtn = document.getElementById("jump-latest");
const themeToggle = document.getElementById("theme-toggle");
const aboutBtn = document.getElementById("about-btn");
const datasetBtn = document.getElementById("dataset-btn");
const viz3dBtn = document.getElementById("viz3d-btn");
const panel = document.getElementById("sidepanel");
const panelTitle = document.getElementById("panel-title");
const panelContent = document.getElementById("panel-content");
const panelClose = document.getElementById("panel-close");
const panelBackdrop = document.getElementById("panel-backdrop");

let userScrolledUp = false;
let loadingNode = null;
let lastSources = [];
let lastTurn = 0;

// ---------- Utilities ----------
function escapeHtml(str){
  return String(str).replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}
function mdToHtml(md){
  if(!md) return "";
  md = md.replace(/```([\s\S]*?)```/g, (_, code) => `<pre><code>${escapeHtml(code)}</code></pre>`);
  md = md.replace(/`([^`]+)`/g, (_, code) => `<code>${escapeHtml(code)}</code>`);
  md = md.replace(/^###\s+(.+)$/gm, (_, t) => `<h3>${escapeHtml(t)}</h3>`);
  md = md.replace(/^##\s+(.+)$/gm,  (_, t) => `<h2>${escapeHtml(t)}</h2>`);
  md = md.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>").replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
  md = md.replace(/^[\-\*]\s+(.+)$/gm, "<li>$1</li>");
  md = md.replace(/(?:^|\n)(<li>.*<\/li>(?:\n<li>.*<\/li>)*)/g, (_, list) => `<ul>${list}</ul>`);
  md = md.split(/\n{2,}/).map(block=>{
    const t = block.trim(); if(!t) return "";
    if(/^<h[23]>|^<ul>|^<pre>|^<p>/.test(t)) return t;
    return `<p>${t.replace(/\n/g,"<br/>")}</p>`;
  }).join("");
  return md;
}
function formatTime(d=new Date()){
  const hh = String(d.getHours()).padStart(2,'0');
  const mm = String(d.getMinutes()).padStart(2,'0');
  return `${hh}:${mm}`;
}
function atBottom(){ return chat.scrollTop + chat.clientHeight >= chat.scrollHeight - 24; }
function maybeAutoScroll(){
  if(!userScrolledUp || atBottom()){ chat.scrollTop = chat.scrollHeight; }
  else{ jumpBtn.style.display = "inline-block"; }
}

// Safe fallbacks for scores (if backend doesn’t return .score)
function getScores(sources){
  if(!Array.isArray(sources) || !sources.length) return [];
  const have = sources.every(s => typeof s.score === "number");
  if(have) return sources.map(s => Math.max(-1, Math.min(1, s.score)));
  const n = sources.length; // rank-based
  return sources.map((_, i) => 0.98 - (0.38 * i) / Math.max(1, n-1));
}

// ---------- Messages ----------
function makeMessage(role, html, sources=[], opts={}){
  const { isLoading=false, copyText=null, timestamp=new Date() } = opts;

  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "🙋" : (role === "system" ? "ℹ️" : "🤖");

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  if(isLoading) bubble.classList.add("loading");
  bubble.innerHTML = html;

  if(role === "bot"){
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "copy-btn";
    btn.title = "Copy answer";
    btn.textContent = "Copy";
    btn.addEventListener("click", async ()=>{
      try{
        const text = copyText ?? bubble.innerText;
        await navigator.clipboard.writeText(text);
        const old = btn.textContent; btn.textContent = "Copied!";
        setTimeout(()=>btn.textContent = old, 1000);
      }catch{
        btn.textContent = "Copy failed";
        setTimeout(()=>btn.textContent = "Copy", 1200);
      }
    });
    bubble.appendChild(btn);
  }

  // Inline sources accordion
  if(role === "bot" && Array.isArray(sources) && sources.length){
    const details = document.createElement("details");
    details.className = "sources";
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${sources.length})`;
    const list = document.createElement("ul");
    list.className = "source-list";

    sources.forEach(s=>{
      const li = document.createElement("li");
      li.className = "source-item";
      const title = document.createElement("div");
      title.className = "source-title";
      title.textContent = `${s.article_id || "Article"}${s.article_title ? " — " + s.article_title : ""}${Number.isInteger(s.chunk_id) ? " (chunk " + s.chunk_id + ")" : ""}`;
      const snippet = document.createElement("div");
      snippet.className = "source-snippet";
      snippet.textContent = (s.snippet || "").trim();
      li.appendChild(title); li.appendChild(snippet);
      list.appendChild(li);
    });
    details.appendChild(summary); details.appendChild(list);
    bubble.appendChild(details);
  }

  const time = document.createElement("div");
  time.className = "time";
  time.textContent = formatTime(timestamp);
  bubble.appendChild(time);

  wrapper.appendChild(avatar); wrapper.appendChild(bubble);
  chat.appendChild(wrapper);
  maybeAutoScroll();
  return wrapper;
}

function appendUser(text){
  makeMessage("user", `<p>${escapeHtml(text)}</p>`, [], { timestamp:new Date() });
}
function appendBot(answerMd, sources){
  makeMessage("bot", mdToHtml(answerMd), sources, { copyText: answerMd, timestamp:new Date() });
}

// ---------- Welcome ----------
function appendWelcome(){
  const html = mdToHtml(
`**Welcome to GDPR RAG Assistant**

Ask GDPR questions and get answers grounded in the GDPR data, with citations.

**Try:**
- Within how many hours must a controller notify a data breach?
- What are the lawful bases for processing under GDPR Article 6?
- What are the administrative fine tiers and assessment criteria in GDPR Article 83?`
  );
  makeMessage("system", html, [], { timestamp:new Date() });
}
appendWelcome();

// ---------- Loading ----------
function showLoading(){
  loadingNode = makeMessage("bot", `<p>Thinking…</p>`, [], { isLoading:true, timestamp:new Date() });
}
function hideLoading(){
  if(loadingNode?.parentNode){ loadingNode.parentNode.removeChild(loadingNode); }
  loadingNode = null;
}

// ---------- API ----------
async function send(message){
  const res = await fetch("/api/chat", {
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message})
  });
  if(!res.ok) throw new Error(`HTTP ${res.status}`);
  return await res.json();
}

// ---------- Form ----------
form.addEventListener("submit", async (e)=>{
  e.preventDefault();
  const q = (input.value || "").trim();
  if(!q) return;

  appendUser(q);
  input.value = ""; input.style.height = "auto";
  sendBtn.disabled = true;
  showLoading();

  try{
    const data = await send(q);
    hideLoading();
    appendBot(data.answer || "(no answer)", data.sources || []);
    lastSources = data.sources || [];
    lastTurn += 1;
  }catch(err){
    hideLoading();
    appendBot(`## Error\n- ${err.message}`);
  }finally{
    sendBtn.disabled = false;
    input.focus();
  }
});

// Enter sends; Shift+Enter newline; Cmd/Ctrl+K focus
input.addEventListener("keydown",(e)=>{
  if(e.key === "Enter" && !e.shiftKey){ e.preventDefault(); form.requestSubmit(); }
});
window.addEventListener("keydown",(e)=>{
  if((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k"){ e.preventDefault(); input.focus(); }
});

// Clear
clearBtn.addEventListener("click", ()=>{
  chat.innerHTML = ""; lastSources = []; lastTurn = 0;
  appendWelcome(); input.focus();
});

// Autosize textarea
input.addEventListener("input", ()=>{
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});

// Scroll + jump to latest
chat.addEventListener("scroll", ()=>{
  userScrolledUp = !atBottom();
  jumpBtn.style.display = userScrolledUp ? "inline-block" : "none";
});
jumpBtn.addEventListener("click", ()=>{
  chat.scrollTop = chat.scrollHeight; userScrolledUp = false; jumpBtn.style.display = "none";
});

// Theme
function applyTheme(t){
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("gdpr-theme", t);
}
(function initTheme(){
  const saved = localStorage.getItem("gdpr-theme");
  applyTheme(saved === "light" ? "light" : "dark");
})();
themeToggle.addEventListener("click", ()=>{
  const cur = document.documentElement.getAttribute("data-theme") || "dark";
  applyTheme(cur === "dark" ? "light" : "dark");
});

// ---------- Slide-in panel helpers ----------
function openPanel(title, innerHTML){
  panelTitle.textContent = title;
  panelContent.innerHTML = innerHTML;
  panel.setAttribute("aria-hidden","false");
  panelBackdrop.hidden = false;
}
function closePanel(){
  panel.setAttribute("aria-hidden","true");
  panelBackdrop.hidden = true;
}
panelClose.addEventListener("click", closePanel);
panelBackdrop.addEventListener("click", closePanel);
window.addEventListener("keydown",(e)=>{ if(e.key === "Escape" && panel.getAttribute("aria-hidden")==="false") closePanel(); });

// ---------- About button ----------
aboutBtn.addEventListener("click", ()=>{
  openPanel("About this Chatbot", `
    <p><strong>GDPR RAG Assistant</strong> answers questions using passages retrieved from your GDPR data.</p>
    <ul>
      <li>Retrieval-augmented: answers include citations.</li>
      <li>Clear fallback: if the dataset lacks an answer, you’ll be told and a general GPT-4o explanation will be shown.</li>
    </ul>
    <p><strong>Developed by:</strong>
        <ul>
          <li>Suraj Bhardwaj</li> 
          <li>Mukul Lambat</li>
          <li>Nikhil Manakali</li>
        </ul>
    </p>
  `);
});

// ---------- Dataset button ----------
datasetBtn.addEventListener("click", ()=>{
  openPanel("GDPR Dataset (Hugging Face)", `
    <p>Source: <a href="https://huggingface.co/datasets/AndreaSimeri/GDPR" target="_blank" rel="noopener">AndreaSimeri/GDPR</a></p>
    <h3>Abstract</h3>
    <p>The General Data Protection Regulation (GDPR) is an EU legal framework for data protection and privacy (in force since May 2018). It consists of 99 Articles (legal requirements) and 173 Recitals (context and interpretation). Articles span 11 chapters, from general provisions and principles to data subject rights, controller/processor obligations, international transfers, supervisory authorities, cooperation, remedies and penalties, specific processing situations, and final provisions.</p>
    <p>Recitals provide guidance to interpret Articles, e.g., Article 7(2) on conditions for consent is clarified by Recital 42 (proof of consent; informed identity and purposes).</p>
    <h3>Reference</h3>
    <pre><code>@inproceedings{SimeriGDPRLirai2023,
  author = {Andrea Simeri and Andrea Tagarelli},
  title  = {GDPR Article Retrieval ...},
  booktitle = {LIRAI 2023 @ HT 2023},
  series = {CEUR Workshop Proceedings},
  volume = {3594}, pages = {63-76}, year = {2023},
  url = {https://ceur-ws.org/Vol-3594/paper5.pdf}
}</code></pre>
    <p>Open the dataset: <a href="https://huggingface.co/datasets/AndreaSimeri/GDPR" target="_blank" rel="noopener">Hugging Face – GDPR</a></p>
  `);
});

// ---------- 3D Viz button (on demand, last turn) ----------
viz3dBtn.addEventListener("click", ()=>{
  const sources = lastSources || [];
  const scores = getScores(sources);
  openPanel("3D Retrieval Visualisation (last turn)", `
    <p>Showing query (at +Z) and Top-K retrieved chunks on the unit sphere (angle = arccos(similarity)).</p>
    <div id="viz3d" style="width:100%;height:320px"></div>
  `);

  // Build plot
  const axisColor = getComputedStyle(document.documentElement).getPropertyValue('--muted') || '#9fb3c8';
  const queryColor = '#c4ffd9';
  const docColor = '#19BC99';
  const n = scores.length;
  const xs=[0], ys=[0], zs=[1], texts=["Query"], colors=[queryColor];

  for(let i=0;i<n;i++){
    const s = Math.max(-1, Math.min(1, scores[i]));
    const theta = Math.acos(s);
    const phi = (2*Math.PI) * (i / Math.max(1,n));
    const x = Math.sin(theta)*Math.cos(phi);
    const y = Math.sin(theta)*Math.sin(phi);
    const z = Math.cos(theta);
    xs.push(x); ys.push(y); zs.push(z);
    const label = `${(sources[i]?.article_id || "Article")}${sources[i]?.article_title ? " — " + sources[i].article_title : ""}\nscore=${s.toFixed(3)}\nθ=${(theta*180/Math.PI).toFixed(1)}°`;
    texts.push(label); colors.push(docColor);
  }

  const pts = {type:"scatter3d", mode:"markers", x:xs, y:ys, z:zs, text:texts,
               hovertemplate:"%{text}<extra></extra>", marker:{size:4, color:colors}};
  const lines=[];
  for(let i=1;i<xs.length;i++){
    lines.push({type:"scatter3d", mode:"lines", x:[0,xs[i]], y:[0,ys[i]], z:[0,zs[i]],
      line:{width:2, color:docColor}, hoverinfo:"skip", showlegend:false});
  }
  const layout = {
    margin:{l:0,r:0,t:0,b:0},
    scene:{
      xaxis:{showgrid:false, zeroline:false, color:axisColor},
      yaxis:{showgrid:false, zeroline:false, color:axisColor},
      zaxis:{showgrid:false, zeroline:false, color:axisColor},
      camera:{eye:{x:1.15,y:1.15,z:0.8}}
    },
    paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)", showlegend:false
  };
  const el = document.getElementById("viz3d");
  Plotly.newPlot(el, [pts, ...lines], layout, {displayModeBar:false, responsive:true});
});
