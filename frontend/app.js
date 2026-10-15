const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const views = ["overview","prediction","dataset","models","evaluation","monitoring","workflow"];
const stages = [
  {title:"Upload and validate dataset",desc:"Start with the official Bank Marketing dataset or inspect a CSV/XLSX of your own.",bullets:["Check accepted file format and schema","Inspect dimensions and sample rows","Keep exploratory uploads separate from the deployed model"],view:"dataset"},
  {title:"Feature inventory",desc:"Understand which columns are available and how each field should be interpreted.",bullets:["Inspect column names and inferred data types","Identify target and post-outcome fields","Exclude duration from pre-call predictions"],view:"dataset"},
  {title:"Descriptive statistics",desc:"Summarize distributions before modelling.",bullets:["Review numeric ranges and central tendency","Compare customer and campaign characteristics","Check unusual values and high-cardinality categories"],view:"dataset"},
  {title:"Missing-value analysis",desc:"Measure data completeness and choose defensible handling policies.",bullets:["Count missing values by column","Inspect duplicate records","Avoid fitting imputation statistics on the test set"],view:"dataset"},
  {title:"Visualize the data",desc:"Explore target balance and feature relationships.",bullets:["Inspect class imbalance","Review feature importance from the trained model","Use the data profile to guide further analysis"],view:"evaluation"},
  {title:"Preprocess safely",desc:"Prepare features without leaking information from held-out data.",bullets:["Keep transformations inside the training pipeline","Use stratified splits for the current dataset","Keep post-call duration excluded"],view:"models"},
  {title:"Train and compare models",desc:"Inspect the current model candidates and validation-based selection.",bullets:["Compare validation average precision","Review the selected champion","Fresh training jobs are not yet exposed in this deployed UI"],view:"models"},
  {title:"Evaluate models",desc:"Read held-out metrics and probability-quality diagnostics.",bullets:["Review precision, recall, F1, ROC-AUC and average precision","Compare the tuned threshold with 0.5","Inspect calibration and feature importance"],view:"evaluation"},
  {title:"Score a customer",desc:"Submit a profile to the deployed inference API.",bullets:["Enter pre-call customer and campaign attributes","Get probability and threshold-based classification","Treat the output as decision support, not a guarantee"],view:"prediction"}
];
let report = null;
let currentStage = 0;

function showView(name, label) {
  $$(".view").forEach(v => v.classList.toggle("active", v.id === `view-${name}`));
  $$(".nav-item").forEach(b => b.classList.toggle("active", (b.dataset.view || "") === name && name !== "workflow"));
  $("#page-crumb").textContent = label || (name === "workflow" ? "ML workbench" : name[0].toUpperCase()+name.slice(1));
  if (name === "workflow") renderStage();
  $("#sidebar").classList.remove("open");
  window.scrollTo({top:0,behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth"});
}
$$("[data-view]").forEach(b => b.addEventListener("click", () => showView(b.dataset.view)));
$$("[data-go]").forEach(b => b.addEventListener("click", () => showView(b.dataset.go)));
$("#mobile-menu").addEventListener("click", () => $("#sidebar").classList.toggle("open"));
$$("[data-stage]").forEach(b => b.addEventListener("click", () => {currentStage=Number(b.dataset.stage)-1;showView("workflow",stages[currentStage].title);$$("[data-stage]").forEach(x=>x.classList.toggle("active",x===b));}));
function fmt(value, digits=3){const n=Number(value);return Number.isFinite(n)?n.toFixed(digits):"—";}
function pct(value,digits=1){const n=Number(value);return Number.isFinite(n)?(n*100).toFixed(digits)+"%":"—";}
function esc(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function metricBars(items){return items.map(([label,value])=>{const n=Number(value);return `<div><div class="metric-row-head"><span>${esc(label)}</span><strong>${Number.isFinite(n)?fmt(n,3):"—"}</strong></div><div class="bar-track"><div class="bar-fill" style="width:${Number.isFinite(n)?Math.max(0,Math.min(100,n*100)):0}%"></div></div></div>`;}).join("");}
function featureMarkup(items){const values=items.map(x=>Number(x.importance??x.value??x.mean_abs_importance??0));const max=Math.max(...values,0.000001);return items.slice(0,10).map((x,i)=>{const val=values[i];const name=x.feature??x.name??x.feature_name??"Feature";return `<div class="feature-item"><span class="feature-name">${esc(String(name).replace(/^(numeric|categorical)__/, ""))}</span><div class="feature-track"><span style="width:${Math.max(2,Math.min(100,val/max*100))}%"></span></div><span class="feature-value">${fmt(val,3)}</span></div>`;}).join("")||'<div class="empty-state">No feature importance in the report.</div>';}
function renderReport(data){
 report=data;
 const metrics=data.final_test_metrics||{};
 const base=data.final_test_metrics_at_0_5||{};
 const rows=Number(data.rows);
 $("#kpi-rows").textContent=Number.isFinite(rows)?rows.toLocaleString():"—";
 $("#kpi-rate").textContent=data.positive_rate!=null?pct(data.positive_rate):"11.7%";
 $("#kpi-auc").textContent=fmt(metrics.roc_auc,3);
 $("#kpi-f1").textContent=fmt(metrics.f1,3);
 $("#champion-name").textContent=data.selected_model||"Not available";
 $("#selection-metric").textContent=data.selection_metric||"Average precision";
 const threshold=Number(data.threshold_selection?.threshold??data.threshold_selection?.selected_threshold);
 $("#decision-threshold").textContent=Number.isFinite(threshold)?pct(threshold,0):"—";
 $("#overview-metrics").innerHTML=metricBars([["Average precision",metrics.average_precision],["ROC-AUC",metrics.roc_auc],["Precision",metrics.precision],["Recall",metrics.recall],["F1 score",metrics.f1]]);
 const features=data.top_encoded_feature_importance||[];
 $("#feature-list").innerHTML=featureMarkup(features);
 $("#evaluation-features").innerHTML=featureMarkup(features);
 $("#eval-auc").textContent=fmt(metrics.roc_auc,3);
 $("#eval-precision").textContent=fmt(metrics.precision,3);
 $("#eval-recall").textContent=fmt(metrics.recall,3);
 $("#eval-brier").textContent=fmt(metrics.brier_score,3);
 $("#threshold-display").textContent=Number.isFinite(threshold)?pct(threshold,0):"—";
 $("#threshold-copy").textContent=data.threshold_selection?.method||"Operating threshold chosen on validation data; test data is reserved for final reporting.";
 $("#baseline-f1").textContent=fmt(base.f1,3);
 $("#tuned-f1").textContent=fmt(metrics.f1,3);
 $("#tuned-recall").textContent=pct(metrics.recall,1);
 $("#tuned-precision").textContent=pct(metrics.precision,1);
 $("#data-rows").textContent=Number.isFinite(rows)?rows.toLocaleString():"—";
 $("#data-rate").textContent=data.positive_rate!=null?pct(data.positive_rate):"11.7%";
 const candidates=data.validation_model_comparison||[];
 $("#model-count").textContent=`${candidates.length} candidate models`;
 $("#model-table").innerHTML=candidates.map(x=>{const name=x.model??x.name??x.model_name??"Candidate";const ap=x.average_precision??x.ap??x.validation_average_precision;const auc=x.roc_auc??x.roc_auc_score;const f1=x.f1??x.validation_f1;const selected=String(name)===String(data.selected_model);return `<tr><td>${esc(name)}</td><td>${fmt(ap,3)}</td><td>${fmt(auc,3)}</td><td>${fmt(f1,3)}</td><td class="table-status">${selected?"CHAMPION":"CANDIDATE"}</td></tr>`;}).join("")||'<tr><td colspan="5">Candidate metrics are not included in the current report.</td></tr>';
}
async function loadHealth(){
 try{
  const res=await fetch("/health",{cache:"no-store"});const data=await res.json();if(!res.ok)throw new Error("Health endpoint unavailable");
  const loaded=Boolean(data.model_loaded);$("#api-status").textContent=loaded?"API and model ready":"API online · model unavailable";$("#sidebar-health").textContent=loaded?"Model ready":"Model unavailable";$("#sidebar-status").classList.toggle("online",loaded);
  $("#health-api").textContent=data.status==="ok"?"Online":"Unavailable";$("#health-model").textContent=loaded?"Available":"Missing";
  $("#health-badge").textContent=loaded?"Service available":"Model unavailable";$("#health-badge").className="pill "+(loaded?"pill-green":"pill-amber");
  $("#health-checks").innerHTML=[["API liveness",data.status==="ok"?"Passing":"Unavailable"],["Model artifact",loaded?"Available":"Missing"],["Prediction endpoint","Configured"],["Prediction logging","Not configured"],["Drift monitoring","Not configured"]].map(([a,b])=>`<div class="check-row"><span>${a}</span><span class="check-state ${["Not configured","Missing","Unavailable"].includes(b)?"warn":""}">${b}</span></div>`).join("");
 }catch(e){$("#api-status").textContent="API connection unavailable";$("#sidebar-health").textContent="API unavailable";$("#sidebar-status").classList.remove("online");$("#health-api").textContent="Unavailable";}
}
async function loadReport(){
 try{const res=await fetch("/model-info",{cache:"no-store"});if(!res.ok)throw new Error("Report unavailable");const data=await res.json();renderReport(data);$("#health-report").textContent="Available";}
 catch(e){$("#health-report").textContent="Unavailable";$("#overview-metrics").innerHTML='<div class="empty-state">Model report unavailable. Check the deployment logs.</div>';$("#feature-list").innerHTML='<div class="empty-state">Feature report unavailable.</div>';}
}
$("#refresh-health").addEventListener("click",()=>{loadHealth();loadReport();});
const form=$("#prediction-form");
form.addEventListener("submit",async event=>{
 event.preventDefault();const btn=$("#submit-button");const error=$("#result-error");error.hidden=true;btn.disabled=true;btn.innerHTML="Scoring customer…";
 const values=Object.fromEntries(new FormData(form).entries());for(const k of ["age","balance","day","campaign","pdays","previous"])values[k]=Number(values[k]);
 try{const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({customer:values})});const data=await res.json();if(!res.ok)throw new Error(data.detail||"Prediction failed");
 const probability=Math.max(0,Math.min(1,Number(data.subscription_probability)));const percentage=(probability*100).toFixed(1);$("#result-empty").hidden=true;$("#result-content").hidden=false;$("#probability-value").textContent=percentage;$("#meter-fill").style.width=percentage+"%";
 $("#outcome").textContent=data.predicted_subscription?"Above selected threshold":"Below selected threshold";$("#outcome").classList.toggle("negative",!data.predicted_subscription);$("#threshold-value").textContent=pct(data.threshold,0);$("#prediction-model").textContent=data.selected_model||"Current champion";$("#result-note").textContent=data.note||"Estimate returned by the current model.";$("#api-status").textContent="API and model ready";
 }catch(e){error.textContent=e.message;error.hidden=false;}finally{btn.disabled=false;btn.innerHTML="Estimate likelihood <span>↗</span>";}
});
form.addEventListener("reset",()=>{setTimeout(()=>{$("#result-error").hidden=true;$("#result-content").hidden=true;$("#result-empty").hidden=false;},0);});
function renderStage(){
 const s=stages[currentStage];$("#workflow-title").textContent=s.title;$("#workflow-description").textContent=s.desc;$("#workflow-stage").textContent="STAGE "+String(currentStage+1).padStart(2,"0")+" / 09";$("#workflow-progress-fill").style.width=((currentStage+1)/9*100)+"%";
 $("#workflow-content").innerHTML=`<div class="eyebrow">STAGE ${String(currentStage+1).padStart(2,"0")}</div><h2>${esc(s.title)}</h2><p>${esc(s.desc)}</p><ul>${s.bullets.map(x=>`<li>${esc(x)}</li>`).join("")}</ul><button class="button button-secondary" id="stage-open-view">Open related workspace →</button>`;
 $("#stage-open-view").addEventListener("click",()=>showView(s.view,s.title));$("#workflow-prev").disabled=currentStage===0;$("#workflow-next").textContent=currentStage===8?"Back to overview →":"Next stage →";
}
$("#workflow-prev").addEventListener("click",()=>{if(currentStage>0){currentStage--;renderStage();}});
$("#workflow-next").addEventListener("click",()=>{if(currentStage<8){currentStage++;renderStage();}else showView("overview");});
const fileInput=$("#dataset-file");
fileInput.addEventListener("change",()=>{if(fileInput.files[0])uploadDataset(fileInput.files[0]);});
const drop=$("#upload-drop");
drop.addEventListener("dragover",e=>{e.preventDefault();drop.classList.add("dragover");});
drop.addEventListener("dragleave",()=>drop.classList.remove("dragover"));
drop.addEventListener("drop",e=>{e.preventDefault();drop.classList.remove("dragover");const f=e.dataTransfer.files[0];if(f)uploadDataset(f);});
async function uploadDataset(file){
 $("#upload-error").hidden=true;$("#profile-output").hidden=true;$("#upload-filename").textContent="Profiling "+file.name+"…";
 if(file.size>10*1024*1024){$("#upload-error").textContent="File exceeds the 10 MB upload limit.";$("#upload-error").hidden=false;$("#upload-filename").textContent=file.name;return;}
 const body=new FormData();body.append("file",file);
 try{const res=await fetch("/dataset/profile",{method:"POST",body});const data=await res.json();if(!res.ok)throw new Error(data.detail||"Could not profile dataset.");renderProfile(data);$("#upload-filename").textContent=file.name+" · profile ready";}
 catch(e){$("#upload-error").textContent=e.message;$("#upload-error").hidden=false;$("#upload-filename").textContent=file.name;}
}
function renderProfile(data){
 const cols=data.columns||[];const sample=data.sample||[];
 const summary=`<div class="profile-summary"><div class="profile-chip"><small>Rows</small><strong>${Number(data.rows).toLocaleString()}</strong></div><div class="profile-chip"><small>Columns</small><strong>${data.column_count}</strong></div><div class="profile-chip"><small>Missing cells</small><strong>${Number(data.missing_cells).toLocaleString()}</strong></div><div class="profile-chip"><small>Duplicate rows</small><strong>${Number(data.duplicate_rows).toLocaleString()}</strong></div><div class="profile-chip"><small>Numeric fields</small><strong>${data.numeric_columns}</strong></div><div class="profile-chip"><small>Categorical fields</small><strong>${data.categorical_columns}</strong></div></div>`;
 const table=`<div class="table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>${cols.map(c=>`<tr><td>${esc(c.name)}</td><td>${esc(c.dtype)}</td><td>${c.missing}</td><td>${c.unique}</td></tr>`).join("")}</tbody></table></div>`;
 const sampleTable=sample.length?`<div class="sample-table-wrap"><table class="sample-table"><thead><tr>${Object.keys(sample[0]).map(k=>`<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${sample.slice(0,8).map(row=>`<tr>${Object.values(row).map(v=>`<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`:"";
 $("#profile-output").innerHTML=`<div class="panel-header"><div><div class="eyebrow">UPLOAD PROFILE</div><h2>${esc(data.filename)}</h2></div><span class="pill pill-green">Profile generated</span></div>${summary}<h3>Column inventory</h3>${table}<h3>Sample records</h3>${sampleTable}<p class="muted-copy">This is a temporary exploratory profile. The uploaded file is not saved as the production dataset and does not retrain or replace the live model.</p>`;
 $("#profile-output").hidden=false;
}
renderStage();loadHealth();loadReport();