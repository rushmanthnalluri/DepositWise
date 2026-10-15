const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const views = ["overview","prediction","dataset","models","evaluation","monitoring","workflow"];
const stages = [
  {title:"Upload and validate dataset",desc:"Start with the official Bank Marketing dataset or inspect a CSV/XLSX of your own.",bullets:["Check accepted file format and schema","Inspect dimensions and sample rows","Keep exploratory uploads separate from the deployed model"],view:"dataset"},
  {title:"Feature inventory",desc:"Understand which columns are available and how each field should be interpreted.",bullets:["Inspect column names and inferred data types","Identify target and post-outcome fields","Exclude duration from pre-call predictions"],view:"dataset"},
  {title:"Descriptive statistics",desc:"Summarize distributions before modelling.",bullets:["Review numeric ranges and central tendency","Compare customer and campaign characteristics","Check unusual values and high-cardinality categories"],view:"dataset"},
  {title:"Missing-value analysis",desc:"Measure data completeness and choose defensible handling policies.",bullets:["Count missing values by column","Inspect duplicate records","Avoid fitting imputation statistics on the test set"],view:"dataset"},
  {title:"Visualize the data",desc:"Explore target balance and feature relationships.",bullets:["Inspect class imbalance","Review feature importance from the trained model","Use the data profile to guide further analysis"],view:"evaluation"},
  {title:"Preprocess safely",desc:"Prepare features without leaking information from held-out data.",bullets:["Choose numeric mean or median imputation","Choose categorical most-frequent or constant imputation","Optionally scale numeric features inside the training pipeline"],view:"dataset"},
  {title:"Train and compare models",desc:"Train temporary candidates on an uploaded binary-classification dataset and compare them safely.",bullets:["Compare Logistic Regression, Random Forest, and Extra Trees","Select the champion using validation average precision","Keep sandbox artifacts separate from the production model"],view:"dataset"},
  {title:"Evaluate models",desc:"Read held-out metrics and probability-quality diagnostics.",bullets:["Review precision, recall, F1, ROC-AUC and average precision","Compare the tuned threshold with 0.5","Inspect calibration and feature importance"],view:"evaluation"},
  {title:"Score a customer",desc:"Submit a profile to the deployed inference API.",bullets:["Enter pre-call customer and campaign attributes","Get probability and threshold-based classification","Treat the output as decision support, not a guarantee"],view:"prediction"}
];
let report = null;
let activeDatasetId = null;
let activeSandboxRunId = null;
let activeSampleRow = null;
let activeTargetColumn = null;
let lastProbability = null;
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
 $("#kpi-rate").textContent=(data.positive_rate_full_dataset??data.positive_rate)!=null?pct(data.positive_rate_full_dataset??data.positive_rate):"11.7%";
 $("#kpi-auc").textContent=fmt(metrics.roc_auc,3);
 $("#kpi-f1").textContent=fmt(metrics.f1,3);
 $("#champion-name").textContent=data.selected_model||"Not available";
 $("#selection-metric").textContent=data.selection_metric||"Average precision";
 const threshold=Number(data.threshold_selection?.threshold??data.threshold_selection?.selected_threshold);
  $("#decision-threshold").textContent=Number.isFinite(threshold)?pct(threshold,0):"—";
 if(Number.isFinite(threshold)){ $("#threshold-control").value=String(Math.max(10,Math.min(90,Math.round(threshold*100)))); $("#threshold-control-label").textContent=pct(Number($("#threshold-control").value)/100,0); }
 $("#overview-metrics").innerHTML=metricBars([["Average precision",metrics.average_precision],["ROC-AUC",metrics.roc_auc],["Precision",metrics.precision],["Recall",metrics.recall],["F1 score",metrics.f1]]);
 const features=data.top_encoded_feature_importance||[];
 $("#feature-list").innerHTML=featureMarkup(features);
 $("#evaluation-features").innerHTML=featureMarkup(features);
 $("#eval-auc").textContent=fmt(metrics.roc_auc,3);
 $("#eval-precision").textContent=fmt(metrics.precision,3);
 $("#eval-recall").textContent=fmt(metrics.recall,3);
 $("#eval-brier").textContent=fmt(metrics.brier_score,3);
 const curve=data.test_calibration_curve||[];
 const plotW=320,plotH=170,left=45,top=15,bottom=top+plotH,right=left+plotW;
 const points=curve.map(p=>({x:Number(p.mean_predicted_probability),y:Number(p.observed_positive_fraction)})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
 const coords=points.map(p=>`${left+Math.max(0,Math.min(1,p.x))*plotW},${bottom-Math.max(0,Math.min(1,p.y))*plotH}`).join(" ");
 const circles=points.map(p=>`<circle cx="${left+Math.max(0,Math.min(1,p.x))*plotW}" cy="${bottom-Math.max(0,Math.min(1,p.y))*plotH}" r="4" fill="#d9a63f" stroke="#171a16" stroke-width="1.5"><title>Predicted ${fmt(p.x,2)}, observed ${fmt(p.y,2)}</title></circle>`).join("");
 const grid=[0,0.25,0.5,0.75,1].map(v=>`<g><line x1="${left}" y1="${bottom-v*plotH}" x2="${right}" y2="${bottom-v*plotH}" stroke="#303630" stroke-dasharray="3 4"/><line x1="${left+v*plotW}" y1="${top}" x2="${left+v*plotW}" y2="${bottom}" stroke="#303630" stroke-dasharray="3 4"/><text x="${left-9}" y="${bottom-v*plotH+3}" fill="#8d978b" text-anchor="end" font-size="9">${v.toFixed(2)}</text><text x="${left+v*plotW}" y="${bottom+15}" fill="#8d978b" text-anchor="middle" font-size="9">${v.toFixed(2)}</text></g>`).join("");
 $("#calibration-chart").innerHTML=points.length?`<svg viewBox="0 0 410 235" role="img" aria-label="Calibration curve comparing predicted probabilities with observed positive fractions" style="width:100%;height:auto"><rect x="0" y="0" width="410" height="235" fill="none"/>${grid}<line x1="${left}" y1="${bottom}" x2="${right}" y2="${top}" stroke="#777f74" stroke-width="1.5" stroke-dasharray="5 4"/><polyline points="${coords}" fill="none" stroke="#d9a63f" stroke-width="2"/>${circles}<text x="${left+plotW/2}" y="226" fill="#aab3a7" text-anchor="middle" font-size="10">Mean predicted probability</text><text x="12" y="${top+plotH/2}" fill="#aab3a7" text-anchor="middle" font-size="10" transform="rotate(-90 12 ${top+plotH/2})">Observed positive fraction</text></svg>`:'<div class="empty-state">Calibration curve is not present in the current report.</div>';
 const cm=metrics.confusion_matrix||{};
 const matrixRows=[["True negative","true_negative"],["False positive","false_positive"],["False negative","false_negative"],["True positive","true_positive"]];
 $("#confusion-matrix").innerHTML=matrixRows.map(([label,key])=>`<div class="detail-row"><span>${label}</span><strong>${Number.isFinite(Number(cm[key]))?Number(cm[key]).toLocaleString():"—"}</strong></div>`).join("");

 $("#threshold-display").textContent=Number.isFinite(threshold)?pct(threshold,0):"—";
 $("#threshold-copy").textContent=data.threshold_selection?.method||"Operating threshold chosen on validation data; test data is reserved for final reporting.";
 $("#baseline-f1").textContent=fmt(base.f1,3);
 $("#tuned-f1").textContent=fmt(metrics.f1,3);
 $("#tuned-recall").textContent=pct(metrics.recall,1);
 $("#tuned-precision").textContent=pct(metrics.precision,1);
 $("#data-rows").textContent=Number.isFinite(rows)?rows.toLocaleString():"—";
 $("#data-rate").textContent=(data.positive_rate_full_dataset??data.positive_rate)!=null?pct(data.positive_rate_full_dataset??data.positive_rate):"11.7%";
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
  try{
   const metricsResponse=await fetch("/metrics",{cache:"no-store"});
   if(metricsResponse.ok){
    const metrics=await metricsResponse.json();
    $("#health-prediction-count").textContent=Number(metrics.prediction_count||0).toLocaleString();
    $("#health-checks").innerHTML += [["Recent prediction samples",String(metrics.recent_sample_count)],["Mean inference latency",metrics.mean_latency_ms==null?"No samples":fmt(metrics.mean_latency_ms,2)+" ms"],["Last successful prediction",metrics.last_prediction_at||"No predictions yet"],["Feature drift monitoring",metrics.drift_monitoring==="not_configured"?"Not configured":"Numeric PSI"],["Numeric PSI status",metrics.drift_status||"Insufficient data"],["Features above warning threshold",(metrics.drift_features||[]).filter(x=>x.psi>=0.1).map(x=>x.feature+" ("+fmt(x.psi,2)+")").join(", ")||"None"]].map(([a,b])=>'<div class="check-row"><span>'+esc(a)+'</span><span class="check-state '+(b==="Not configured"?"warn":"")+'">'+esc(b)+'</span></div>').join("");
   }
  }catch(_){}

 }catch(e){$("#api-status").textContent="API connection unavailable";$("#sidebar-health").textContent="API unavailable";$("#sidebar-status").classList.remove("online");$("#health-api").textContent="Unavailable";}
}

async function loadExperiments(){
 try{
  const response=await fetch("/experiments",{cache:"no-store"});if(!response.ok)throw new Error("Run history unavailable");
  const data=await response.json();
  $("#experiment-table").innerHTML=(data.runs||[]).slice().reverse().map(run=>'<tr><td>'+esc(new Date(run.created_at).toLocaleString())+'</td><td>'+esc(run.target)+'</td><td>'+esc(run.selected_model)+'</td><td>'+Number(run.rows).toLocaleString()+'</td><td>'+fmt(run.validation_average_precision,3)+'</td><td>'+fmt(run.test_f1,3)+'</td></tr>').join("")||'<tr><td colspan="6">No sandbox runs in this process yet.</td></tr>';
 }catch(_){$("#experiment-table").innerHTML='<tr><td colspan="6">Experiment history unavailable.</td></tr>';}
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
 try{const res=await fetch("/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({customer:values,threshold:Number($("#threshold-control").value)/100})});const data=await res.json();if(!res.ok)throw new Error(data.detail||"Prediction failed");
 const probability=Math.max(0,Math.min(1,Number(data.subscription_probability)));const percentage=(probability*100).toFixed(1);lastProbability=probability;updateEconomics();$("#result-empty").hidden=true;$("#result-content").hidden=false;$("#probability-value").textContent=percentage;$("#meter-fill").style.width=percentage+"%";
 $("#outcome").textContent=data.predicted_subscription?"Above selected threshold":"Below selected threshold";$("#outcome").classList.toggle("negative",!data.predicted_subscription);$("#threshold-value").textContent=pct(data.threshold,0);$("#prediction-model").textContent=data.selected_model||"Current champion";$("#result-note").textContent=data.note||"Estimate returned by the current model.";
 const impacts=data.top_feature_impacts||[];
 $("#prediction-explanations").innerHTML=impacts.length?'<div class="explanation-heading">Top feature signals <span>baseline sensitivity</span></div><div class="feature-list">'+impacts.map(item=>'<div class="explanation-row"><span class="feature-name">'+esc(item.feature)+'</span><span class="explanation-impact '+(Number(item.impact_probability)>=0?"positive":"negative")+'">'+(Number(item.impact_probability)>=0?"+":"")+fmt(Number(item.impact_probability)*100,2)+' pp</span></div>').join("")+'</div><p class="muted-copy">These single-feature changes compare this input with a typical training profile. They are not causal effects or SHAP values.</p>':'<p class="muted-copy">Per-prediction explanations are unavailable for this model artifact.</p>';
 $("#api-status").textContent="API and model ready";
 }catch(e){error.textContent=e.message;error.hidden=false;}finally{btn.disabled=false;btn.innerHTML="Estimate subscription likelihood <span>↗</span>";}
});

$("#threshold-control").addEventListener("input",()=>{
 const threshold=Number($("#threshold-control").value)/100;
 $("#threshold-control-label").textContent=pct(threshold,0);
 $("#threshold-value").textContent=pct(threshold,0);
 if(lastProbability!==null){
  const above=lastProbability>=threshold;
  $("#outcome").textContent=above?"Above selected threshold":"Below selected threshold";
  $("#outcome").classList.toggle("negative",!above);
 }
});
form.addEventListener("reset",()=>{lastProbability=null;setTimeout(()=>{$("#result-error").hidden=true;$("#result-content").hidden=true;$("#result-empty").hidden=false;},0);});

function updateEconomics(){
 if(lastProbability===null)return;
 const value=Math.max(0,Number($("#subscription-value").value)||0);
 const cost=Math.max(0,Number($("#contact-cost").value)||0);
 const expected=lastProbability*value-cost;
 const net=$("#expected-net");net.textContent=(expected>=0?"+":"−")+"₹"+Math.abs(expected).toFixed(2);net.classList.toggle("positive",expected>=0);net.classList.toggle("negative",expected<0);
 $("#break-even-probability").textContent=value>0?pct(cost/value,1):"Not defined when value is ₹0";
}
$("#subscription-value").addEventListener("input",updateEconomics);
$("#contact-cost").addEventListener("input",updateEconomics);

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


async function scoreUploadedBatch(){
 const button=$("#batch-score-button");const error=$("#upload-error");error.hidden=true;button.disabled=true;button.textContent="Scoring rows…";
 try{
  const res=await fetch("/workflow/"+encodeURIComponent(activeDatasetId)+"/score-batch",{method:"POST"});
  if(!res.ok){const data=await res.json();throw new Error(data.detail||"Batch scoring failed.");}
  const blob=await res.blob();const url=URL.createObjectURL(blob);const link=document.createElement("a");link.href=url;link.download="depositwise_scored.csv";document.body.appendChild(link);link.click();link.remove();URL.revokeObjectURL(url);
  button.textContent="CSV downloaded";
 }catch(e){error.textContent=e.message;error.hidden=false;button.textContent="Score full dataset ↓";}
 finally{button.disabled=false;if(button.textContent==="Scoring rows…")button.textContent="Score full dataset ↓";}
}

async function trainSandboxDataset(){
 if(!activeDatasetId||!activeTargetColumn)return;
 const output=$("#sandbox-training-output");const button=$("#sandbox-train-button");button.disabled=true;button.textContent="Training sandbox models…";output.innerHTML='<div class="empty-state">Training isolated candidate models. This can take a little while.</div>';
 try{
  const res=await fetch("/workflow/"+encodeURIComponent(activeDatasetId)+"/train",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({target:activeTargetColumn,models:$$('input[name="sandbox-model"]:checked').map(input=>input.value),row_limit:20000,numeric_imputation:$("#sandbox-numeric-imputation").value,categorical_imputation:$("#sandbox-categorical-imputation").value,scale_numeric:$("#sandbox-scale-numeric").value==="true"})});
  const data=await res.json();if(!res.ok)throw new Error(data.detail||"Sandbox training failed.");
  activeSandboxRunId=data.run_id;loadExperiments();
  const rows=(data.validation_model_comparison||[]).map(m=>"<tr><td>"+esc(m.model)+"</td><td>"+fmt(m.average_precision,3)+"</td><td>"+fmt(m.roc_auc,3)+"</td><td>"+fmt(m.f1,3)+"</td><td class=\"table-status\">"+(m.model===data.selected_model?"CHAMPION":"CANDIDATE")+"</td></tr>").join("");
  const example={...(activeSampleRow||{})};delete example[activeTargetColumn];
  output.innerHTML='<div class="panel" style="margin-top:16px"><div class="panel-header"><div><div class="eyebrow">SANDBOX RUN · NOT PRODUCTION</div><h2>Training results</h2></div><span class="pill pill-green">Completed</span></div><div class="profile-summary"><div class="profile-chip"><small>Champion</small><strong>'+esc(data.selected_model)+'</strong></div><div class="profile-chip"><small>Test ROC-AUC</small><strong>'+fmt(data.final_test_metrics.roc_auc,3)+'</strong></div><div class="profile-chip"><small>Test F1</small><strong>'+fmt(data.final_test_metrics.f1,3)+'</strong></div><div class="profile-chip"><small>Test precision</small><strong>'+pct(data.final_test_metrics.precision,1)+'</strong></div><div class="profile-chip"><small>Test recall</small><strong>'+pct(data.final_test_metrics.recall,1)+'</strong></div><div class="profile-chip"><small>Threshold</small><strong>'+pct(data.threshold_selection.threshold,0)+'</strong></div></div><div class="table-wrap"><table><thead><tr><th>Candidate</th><th>Validation AP</th><th>Validation ROC-AUC</th><th>Validation F1</th><th>Selection</th></tr></thead><tbody>'+rows+'</tbody></table></div><p class="muted-copy">This run is held in memory and does not change the production model. Test metrics are shown after validation-based model and threshold selection.</p><div class="divider"></div><h3>Sandbox prediction input (JSON)</h3><p class="muted-copy">An example feature row is prefilled below. Edit it to represent a new customer profile, and remove any fields not available at decision time.</p><textarea id="sandbox-customer-json" rows="7" style="width:100%;background:#101310;color:#e9eee5;border:1px solid #343a34;border-radius:7px;padding:12px;font:11px \'DM Mono\',monospace">'+esc(JSON.stringify(example,null,2))+'</textarea><button class="button button-primary" id="sandbox-score-button" style="margin-top:12px">Score sandbox profile ↗</button><div id="sandbox-prediction-output"></div></div>';
  $("#sandbox-score-button").addEventListener("click",scoreSandboxCustomer);
 }catch(error){output.innerHTML='<div class="error-message">'+esc(error.message)+'</div>';}
 finally{button.disabled=false;button.textContent="Train sandbox models ↗";}
}
async function scoreSandboxCustomer(){
 const output=$("#sandbox-prediction-output");
 try{
  const customer=JSON.parse($("#sandbox-customer-json").value);
  if(!customer||Array.isArray(customer)||typeof customer!=="object")throw new Error("Enter a JSON object of customer features.");
  const res=await fetch("/workflow/"+encodeURIComponent(activeDatasetId)+"/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({run_id:activeSandboxRunId,customer})});
  const data=await res.json();if(!res.ok)throw new Error(data.detail||"Sandbox prediction failed.");
  output.innerHTML='<div class="notice" style="margin-top:14px"><span class="notice-mark">i</span><span>Sandbox estimate: <strong>'+pct(data.subscription_probability,1)+'</strong> positive-class probability · threshold '+pct(data.threshold,0)+' · '+(data.predicted_positive?"above threshold":"below threshold")+'. This is not a production prediction.</span></div>';
 }catch(error){output.innerHTML='<div class="error-message">'+esc(error.message)+'</div>';}
}

function renderProfile(data){
 activeDatasetId=data.dataset_id||null;activeSandboxRunId=null;activeSampleRow=(data.sample||[])[0]||null;activeTargetColumn=data.target_summary?.column||null;
 const cols=data.columns||[];const sample=data.sample||[];
 const summary=`<div class="profile-summary"><div class="profile-chip"><small>Rows</small><strong>${Number(data.rows).toLocaleString()}</strong></div><div class="profile-chip"><small>Columns</small><strong>${data.column_count}</strong></div><div class="profile-chip"><small>Missing cells</small><strong>${Number(data.missing_cells).toLocaleString()}</strong></div><div class="profile-chip"><small>Duplicate rows</small><strong>${Number(data.duplicate_rows).toLocaleString()}</strong></div><div class="profile-chip"><small>Numeric fields</small><strong>${data.numeric_columns}</strong></div><div class="profile-chip"><small>Categorical fields</small><strong>${data.categorical_columns}</strong></div></div>`;
 const table=`<div class="table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr></thead><tbody>${cols.map(c=>`<tr><td>${esc(c.name)}</td><td>${esc(c.dtype)}</td><td>${c.missing}</td><td>${c.unique}</td></tr>`).join("")}</tbody></table></div>`;
 const targetSummary=data.target_summary?'<div class="panel" style="margin:14px 0"><div class="eyebrow">TARGET DISTRIBUTION · '+esc(data.target_summary.column)+'</div><div class="feature-list" style="margin-top:12px">'+Object.entries(data.target_summary.counts).map(([label,count])=>'<div class="detail-row"><span>'+esc(label)+'</span><strong>'+Number(count).toLocaleString()+'</strong></div>').join("")+'</div></div>':"";
 const sampleTable=sample.length?`<div class="sample-table-wrap"><table class="sample-table"><thead><tr>${Object.keys(sample[0]).map(k=>`<th>${esc(k)}</th>`).join("")}</tr></thead><tbody>${sample.slice(0,8).map(row=>`<tr>${Object.values(row).map(v=>`<td>${esc(v)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`:"";
 $("#profile-output").innerHTML=`<div class="panel-header"><div><div class="eyebrow">UPLOAD PROFILE</div><h2>${esc(data.filename)}</h2></div><span class="pill pill-green">Profile generated</span></div>${summary}${targetSummary}<h3>Column inventory</h3>${table}<h3>Sample records</h3>${sampleTable}<div class="profile-actions"><button class="button button-secondary" data-profile-action="features">Feature inventory</button><button class="button button-secondary" data-profile-action="statistics">Descriptive stats</button><button class="button button-secondary" data-profile-action="missing">Missing values</button><button class="button button-secondary" data-profile-action="visualization">Correlations & target</button><button class="button button-primary" id="sandbox-train-button" type="button">Train sandbox models ↗</button></div><div class="preprocess-options"><label>Numeric imputation<select id="sandbox-numeric-imputation"><option value="median">Median</option><option value="mean">Mean</option></select></label><label>Categorical imputation<select id="sandbox-categorical-imputation"><option value="most_frequent">Most frequent</option><option value="constant">Fill with “missing”</option></select></label><label>Scale numeric features<select id="sandbox-scale-numeric"><option value="true">Yes</option><option value="false">No</option></select></label></div><div id="profile-insights"></div><div id="sandbox-training-output"></div><p class="muted-copy">This is a temporary exploratory profile. The uploaded file is not saved as the production dataset and does not retrain or replace the live model.</p>`;
 $("#profile-output").hidden=false;
 $("[data-profile-action]").forEach(button=>button.addEventListener("click",()=>loadProfileInsight(button.dataset.profileAction)));
}
async function loadProfileInsight(action){
 if(!activeDatasetId){return;}
 const output=$("#profile-insights");output.innerHTML='<div class="empty-state">Loading analysis…</div>';
 try{
  const res=await fetch(`/workflow/${encodeURIComponent(activeDatasetId)}/${action}`);
  const data=await res.json();if(!res.ok)throw new Error(data.detail||"Analysis unavailable");
  let title=action==="features"?"Feature inventory":action==="statistics"?"Descriptive statistics":action==="missing"?"Missing-value analysis":"Correlations and target distribution";
  let body="";
  if(action==="features")body='<div class="table-wrap"><table><thead><tr><th>Column</th><th>Type</th><th>Role</th><th>Missing</th><th>Unique</th></tr></thead><tbody>'+data.columns.map(x=>'<tr><td>'+esc(x.name)+'</td><td>'+esc(x.dtype)+'</td><td>'+esc(x.role)+'</td><td>'+x.missing+'</td><td>'+x.unique+'</td></tr>').join("")+'</tbody></table></div>';
  if(action==="statistics")body=data.statistics.length?'<div class="table-wrap"><table><thead><tr>'+Object.keys(data.statistics[0]).map(k=>'<th>'+esc(k)+'</th>').join("")+'</tr></thead><tbody>'+data.statistics.map(row=>'<tr>'+Object.values(row).map(v=>'<td>'+esc(v==null?"—":v)+'</td>').join("")+'</tr>').join("")+'</tbody></table></div>':'<p class="muted-copy">No numeric columns were detected.</p>';
  if(action==="missing")body='<p class="muted-copy">'+Number(data.missing_cells).toLocaleString()+' missing cells · '+Number(data.duplicate_rows).toLocaleString()+' duplicate rows</p><div class="table-wrap"><table><thead><tr><th>Column</th><th>Missing</th><th>Missing %</th></tr></thead><tbody>'+data.columns.map(x=>'<tr><td>'+esc(x.column)+'</td><td>'+x.missing+'</td><td>'+pct(x.missing_fraction,1)+'</td></tr>').join("")+'</tbody></table></div>';
  if(action==="visualization"){const t=data.target_distribution;body=(t?'<h3>Target distribution · '+esc(t.column)+'</h3><div class="feature-list">'+t.labels.map((label,i)=>'<div class="detail-row"><span>'+esc(label)+'</span><strong>'+Number(t.counts[i]).toLocaleString()+'</strong></div>').join("")+'</div>':'<p class="muted-copy">No conventional target column (y/target/subscribed/subscription) was detected.</p>')+'<h3>Numeric distributions</h3>'+(data.numeric_distributions&&data.numeric_distributions.length?'<div class="eda-chart-grid">'+data.numeric_distributions.map(d=>{const max=Math.max(1,...d.counts);return '<article class="eda-chart"><h4>'+esc(d.feature)+'</h4>'+d.labels.map((label,i)=>'<div class="eda-bar-row"><span title="'+esc(label)+'">'+esc(label)+'</span><div class="eda-bar-track"><i style="width:'+((d.counts[i]/max)*100)+'%"></i></div><b>'+Number(d.counts[i]).toLocaleString()+'</b></div>').join("")+'</article>';}).join("")+'</div>':'<p class="muted-copy">No numeric distributions available.</p>')+'<h3>Categorical frequencies</h3>'+(data.categorical_distributions&&data.categorical_distributions.length?'<div class="eda-chart-grid">'+data.categorical_distributions.map(d=>{const max=Math.max(1,...d.counts);return '<article class="eda-chart"><h4>'+esc(d.feature)+'</h4>'+d.labels.map((label,i)=>'<div class="eda-bar-row"><span title="'+esc(label)+'">'+esc(label)+'</span><div class="eda-bar-track"><i style="width:'+((d.counts[i]/max)*100)+'%"></i></div><b>'+Number(d.counts[i]).toLocaleString()+'</b></div>').join("")+'</article>';}).join("")+'</div>':'<p class="muted-copy">No categorical distributions available.</p>')+'<h3>Subscription rate by segment</h3>'+(data.categorical_target_rates&&data.categorical_target_rates.length?'<div class="table-wrap"><table><thead><tr><th>Feature</th><th>Segment</th><th>Count</th><th>Subscription rate</th></tr></thead><tbody>'+data.categorical_target_rates.map(x=>'<tr><td>'+esc(x.feature)+'</td><td>'+esc(x.group)+'</td><td>'+Number(x.count).toLocaleString()+'</td><td>'+pct(x.positive_rate,1)+'</td></tr>').join("")+'</tbody></table></div>':'<p class="muted-copy">No segment groups with enough labelled examples were detected.</p>')+'<h3>Strongest numeric correlations</h3>'+(data.top_numeric_correlations.length?'<div class="table-wrap"><table><thead><tr><th>Feature A</th><th>Feature B</th><th>Correlation</th></tr></thead><tbody>'+data.top_numeric_correlations.slice(0,15).map(x=>'<tr><td>'+esc(x.feature_a)+'</td><td>'+esc(x.feature_b)+'</td><td>'+fmt(x.correlation,3)+'</td></tr>').join("")+'</tbody></table></div>':'<p class="muted-copy">At least two numeric columns are needed for correlations.</p>');}
  output.innerHTML='<div class="panel" style="margin-top:16px"><div class="panel-header"><div><div class="eyebrow">DATASET ANALYSIS</div><h2>'+title+'</h2></div><span class="pill pill-green">Computed live</span></div>'+body+'</div>';
 }catch(error){output.innerHTML='<div class="error-message">'+esc(error.message)+'</div>';}
}
renderStage();loadHealth();loadReport();loadExperiments();