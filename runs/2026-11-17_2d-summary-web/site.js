'use strict';
const DATA = JSON.parse(document.getElementById('research-data').textContent);
const $ = id => document.getElementById(id);
const getState = () => ({method:$('method').value,budget:Number($('budget').value),metric:$('metric').value});
const getSummary = (condition,state=getState()) => DATA.summaries.find(r => r.method===state.method && r.budget===state.budget && r.metric===state.metric && r.condition===condition);
const getRecord = (condition,pair,state=getState()) => DATA.records.find(r=>r.method===state.method && r.budget===state.budget && r.condition===condition && r.pair===pair);
const number = (value,digits=2) => Number(value).toFixed(digits);
const signed = value => (value>=0?'+':'')+number(value);
const direction = value => value>1?'good':value< -1?'bad':'neutral';
const textDirection = value => value>1?'positive':value< -1?'negative':'';
let inspectedJoint=0;
function renderGlobal(){
  const state=getState(),conditions=['baseline','all_half','all_full'];
  const full=getSummary('all_full'),half=getSummary('all_half');
  const values=[full.baseline_mean_mm,half.corrected_mean_mm,full.corrected_mean_mm];
  const alpha=Math.max(0,Math.min(2,Number($('alpha').value))),maximum=Math.max(...values,1);
  const method=state.method==='2d'?'直接二维':'固定混合',metric=state.metric==='absolute'?'不对齐三维误差':'PA-MPJPE';
  $('chart-title').textContent=`${method} · ${state.budget.toLocaleString()} 步 · ${metric}（mm）`;
  $('global-chart').innerHTML=conditions.map((c,i)=>`<div class="bar-row ${i===alpha?'active':''}"><span>${DATA.labels[c]}</span><div class="bar-track"><div class="bar-fill" style="width:${values[i]/maximum*100}%"></div></div><span class="bar-number">${number(values[i])}</span></div>`).join('');
  $('alpha').setAttribute('aria-valuetext',DATA.labels[conditions[alpha]]);
  $('input-stat').textContent=`输入二维平均参考距离：${number(DATA.mean_input_2d_px[conditions[alpha]])} px`;
  $('stage-label').textContent=DATA.labels[conditions[alpha]]+' · '+metric;
  $('stage-value').innerHTML=number(values[alpha])+'<span class="unit">mm</span>';
  if(alpha===0){$('stage-summary').textContent=`原预测条件。严格末段稳定性诊断通过 ${full.baseline_convergence_pass}/10；此诊断保留低置信度点。`;}
  else{
    const row=alpha===1?half:full;
    $('stage-summary').textContent=`${row.mean_joint_improved_count} 个关节均值改善超过 1 mm，${row.mean_joint_worsened_count} 个均值退化超过 1 mm；严格末段稳定性诊断通过 ${row.corrected_convergence_pass}/10。`;
  }
  $('csv-link').href=`https://github.com/cdh290718-oss/DownSight/blob/${DATA.experiment_commit}/runs/2026-11-17_2d-intervention/evidence/effects_${state.method}_${state.budget}.csv`;
}
function renderMatrix(){
  const state=getState(),condition=$('condition').value;
  const header='<thead><tr><th scope="col">修正的 2D 部位</th>'+DATA.joint_labels.map(n=>`<th scope="col">${n}</th>`).join('')+'</tr></thead>';
  const rows=DATA.conditions.map(c=>{
    const q=getSummary(c,state);
    return `<tr><th scope="row">${DATA.labels[c]}</th>`+q.per_joint_improvement_mm.map((v,j)=>`<td><button type="button" class="cell ${direction(v)} ${condition===c&&inspectedJoint===j?'chosen':''}" data-condition="${c}" data-joint="${j}" aria-label="${DATA.labels[c]}，${DATA.joint_labels[j]}，平均改善 ${number(v)} 毫米">${v>=0?'+':''}${number(v,1)}</button></td>`).join('')+'</tr>';
  }).join('');
  $('matrix').innerHTML='<caption class="sr-only">十段序列的逐关节平均改善量，单位毫米。正数改善，负数退化。</caption>'+header+'<tbody>'+rows+'</tbody>';
  renderInspector();
}
function renderInspector(){
  const q=getSummary($('condition').value),j=inspectedJoint;
  $('inspector').innerHTML=`<strong>${DATA.labels[q.condition]} → ${DATA.joint_labels[j]}</strong>　平均改善 ${signed(q.per_joint_improvement_mm[j])} mm。<br>95% 同时区间 [${number(q.per_joint_simultaneous_ci95_lower_mm[j])}, ${number(q.per_joint_simultaneous_ci95_upper_mm[j])}] mm；改善／退化样本数 ${q.per_joint_sample_improved_count[j]}/${q.per_joint_sample_worsened_count[j]}（其余变化在 ±1 mm 内）。`;
}
function renderDetail(){
  const state=getState(),c=$('condition').value,pair=$('sample').value,q=getSummary(c,state);
  let before,after;
  if(pair==='mean'){before=q.per_joint_baseline_mm;after=q.per_joint_corrected_mm;}
  else{
    const key=state.metric==='absolute'?'absolute_per_joint_mm':'pa_per_joint_mm';
    before=getRecord('baseline',pair,state)[key];after=getRecord(c,pair,state)[key];
  }
  $('count-heading').textContent=pair==='mean'?'改善／退化样本数':'本样本变化';
  $('joint-rows').innerHTML=DATA.joint_labels.map((name,j)=>{
    const d=before[j]-after[j],note=pair==='mean'?`${q.per_joint_sample_improved_count[j]}/${q.per_joint_sample_worsened_count[j]}`:d>1?'改善':d< -1?'退化':'变化小';
    return `<tr class="${j===inspectedJoint?'picked':''}"><th scope="row">${name}</th><td>${number(before[j])}</td><td>${number(after[j])}</td><td class="${textDirection(d)}">${signed(d)}</td><td>${note}</td></tr>`;
  }).join('');
}
function renderAll(){renderGlobal();renderMatrix();renderDetail();}
['method','budget','metric'].forEach(id=>$(id).addEventListener('change',renderAll));
$('alpha').addEventListener('input',renderGlobal);
$('condition').addEventListener('change',()=>{renderMatrix();renderDetail();});
$('sample').addEventListener('change',renderDetail);
$('matrix').addEventListener('click',event=>{
  const button=event.target.closest('button[data-condition]');
  if(!button)return;
  $('condition').value=button.dataset.condition;inspectedJoint=Number(button.dataset.joint);
  renderMatrix();renderDetail();
});
renderAll();
