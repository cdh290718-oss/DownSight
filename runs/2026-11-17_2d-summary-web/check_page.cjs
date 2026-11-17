/* Execute real page interactions with a small DOM adapter; not a rendering test. */
const fs=require('fs');
const path=require('path');
const vm=require('vm');
const assert=require('assert/strict');
const run=__dirname;
const html=fs.readFileSync(path.join(run,'site/dist/index.html'),'utf8');
const source=fs.readFileSync(path.join(run,'site.js'),'utf8');
const dataText=html.match(/<script type="application\/json" id="research-data">([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(dataText);
const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
assert.equal(ids.length,new Set(ids).size,'Duplicate HTML IDs');
class Element {
  constructor(id){this.id=id;this.value='';this.innerHTML='';this.textContent='';this.href='';this.listeners={};this.attributes={};}
  addEventListener(type,fn){this.listeners[type]=fn;}
  setAttribute(key,value){this.attributes[key]=value;}
}
const nodes=Object.fromEntries(ids.map(id=>[id,new Element(id)]));
Object.assign(nodes.method,{value:'2d'});Object.assign(nodes.budget,{value:'20000'});
Object.assign(nodes.metric,{value:'absolute'});Object.assign(nodes.condition,{value:'all_full'});
Object.assign(nodes.sample,{value:'mean'});Object.assign(nodes.alpha,{value:'2'});
nodes['research-data'].textContent=dataText;
const document={getElementById(id){assert.ok(nodes[id],`Missing element ${id}`);return nodes[id];}};
const context=vm.createContext({document,console});vm.runInContext(source,context);
let states=0,detailCases=0;
for(const method of data.methods)for(const budget of data.budgets)for(const metric of ['absolute','pa']){
  nodes.method.value=method;nodes.budget.value=String(budget);nodes.metric.value=metric;
  context.renderAll();states++;
  assert.equal((nodes.matrix.innerHTML.match(/data-condition=/g)||[]).length,120);
  assert.equal((nodes['global-chart'].innerHTML.match(/class="bar-row/g)||[]).length,3);
  for(let alpha=0;alpha<3;alpha++){
    nodes.alpha.value=String(alpha);context.renderGlobal();
    const condition=['baseline','all_half','all_full'][alpha];
    assert.ok(nodes['input-stat'].textContent.includes(data.mean_input_2d_px[condition].toFixed(2)));
  }
  for(const condition of data.conditions){
    nodes.condition.value=condition;nodes.condition.listeners.change();
    const q=data.summaries.find(r=>r.method===method&&r.budget===budget&&r.metric===metric&&r.condition===condition);
    for(const pair of ['mean',...data.pair_order]){
      nodes.sample.value=pair;nodes.sample.listeners.change();detailCases++;
      assert.equal((nodes['joint-rows'].innerHTML.match(/<tr /g)||[]).length,15);
      const key=metric==='absolute'?'absolute_per_joint_mm':'pa_per_joint_mm';
      const expected=pair==='mean'?q.per_joint_corrected_mm:
        data.records.find(r=>r.method===method&&r.budget===budget&&r.condition===condition&&r.pair===pair)[key];
      for(const value of expected)assert.ok(nodes['joint-rows'].innerHTML.includes(value.toFixed(2)));
    }
  }
}
nodes.method.value='2d';nodes.budget.value='20000';nodes.metric.value='absolute';nodes.sample.value='mean';
nodes.matrix.listeners.click({target:{closest(){return {dataset:{condition:'hip',joint:'1'}};}}});
assert.equal(nodes.condition.value,'hip');assert.ok(nodes.inspector.innerHTML.includes('-17.00'));
nodes.condition.value='all_full';nodes.sample.value='body02';context.renderDetail();
assert.ok(nodes['joint-rows'].innerHTML.includes('40.61'));
assert.ok(nodes['joint-rows'].innerHTML.includes('61.25'));
const result={command:'Existing Node runtime: node check_page.cjs',exit_status:0,
  stateCombinations:states,detailSelections:detailCases,matrixCellsPerState:120,
  checks:['All method/budget/metric combinations execute','All sample/condition detail selections match evidence',
          'Matrix click updates condition and inspected joint','Input correction values are consistent',
          'Body02 wrist exception remains visible','Unique HTML IDs and existing referenced elements'],
  verificationLimit:'DOM-adapter execution and structural checks; no real-browser rendering or pointer-layout verification.'};
fs.writeFileSync(path.join(run,'evidence/page_checks.json'),JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result));
