/* Static evidence explorer. It never fits or executes a surrogate model. */
'use strict';
const $ = id => document.getElementById(id);
const NS = 'http://www.w3.org/2000/svg';
let data;
function node(tag, attrs={}, text='') {
  const el=document.createElementNS(NS,tag);
  for(const [key,value] of Object.entries(attrs)) el.setAttribute(key,String(value));
  if(text)el.textContent=text;
  return el;
}
function shape(cst,upper) {
  const binomial=[1,9,36,84,126,126,84,36,9,1], offset=upper?0:10;
  return Array.from({length:301},(_,j)=>{
    const x=(1-Math.cos(Math.PI*j/300))/2;
    let sum=0;
    for(let i=0;i<10;i++)sum+=cst[offset+i]*binomial[i]*x**i*(1-x)**(9-i);
    return [x,Math.sqrt(x)*(1-x)*sum];
  });
}
function drawAirfoil() {
  const row=data.cases[Number($('case').value)], svg=$('airfoil');svg.replaceChildren();
  const path=coeff=>[true,false].map(upper=>shape(coeff,upper).map(([x,y],i)=>`${i?'L':'M'}${70+860*x},${158-860*y}`).join(' '));
  svg.append(node('line',{x1:60,y1:158,x2:945,y2:158,stroke:'#ddd5e5','stroke-dasharray':'3 5'}));
  for(let t=0;t<=5;t++){const x=70+860*t/5;svg.append(node('text',{x,y:265,'text-anchor':'middle',fill:'#777080','font-size':15},(t/5).toFixed(1)));}
  svg.append(node('text',{x:500,y:292,'text-anchor':'middle',fill:'#777080','font-size':15},'x / c'));
  if($('baseline').checked)for(const d of path(data.baseline))svg.append(node('path',{d,fill:'none',stroke:'#a59aaf','stroke-width':1.6,'stroke-dasharray':'7 6'}));
  for(const d of path(row.cst))svg.append(node('path',{d,fill:'none',stroke:'#73508e','stroke-width':2.6}));
  svg.append(node('text',{x:72,y:35,fill:'#73508e','font-size':18},`LHS_2000 : ${row.id}`));
  const tbody=$('predictions');tbody.replaceChildren();
  ['CL','CD','CL / CD'].forEach((name,j)=>{
    const tr=document.createElement('tr');
    const values=[name,row.truth[j].toFixed(7),row.MLP[j].toFixed(7),`${(100*Math.abs((row.MLP[j]-row.truth[j])/row.truth[j])).toFixed(4)}%`,row.Kriging[j].toFixed(7),`${(100*Math.abs((row.Kriging[j]-row.truth[j])/row.truth[j])).toFixed(4)}%`];
    for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}tbody.append(tr);
  });
  $('coefficients').textContent=['U: '+row.cst.slice(0,10).map(x=>x.toPrecision(9)).join(', '),'L: '+row.cst.slice(10).map(x=>x.toPrecision(9)).join(', ')].join('\n');
}
function drawCurve(){
  const key=$('metric').value, metric=key==='E2'?'E2_mean_percent':`${key}_MAPE_percent_mean`,sd=key==='E2'?'E2_std_percent':`${key}_MAPE_percent_std`;
  const svg=$('curve');svg.replaceChildren();
  const values=data.learning.map(r=>r[metric]);
  const ymin=Math.floor(Math.log10(Math.min(...values)))-.15,ymax=Math.ceil(Math.log10(Math.max(...values)))+.1;
  const X=n=>88+850*(Math.log10(n)-Math.log10(50))/(Math.log10(10000)-Math.log10(50));
  const Y=v=>390-340*(Math.log10(v)-ymin)/(ymax-ymin);
  for(let p=Math.ceil(ymin);p<=Math.floor(ymax);p++){
    const y=Y(10**p);svg.append(node('line',{x1:88,y1:y,x2:938,y2:y,stroke:'#e8e2ed'}));
    svg.append(node('text',{x:73,y:y+5,'text-anchor':'end','font-size':15,fill:'#70697a'},`${10**p}%`));
  }
  const budgets=[...new Set(data.learning.map(r=>r.n_train))];
  for(const n of budgets){const x=X(n);svg.append(node('line',{x1:x,y1:50,x2:x,y2:390,stroke:'#f1edf4'}));svg.append(node('text',{x,y:421,'text-anchor':'middle','font-size':15,fill:'#70697a'},String(n)));}
  for(const [family,color] of [['MLP','#73508e'],['Kriging','#428b95']]){
    const rows=data.learning.filter(r=>r.family===family).sort((a,b)=>a.n_train-b.n_train);
    svg.append(node('path',{d:rows.map((r,i)=>`${i?'L':'M'}${X(r.n_train)},${Y(r[metric])}`).join(' '),fill:'none',stroke:color,'stroke-width':3}));
    for(const r of rows){const point=node('circle',{cx:X(r.n_train),cy:Y(r[metric]),r:6,fill:color,tabindex:0,role:'img','aria-label':`${family}, ${r.n_train} training cases: ${r[metric].toPrecision(4)} percent, standard deviation ${r[sd].toPrecision(3)}`});point.append(node('title',{},`${family} | n=${r.n_train}\nMean ${r[metric].toPrecision(5)}%\nSD ${r[sd].toPrecision(4)} percentage points`));svg.append(point);}
  }
  svg.append(node('text',{x:510,y:460,'text-anchor':'middle','font-size':17,fill:'#57505f'},'Training geometries n (+500 validation labels)'));
  svg.append(node('text',{x:88,y:26,'font-size':17,fill:'#57505f'},key==='E2'?'Normalized two-output error E₂ (%)':`${key.replace('_','/')} mean absolute percentage error (%)`));
  $('curve-note').textContent=key==='E2'?'The mean E₂ curves change order between 2,000 and 5,000 training samples. This is a measured interval, not an exact or universal threshold.':'Relative errors are shown for each physical quantity. These measured means use the same test cases and three recorded seeds at every training budget.';
}
async function main(){
  try{
    const response=await fetch('assets/benchmark.json');if(!response.ok)throw new Error(`HTTP ${response.status}`);data=await response.json();
    for(let i=0;i<data.cases.length;i++){const option=document.createElement('option');option.value=String(i);option.textContent=`LHS_2000:${data.cases[i].id}`;$('case').append(option);}
    $('case').disabled=false;$('previous').disabled=false;$('next').disabled=false;
    $('case').addEventListener('change',drawAirfoil);$('baseline').addEventListener('change',drawAirfoil);$('metric').addEventListener('change',drawCurve);
    for(const [id,step] of [['previous',-1],['next',1]])$(id).addEventListener('click',()=>{$('case').value=String((Number($('case').value)+step+data.cases.length)%data.cases.length);drawAirfoil();});
    drawAirfoil();drawCurve();$('status').textContent=`Loaded ${data.test_n.toLocaleString()} test geometries · frozen representative models · saved predictions`;
  }catch(error){$('status').textContent=`Could not load experiment data: ${error.message}. For local viewing, serve docs with python -m http.server 8000 --directory docs.`;}
}
main();
