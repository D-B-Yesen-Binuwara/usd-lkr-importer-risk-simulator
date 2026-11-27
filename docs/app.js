/* Offline GBM demo. Explicit-shock parity uses the same exact transition as src/gbm.py. */
const REF = typeof window !== "undefined" ? window.IMPORTER_REFERENCE : undefined;
function exactStep(rate, alpha, sigma, days, shock) {
  return rate * Math.exp(alpha * days / 365 + sigma * Math.sqrt(days / 365) * shock);
}
function parityResult(ref = REF) {
  let rate = ref.rate;
  ref.dayIntervals.forEach((days, i) => { rate = exactStep(rate, ref.alpha, ref.sigma, days, ref.shocks[i]); });
  const payment = rate * ref.invoiceUsd;
  const tolerance = ref.relativeTolerance;
  return {rate, payment, passed: Math.abs(rate-ref.terminalRate) <= tolerance*Math.max(1,Math.abs(ref.terminalRate)) && Math.abs(payment-ref.paymentLkr) <= tolerance*Math.max(1,Math.abs(ref.paymentLkr))};
}
function rng(seed) { let x=seed>>>0; return () => { x=(x+0x6D2B79F5)>>>0; let t=x; t=Math.imul(t^(t>>>15),t|1); t^=t+Math.imul(t^(t>>>7),t|61); return ((t^(t>>>14))>>>0)/4294967296; }; }
function normal(rand) { const u=Math.max(rand(),Number.MIN_VALUE),v=rand(); return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v); }
function upperTail(values, level=.95) {
  const a=[...values].sort((x,y)=>x-y), n=a.length, k=(1-level)*n, whole=Math.floor(k), fraction=k-whole;
  const var95=a[Math.ceil(level*n)-1]; let sum=0;
  for(let i=0;i<whole;i++)sum+=a[n-1-i];
  if(fraction)sum+=fraction*a[n-whole-1];
  return {var95,es95:sum/k};
}
function histogramData(values,bins=24) {
  if(!Number.isInteger(bins)||bins<1)throw new RangeError("bins must be a positive integer");
  const counts=Array(bins).fill(0);
  if(!values.length)return {min:0,max:1,dataMin:null,dataMax:null,counts,belowRange:0,aboveRange:0};
  let dataMin=Infinity,dataMax=-Infinity;
  for(const x of values){
    if(!Number.isFinite(x))throw new RangeError("histogram values must be finite");
    dataMin=Math.min(dataMin,x);dataMax=Math.max(dataMax,x);
  }
  // Use the complete observed support. Only a constant sample needs visual padding.
  const padding=Math.max(Math.abs(dataMin)*.005,.5);
  const min=dataMin===dataMax?dataMin-padding:dataMin;
  const max=dataMin===dataMax?dataMax+padding:dataMax;
  let belowRange=0,aboveRange=0;
  for(const x of values){
    if(x<min){belowRange++;continue;}
    if(x>max){aboveRange++;continue;}
    const index=Math.min(bins-1,Math.floor((x-min)/(max-min)*bins));
    counts[index]++;
  }
  return {min,max,dataMin,dataMax,counts,belowRange,aboveRange};
}
function niceStep(range,target=4) {
  const raw=range/target, power=10**Math.floor(Math.log10(raw)), fraction=raw/power;
  return (fraction<=1?1:fraction<=2?2:fraction<=5?5:10)*power;
}
function metrics(rates,invoice,budget,referenceRate=REF.rate) {
  const costs=rates.map(x=>x*invoice), extra=costs.map(x=>x-invoice*referenceRate), n=costs.length;
  const mean=costs.reduce((a,b)=>a+b,0)/n, variance=costs.reduce((a,b)=>a+(b-mean)**2,0)/(n-1);
  return {mean,ciLow:mean-1.959963984540054*Math.sqrt(variance/n),ciHigh:mean+1.959963984540054*Math.sqrt(variance/n),budgetProbability:costs.filter(x=>x>budget).length/n,...upperTail(extra)};
}
function parseSettings(raw) {
  const fields=["invoice","days","volatility","paths","seed"];
  if(fields.some(key=>typeof raw[key]!=="string"||raw[key].trim()===""))return null;
  const invoice=Number(raw.invoice),days=Number(raw.days),multiplier=Number(raw.volatility),target=Number(raw.paths),seed=Number(raw.seed);
  if(!Number.isInteger(invoice)||invoice<1||invoice>10000000||
     !Number.isInteger(days)||days<1||days>365||
     !Number.isFinite(multiplier)||multiplier<0||multiplier>3||
     !Number.isInteger(target)||target<500||target>20000||
     !Number.isInteger(seed)||seed<0||seed>4294967295)return null;
  return {invoice,days,multiplier,target,seed};
}
if(typeof module!=="undefined")module.exports={exactStep,parityResult,rng,normal,upperTail,histogramData,niceStep,metrics,parseSettings};
if(typeof document!=="undefined")document.addEventListener("DOMContentLoaded",()=>{
  const $=id=>document.getElementById(id), fmt=x=>new Intl.NumberFormat("en-US",{maximumFractionDigits:0}).format(x);
  const form=$("settings"), start=$("start"), pause=$("pause"), reset=$("reset"), canvas=$("chart"), ctx=canvas.getContext("2d");
  let state=null,timer=null;
  const parity=parityResult();$("parity").textContent=parity.passed?`PASS: Python parity rate ${parity.rate.toFixed(8)} LKR/USD; payment LKR ${parity.payment.toFixed(2)} (relative tolerance 1e-7)`:`FAIL: Python parity mismatch`;
  function draw(rates){
    const dpr=window.devicePixelRatio||1,W=canvas.clientWidth,H=300;
    canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);
    const left=78,right=W-18,top=18,bottom=H-62,plotW=right-left,plotH=bottom-top;
    ctx.font="12px system-ui";ctx.fillStyle="#415866";ctx.strokeStyle="#d8e2e9";ctx.lineWidth=1;
    if(!rates.length){ctx.fillText("Press Start to generate outcomes.",left+8,top+20);$("chartSummary").textContent="The chart will show all simulated terminal rates.";return;}
    const bins=24,hist=histogramData(rates,bins),counts=hist.counts,maxCount=Math.max(...counts),yStep=Math.max(1,Math.floor(niceStep(maxCount))),yTop=Math.ceil(maxCount/yStep)*yStep;
    const xTickCount=W<440?2:4,xFormat=new Intl.NumberFormat("en-US",{maximumFractionDigits:2});
    ctx.textAlign="right";ctx.textBaseline="middle";
    for(let v=0;v<=yTop;v+=yStep){
      const y=bottom-(v/yTop)*plotH;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(right,y);ctx.stroke();ctx.fillText(fmt(v),left-8,y);
    }
    const binW=plotW/bins;
    ctx.fillStyle="#287ba0";counts.forEach((v,i)=>{const bh=v/yTop*plotH;ctx.fillRect(left+i*binW+1,bottom-bh,Math.max(1,binW-2),bh);});
    ctx.strokeStyle="#647b89";ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,bottom);ctx.lineTo(right,bottom);ctx.stroke();
    ctx.fillStyle="#415866";ctx.textAlign="center";ctx.textBaseline="top";
    for(let i=0;i<=xTickCount;i++){
      const fraction=i/xTickCount,x=left+fraction*plotW,value=hist.min+fraction*(hist.max-hist.min);
      ctx.beginPath();ctx.moveTo(x,bottom);ctx.lineTo(x,bottom+4);ctx.stroke();ctx.fillText(xFormat.format(value),x,bottom+7);
    }
    ctx.font="13px system-ui";ctx.textBaseline="alphabetic";ctx.fillText("Terminal exchange rate (LKR/USD)",(left+right)/2,H-7);
    ctx.save();ctx.translate(17,(top+bottom)/2);ctx.rotate(-Math.PI/2);ctx.fillText("Frequency (simulation paths)",0,0);ctx.restore();
    const binned=counts.reduce((a,b)=>a+b,0),accounted=binned+hist.belowRange+hist.aboveRange;
    if(accounted!==rates.length)throw new Error(`Histogram accounts for ${accounted} of ${rates.length} paths`);
    $("chartSummary").textContent=`All ${fmt(rates.length)} paths shown across the full observed range ${xFormat.format(hist.dataMin)}–${xFormat.format(hist.dataMax)} LKR/USD; outside-range paths: ${hist.belowRange} below, ${hist.aboveRange} above.`;
  }
  function render(){const s=state;$("status").textContent=s?`${s.paused?"Paused":s.done>=s.target?"Complete":"Running"}: ${fmt(s.done)} / ${fmt(s.target)} paths`:'Ready';start.disabled=!!s&&!s.paused&&s.done<s.target;pause.disabled=!s||s.paused||s.done>=s.target;form.querySelectorAll("input").forEach(x=>x.disabled=!!s&&s.done<s.target);draw(s?s.rates:[]);if(!s||s.done<2)return;
    const m=metrics(s.rates,s.invoice,s.budget);$("mean").textContent=`LKR ${fmt(m.mean)}`;$("interval").textContent=`LKR ${fmt(m.ciLow)} – ${fmt(m.ciHigh)}`;$("var").textContent=`LKR ${fmt(m.var95)}`;$("es").textContent=`LKR ${fmt(m.es95)}`;$("budgetOut").textContent=`${(100*m.budgetProbability).toFixed(1)}%`;
  }
  function tick(){if(!state||state.paused)return;const s=state, n=Math.min(250,s.target-s.done),T=s.days/365;for(let i=0;i<n;i++)s.rates.push(REF.rate*Math.exp(REF.alpha*T+REF.sigma*s.multiplier*Math.sqrt(T)*normal(s.rand)));s.done+=n;render();if(s.done<s.target)timer=setTimeout(tick,16);}
  start.onclick=()=>{if(state&&state.paused){state.paused=false;render();tick();return;}const settings=parseSettings(Object.fromEntries(new FormData(form)));if(!form.checkValidity()||!settings){$("status").textContent="Check the input ranges.";return;}const {invoice,days,multiplier,target,seed}=settings;state={invoice,days,multiplier,target,seed,rand:rng(seed),budget:invoice*REF.rate*1.05,rates:[],done:0,paused:false};render();tick();};
  pause.onclick=()=>{if(timer)clearTimeout(timer);timer=null;state.paused=true;render();};
  reset.onclick=()=>{if(timer)clearTimeout(timer);timer=null;state=null;form.reset();for(const id of ["mean","interval","var","es","budgetOut"])$(id).textContent="—";render();};
  render();
});
