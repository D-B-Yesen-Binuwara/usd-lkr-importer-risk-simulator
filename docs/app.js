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
if(typeof module!=="undefined")module.exports={exactStep,parityResult,rng,normal,upperTail,metrics,parseSettings};
if(typeof document!=="undefined")document.addEventListener("DOMContentLoaded",()=>{
  const $=id=>document.getElementById(id), fmt=x=>new Intl.NumberFormat("en-US",{maximumFractionDigits:0}).format(x);
  const form=$("settings"), start=$("start"), pause=$("pause"), reset=$("reset"), canvas=$("chart"), ctx=canvas.getContext("2d");
  let state=null,timer=null;
  const parity=parityResult();$("parity").textContent=parity.passed?`PASS: Python parity rate ${parity.rate.toFixed(8)} LKR/USD; payment LKR ${parity.payment.toFixed(2)} (relative tolerance 1e-7)`:`FAIL: Python parity mismatch`;
  function draw(rates){
    const w=canvas.width=canvas.clientWidth*devicePixelRatio,h=canvas.height=300*devicePixelRatio;ctx.scale(devicePixelRatio,devicePixelRatio);const W=w/devicePixelRatio,H=h/devicePixelRatio;ctx.clearRect(0,0,W,H);
    ctx.fillStyle="#415866";ctx.font="13px system-ui";
    if(!rates.length){ctx.fillText("Press Start to generate payment-cost outcomes.",16,35);return;}
    const sorted=[...rates].sort((a,b)=>a-b), lo=sorted[Math.floor(.01*(sorted.length-1))],hi=sorted[Math.ceil(.99*(sorted.length-1))],bins=24,counts=Array(bins).fill(0);
    for(const x of rates){const b=Math.max(0,Math.min(bins-1,Math.floor((x-lo)/(hi-lo||1)*bins)));counts[b]++;}
    const max=Math.max(...counts), left=42,bottom=H-35,plotW=W-left-10,plotH=H-60;
    ctx.fillStyle="#287ba0";counts.forEach((v,i)=>{const bh=v/max*plotH;ctx.fillRect(left+i*plotW/bins,bottom-bh,Math.max(1,plotW/bins-2),bh);});
    ctx.fillStyle="#415866";ctx.fillText(`${fmt(lo)} LKR/USD`,left,H-10);ctx.fillText(`${fmt(hi)} LKR/USD`,Math.max(left,W-120),H-10);ctx.fillText("Paths",7,20);
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
