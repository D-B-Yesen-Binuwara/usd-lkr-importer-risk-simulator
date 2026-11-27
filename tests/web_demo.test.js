const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

const root = path.resolve(__dirname, '..');
const window = {};
vm.runInNewContext(fs.readFileSync(path.join(root, 'docs/reference_config.js'), 'utf8'), {window});
const {exactStep, parityResult, rng, upperTail, metrics, parseSettings} = require('../docs/app.js');

test('frozen Python fixture matches the browser transition and payment', () => {
  const ref = window.IMPORTER_REFERENCE;
  const result = parityResult(ref);
  assert.equal(result.passed, true);
  assert.ok(Math.abs(result.payment - result.rate * ref.invoiceUsd) < 1e-6);
  assert.equal(exactStep(ref.rate, ref.alpha, 0, 0, 0), ref.rate);
});

test('seeded draws repeat and extra-cost tail follows Python convention', () => {
  const a = rng(42), b = rng(42);
  assert.deepEqual(Array.from({length: 5}, () => a()), Array.from({length: 5}, () => b()));
  const tail = upperTail([0, 1, 2, 3, 4], 0.8);
  assert.equal(tail.var95, 3);
  assert.equal(tail.es95, 4);
  const result = metrics([1, 2, 3, 4, 5], 10, 30, 1);
  assert.equal(result.mean, 30);
  assert.equal(result.budgetProbability, 0.4);
  assert.equal(result.var95, 40);
  assert.equal(result.es95, 40);
});

test('declared input boundaries are enforced before seeding a run', () => {
  const valid = {invoice:'100000', days:'90', volatility:'1', paths:'10000', seed:'42'};
  assert.deepEqual(parseSettings(valid), {invoice:100000,days:90,multiplier:1,target:10000,seed:42});
  assert.equal(parseSettings({...valid, seed:'4294967295'}).seed, 4294967295);
  for (const override of [
    {seed:'4294967296'}, {seed:'-1'}, {seed:'1.5'}, {seed:''},
    {invoice:'10000001'}, {invoice:'0'}, {invoice:'Infinity'},
    {days:'366'}, {volatility:''}, {volatility:'3.1'},
    {paths:'20001'}, {paths:'499'},
  ]) assert.equal(parseSettings({...valid, ...override}), null);
});
