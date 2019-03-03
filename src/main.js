'use strict';

const {PerformanceObserver, performance} = require('perf_hooks');

const loadModel = require('./loader');
const dumpResult = require('./dumper');
const ShrinkWrapper = require('./shrink_wrapper');

function main(filename) {
  let {text, mesh} = loadModel(filename);
  let wrapper = new ShrinkWrapper(mesh, text);
  wrapper.modify(5);
  dumpResult(wrapper, filename);
  console.log(`Processed ${wrapper.output.length} vertices.`);
}

const wrapped = performance.timerify(main);

const obs = new PerformanceObserver((list) => {
  console.log(list.getEntries()[0].duration);
  obs.disconnect();
});
obs.observe({entryTypes: ['function']});

module.exports = wrapped;
