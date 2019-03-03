'use strict';

const {PerformanceObserver, performance} = require('perf_hooks');

const loadModel = require('./src/loader');
const ShrinkWrapper = require('./src/shrink_wrapper');

function main() {
  let {text, mesh} = loadModel('1abeca7159db7ed9f200a72c9245aee7.obj');
  let wrapper = new ShrinkWrapper(mesh, text);
  wrapper.modify(2);
  console.log(wrapper.output);
  console.log(`Processed ${wrapper.output.length} vertices.`);
}

const wrapped = performance.timerify(main);

const obs = new PerformanceObserver((list) => {
  console.log(list.getEntries()[0].duration);
  obs.disconnect();
});
obs.observe({entryTypes: ['function']});

// A performance timeline entry will be created
wrapped();




