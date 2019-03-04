'use strict';

const {PerformanceObserver, performance} = require('perf_hooks');

const {loadFile, parseModel} = require('./file_loader');
const dumpResult = require('./dumper');
const ShrinkWrapper = require('./shrink_wrapper');

function main(filename) {
  const text = loadFile(filename);
  const mesh = parseModel(text);
  const wrapper = new ShrinkWrapper(mesh, text);

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

