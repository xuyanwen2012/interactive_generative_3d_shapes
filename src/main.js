'use strict';

const {PerformanceObserver, performance} = require('perf_hooks');

const ShrinkWrapper = require('./shrink_wrapper');
const dumpResult = require('./dumper');
const loadFile = require('./file_loader');
const parser = require('./obj_parser');

function main(filename) {
  const text = loadFile(filename);
  const mesh = parser.parseModel(text);
  const wrapper = new ShrinkWrapper(mesh);

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

