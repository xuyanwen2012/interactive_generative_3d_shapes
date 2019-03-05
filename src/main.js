'use strict';

const {PerformanceObserver, performance} = require('perf_hooks');

const ShrinkWrapper = require('./core/shrink_wrapper');
const dumpResult = require('./core/dumper');
const loadFile = require('./core/file_loader');
const parser = require('./core/obj_parser');

function main(args) {
  let filename = args.input;
  console.dir(args.input);
  const text = loadFile(filename);
  const mesh = parser.parseModel(text);
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

