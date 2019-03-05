'use strict';

const {PerformanceObserver, performance} = require('perf_hooks');

const ShrinkWrapper = require('./core/shrink_wrapper');
const dumpResult = require('./core/dumper');
const loadFile = require('./core/file_loader');
const parser = require('./core/obj_parser');

function process_file(args) {
  let filename = args.input;
  console.log(`loading ${filename}`);
  console.dir(args.input);
  const text = loadFile(filename);
  const mesh = parser.parseModel(text);
  const wrapper = new ShrinkWrapper(mesh, text);

  console.log(`processing with ${args.levels} subdivision levels`);
  wrapper.modify(args.levels);
  dumpResult(wrapper, filename);

  console.log(`Processed ${wrapper.output.length} vertices.`);
}

const wrapped = performance.timerify(process_file);

const obs = new PerformanceObserver((list) => {
  console.log(list.getEntries()[0].duration);
  obs.disconnect();
});
obs.observe({entryTypes: ['function']});

module.exports = wrapped;
