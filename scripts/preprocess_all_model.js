const Promise = require('bluebird');
const fs = Promise.promisifyAll(require('fs'));
const path = require('path');

const ShrinkWrapper = require('../src/core/shrink_wrapper');
const dumpResult = require('../src/core/dumper');
const loadFile = require('../src/core/file_loader');
const parser = require('../src/core/obj_parser');

const modelDirectory = path.join(__dirname, '../models');

function getFiles() {
  return fs.readdirAsync(modelDirectory);
}

function getContent(filename) {
  return fs.readFileAsync(path.join(modelDirectory, filename), 'utf8');
}

async function processModel(filename) {
  const text = loadFile(filename);
  const mesh = parser.parseModel(text);
  const wrapper = new ShrinkWrapper(mesh, text);

  wrapper.modify(5);
  dumpResult(wrapper, filename);

  console.log(`Processed ${wrapper.output.length} vertices.`);
}

processModel('1abeca7159db7ed9f200a72c9245aee7.obj').then(() => console.log('done'));
processModel('1acfbda4ce0ec524bedced414fad522f.obj').then(() => console.log('done'));
processModel('1ae530f49a914595b491214a0cc2380.obj').then(() => console.log('done'));
processModel('1aef0af3cdafb118c6a40bdf315062da.obj').then(() => console.log('done'));
processModel('1b5b5a43e0281030b96212c8f6cd06e.obj').then(() => console.log('done'));
processModel('1b85c850cb4b93a6e9415adaaf77fdbf.obj').then(() => console.log('done'));


// Promise.promisify(processModel);
//
// getFiles().map((filename) => getContent(filename)).then(content => {
//   content.forEach(text => processModel(filename, text));
// });
