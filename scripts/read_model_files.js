const Promise = require('bluebird');
const fs = require('fs');
const path = require('path');

const modelDirectory = './models/';

let output = {models: []};

// Read files names, then store them to JSON
readFiles().then(filenames => {
  // Remove _index.json
  const index = filenames.indexOf('_index.json');
  if (index !== -1) {
    filenames.splice(index, 1);
  }

  // write to file
  output.models = filenames;

  fs.writeFileSync(
    path.join(modelDirectory, '_index.json'),
    JSON.stringify(output),
    'utf-8',
    (err) => {
      if (err) {
        throw err;
      } else {
        console.log('complete');
      }
    });
});

function readFiles() {
  const readdirAsync = Promise.promisify(fs.readdir);
  return readdirAsync(modelDirectory);
}
