'use strict';

const fs = require('fs');
const path = require('path');

/**
 * @param filename {string} model's filename
 * @param directory {string=}
 * @return {string} The content of the obj file as string.
 */
function loadFile(filename, directory = '../../models') {
  if (fs.existsSync(path.join(__dirname, directory, filename))) {
    return fs.readFileSync(path.join(__dirname, directory, filename)).toString();
  } else {
    return fs.readFileSync(filename).toString();
  }
}

module.exports = loadFile;
