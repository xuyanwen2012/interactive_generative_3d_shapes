'use strict';

const fs = require('fs');
const path = require('path');

/**
 * @param wrapper {ShrinkWrapper}
 * @return {Array.<Number>}
 */
function generateOutput(wrapper) {
  let result = [];
  wrapper.cornerPoints.forEach(v => result.push(v.x, v.y, v.z));
  return result.concat(wrapper.output);
}

/**
 * @param wrapper {ShrinkWrapper}
 * @param filename {string}
 */
module.exports = (wrapper, filepath) => {
  const json = JSON.stringify(generateOutput(wrapper));
  console.log(`writing to ${filepath}`);
  fs.writeFileSync(filepath, json, 'utf8');
  // fs.writeFile(filepath, json, 'utf8', (err) => {
  //   if (err) throw err;
  //   console.log(`The file ${outFilename} has been saved!`);
  // });
};
