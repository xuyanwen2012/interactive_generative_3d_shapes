'use strict';

const fs = require('fs');
const path = require('path');
const THREE = require('three');
const Reconstructor = require('./core/reconstructor');

const OBJExporter = require('three-obj-exporter');
const exporter = new OBJExporter();

const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../output/1abeca7159db7ed9f200a72c9245aee7.json'), 'utf8'));

const reconstructor = new Reconstructor(data);
reconstructor.modify(5);

const material = new THREE.MeshNormalMaterial();
const mesh = new THREE.Mesh(reconstructor.geometry, material);
const result = exporter.parse(mesh);
// console.log(result);

const outPath = path.join(__dirname, '../output', 'sample.obj');
fs.writeFile(outPath, result, 'utf8', (err) => {
  if (err) throw err;
  console.log('done');
});
