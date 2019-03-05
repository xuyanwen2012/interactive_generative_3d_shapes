'use strict';

const fs = require('fs');
const path = require('path');
const THREE = require('three');
const Reconstructor = require('./core/reconstructor');

const OBJExporter = require('three-obj-exporter');
const exporter = new OBJExporter();

function reconstruct(args) {
  let inputPath = path.join(__dirname, "../output", args.input);
  if (!fs.existsSync(inputPath)) {
    inputPath = args.input;
  }
  console.log(`loading ${inputPath}`);
  const data = JSON.parse(fs.readFileSync(inputPath));

  console.log(`reconstructing with ${args.levels} subdivision levels`);
  const reconstructor = new Reconstructor(data);
  reconstructor.modify(args.levels);

  const material = new THREE.MeshNormalMaterial();
  const mesh = new THREE.Mesh(reconstructor.geometry, material);
  const result = exporter.parse(mesh);

  const outPath = args.outPath || inputPath.replace('.json', '.gen.obj');
  console.log(`saving as ${outPath}`);
  fs.writeFile(outPath, result, 'utf8', (err) => {
    if (err) throw err;
    console.log('done');
  });
}

module.exports = reconstruct;
