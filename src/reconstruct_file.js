'use strict';

const fs = require('fs');
const path = require('path');
const THREE = require('three');
const Reconstructor = require('./core/reconstructor');

const OBJExporter = require('three-obj-exporter');
const exporter = new OBJExporter();

function reconstruct(args) {
  console.dir(args);

  let data;
  if (args.data) {
    // Directly reconstruct data
    data = args.data;
  } else {
    // Reading data from file
    console.log(`loading ${args.input}`);
    data = JSON.parse(fs.readFileSync(args.input));
  }

  console.log(`reconstructing with ${args.levels} subdivision levels`);
  const reconstructor = new Reconstructor(data);
  reconstructor.modify(args.levels);

  const material = new THREE.MeshNormalMaterial();
  const mesh = new THREE.Mesh(reconstructor.geometry, material);
  const result = exporter.parse(mesh);

  console.log(`saving as ${args.output}`);
  fs.writeFileSync(args.output, result, 'utf8');
}

module.exports = reconstruct;
