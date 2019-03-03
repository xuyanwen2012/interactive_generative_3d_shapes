'use strict';

const THREE = require('three');

const fs = require('fs');
const path = require('path');
const OBJLoader = require('three-obj-loader');
OBJLoader(THREE);

const loader = new THREE.OBJLoader();

/**
 * @param relPath
 * @return {Buffer}
 */
function bufferFile(relPath) {
  return fs.readFileSync(path.join(__dirname, relPath));
}

/**
 *
 * @param filename {string} the filename of the .obj model
 * @param directory {string} the directory that con
 * @return {{mesh:Mesh, text: string}}
 */
function loadModel(filename, directory = '../models') {
  const BUFFER = bufferFile(path.join(directory, filename));
  const text = BUFFER.toString();

  // parse & construct the mesh
  const group = loader.parse(text);
  const mesh = group.children[0];
  mesh.material = new THREE.MeshBasicMaterial({
    side: THREE.DoubleSide,
  });
  mesh.position.set(0, 0, 0);
  mesh.updateMatrixWorld(true);

  return {
    text: text,
    mesh: mesh,
  };
}

module.exports = loadModel;
