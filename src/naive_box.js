'use strict';

const THREE = require('three');

class NaiveBox extends THREE.Geometry {

  /**
   * @param cornerPoints {Array.<Vector3>} Must contain 8 Vector3 describing
   * the corresponding corner points.
   */
  constructor(cornerPoints) {
    super();

    this.vertices = cornerPoints.slice(0);
    this.faces.length = 0;

    /**
     *      Note the index here should match the .vertices
     *
     *      @type {Array.<Vector3>}
     */
    this.vertexNormals = [
      new THREE.Vector3(-1, -1, -1),
      new THREE.Vector3(1, -1, -1),
      new THREE.Vector3(-1, 1, -1),
      new THREE.Vector3(1, 1, -1),
      new THREE.Vector3(-1, -1, 1),
      new THREE.Vector3(1, -1, 1),
      new THREE.Vector3(-1, 1, 1),
      new THREE.Vector3(1, 1, 1),
    ];

    this.vertexNormals.forEach(v => v.normalize());

    /*
        7____6      y
      3/___2/|    x | z
      | 5__|_4     \|/
      1/___0/
    */
    this.faces.push(new THREE.Face3(1, 0, 2));
    this.faces.push(new THREE.Face3(3, 1, 2));
    this.faces.push(new THREE.Face3(0, 4, 6));
    this.faces.push(new THREE.Face3(2, 0, 6));
    this.faces.push(new THREE.Face3(4, 5, 7));
    this.faces.push(new THREE.Face3(6, 4, 7));
    this.faces.push(new THREE.Face3(5, 1, 3));
    this.faces.push(new THREE.Face3(7, 5, 3));
    this.faces.push(new THREE.Face3(3, 2, 6));
    this.faces.push(new THREE.Face3(7, 3, 6));
    this.faces.push(new THREE.Face3(5, 4, 0));
    this.faces.push(new THREE.Face3(1, 5, 0));
  }
}

module.exports = NaiveBox;
