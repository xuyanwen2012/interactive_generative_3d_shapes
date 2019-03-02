'use strict';
const {PerformanceObserver, performance} = require('perf_hooks');
const fs = require('fs');
const path = require('path');
const THREE = require('three');
const OBJLoader = require('three-obj-loader');
OBJLoader(THREE);

const BUFFER = bufferFile('./models/1abeca7159db7ed9f200a72c9245aee7.obj');

/**
 * @param relPath
 * @return {Buffer}
 */
function bufferFile(relPath) {
  return fs.readFileSync(path.join(__dirname, relPath));
}

const loader = new THREE.OBJLoader();

let text = BUFFER.toString();

let scene = new THREE.Scene();
let group = loader.parse(text);
let mesh = group.children[0];
mesh.material = new THREE.MeshNormalMaterial();
mesh.position.set(0, 0, 0);
mesh.updateMatrixWorld(true);
group.updateMatrixWorld(true);
scene.add(mesh);

class NaiveBox extends THREE.Geometry {
  constructor() {
    super();

    this.vertices.length = 0;
    this.faces.length = 0;

    this.vertices.push(
      new THREE.Vector3(-0.695, 0.125, -2.04054),
      new THREE.Vector3(0.695, 0.125, -2.04054),
      new THREE.Vector3(-0.705, 0.691878, -1.845),
      new THREE.Vector3(0.705, 0.691878, -1.845),
      new THREE.Vector3(-0.665, 0.285, 2.03903),
      new THREE.Vector3(0.665, 0.285, 2.03903),
      new THREE.Vector3(-0.578869, 0.975, 2.045),
      new THREE.Vector3(0.57887, 0.975, 2.045),
    );

    /*
        7____6      y
      3/___2/|    x | z
      | 5__|_4     \|/
      1/___0/
    */

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

    this.vertexNormals.forEach((v, key) => v.normalize());

    /*
        7____6
      3/___2/|
      | 5__|_4
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

class ShrinkWrapper {

  /**
   * @param target {Mesh}
   */
  constructor(target) {

    /**
     * @type {Mesh}
     */
    this.target = target;

    this.geometry = new NaiveBox();

    /**
     * @type {MeshBasicMaterial}
     */
    this.material = new THREE.MeshNormalMaterial();
    // this.material = new THREE.MeshBasicMaterial({color: 0xFFFFFF});

    /**
     * @type {Mesh}
     */
    this.mesh = new THREE.Mesh(this.geometry, this.material);

    /**
     * @type {Array}
     */
    this.output = [];
  }

  /**
   * @private
   * @param a
   * @param b
   * @param vertices
   * @param map
   * @param face
   */
  static processEdge(a, b, vertices, map, face) {
    let vertexIndexA = Math.min(a, b);
    let vertexIndexB = Math.max(a, b);

    let key = `${vertexIndexA}_${vertexIndexB}`;

    let edge;

    if (key in map) {

      edge = map[key];

    } else {

      let vertexA = vertices[vertexIndexA];
      let vertexB = vertices[vertexIndexB];

      edge = {
        a: vertexA, // pointer reference
        b: vertexB,
        newEdge: null,
        faces: [] // pointers to face
      };

      map[key] = edge;
    }

    edge.faces.push(face);
  }

  /**
   * @private
   * @param newFaces
   * @param a
   * @param b
   * @param c
   */
  static newFace(newFaces, a, b, c) {
    newFaces.push(new THREE.Face3(a, b, c));
  }

  /**
   * @private
   * @param a
   * @param b
   * @param map
   * @return {*}
   */
  static getEdge(a, b, map) {
    let vertexIndexA = Math.min(a, b);
    let vertexIndexB = Math.max(a, b);
    let key = `${vertexIndexA}_${vertexIndexB}`;
    return map[key];
  }

  /**
   * @private
   * @param vertices
   * @param faces
   * @param edges
   */
  generateLookups(vertices, faces, edges) {
    faces.forEach(face => {
      ShrinkWrapper.processEdge(face.a, face.b, vertices, edges, face);
      ShrinkWrapper.processEdge(face.b, face.c, vertices, edges, face);
      ShrinkWrapper.processEdge(face.c, face.a, vertices, edges, face);
    });
  }

  /**
   * Main function to shrink wrap
   * @param repeats {Number}
   */
  modify(repeats = 5) {

    while (repeats-- > 0) {
      this.shrink();
    }

    delete this.geometry.__tmpVertices;

    this.geometry.computeFaceNormals();
    this.geometry.computeVertexNormals();
  }

  /**
   * @private
   */
  shrink() {
    let oldVertices, oldFaces, oldVerticiesNormal;
    let newVertices, newFaces, newVerticiesNormals; // newUVs = [];

    let i, il;

    // new stuff.
    let sourceEdges, newEdgeVertices;

    oldVertices = this.geometry.vertices; // { x, y, z}
    oldVerticiesNormal = this.geometry.vertexNormals;
    oldFaces = this.geometry.faces; // { a: oldVertex1, b: oldVertex2, c: oldVertex3 }

    /******************************************************
     *
     * Step 0: Preprocess Geometry to Generate edges Lookup
     *
     *******************************************************/

    sourceEdges = {}; // Edge => { oldVertex1, oldVertex2, faces[]  }

    this.generateLookups(oldVertices, oldFaces, sourceEdges);

    // console.log(sourceEdges);

    /******************************************************
     *
     *  Step 1.
     *  For each edge, create a new Edge Vertex,
     *  then position it.
     *
     *******************************************************/

    newEdgeVertices = [];
    newVerticiesNormals = [];
    let currentEdge, newEdge, face;
    let connectedFaces;

    for (i in sourceEdges) {
      currentEdge = sourceEdges[i];
      newEdge = new THREE.Vector3();

      connectedFaces = currentEdge.faces.length;

      // check how many linked faces. 2 should be correct.
      if (connectedFaces !== 2) {

        console.warn('Subdivision Modifier: Number of connected faces != 2, is: ', connectedFaces, currentEdge);
      }

      // IVAN: find the center point of this edge
      let tmp = new THREE.Vector3();
      tmp.set(0, 0, 0);
      tmp.addVectors(currentEdge.a, currentEdge.b).divideScalar(2);

      /**
       * @type {Vector3}
       */
        // let normal = this.predefinedNormals.get(i);
      let point = tmp;

      // show vertex normal
      let fff = i.split('_');
      let aIndex = parseInt(fff[0]);
      let bIndex = parseInt(fff[1]);
      let normalA = this.geometry.vertexNormals[aIndex];
      let normalB = this.geometry.vertexNormals[bIndex];

      // calculate normal
      let tmpNormal = new THREE.Vector3();
      tmpNormal.set(0, 0, 0);
      tmpNormal.addVectors(normalA, normalB).divideScalar(2).normalize();

      if (tmpNormal) {
        point = this.debugProjectPint(tmp, tmpNormal);
      }


      if (point) {
        newEdge.add(point);
      }

      currentEdge.newEdge = newEdgeVertices.length;

      newEdgeVertices.push(newEdge);
      newVerticiesNormals.push(tmpNormal);

      // console.log(i, currentEdge, newEdge);
    }

    /******************************************************
     *
     *  Step 3.
     *  Generate Faces between source vertecies
     *  and edge vertices.
     *
     *******************************************************/

    newVertices = oldVertices.concat(newEdgeVertices);
    newVerticiesNormals = oldVerticiesNormal.concat(newVerticiesNormals);
    let sl = oldVertices.length, edge1, edge2, edge3;
    newFaces = [];

    // console.log(newVertices);

    for (i = 0, il = oldFaces.length; i < il; i++) {

      face = oldFaces[i];

      // find the 3 new edges vertex of each old face
      // Edge => { oldVertex1, oldVertex2, faces[]  }

      edge1 = ShrinkWrapper.getEdge(face.a, face.b, sourceEdges).newEdge + sl;
      edge2 = ShrinkWrapper.getEdge(face.b, face.c, sourceEdges).newEdge + sl;
      edge3 = ShrinkWrapper.getEdge(face.c, face.a, sourceEdges).newEdge + sl;

      // console.log(`${edge1.a}_${edge1.b}`);
      // console.log(this.getEdge(face.a, face.b, sourceEdges));

      // create 4 faces.

      ShrinkWrapper.newFace(newFaces, edge1, edge2, edge3);
      ShrinkWrapper.newFace(newFaces, face.a, edge1, edge3);
      ShrinkWrapper.newFace(newFaces, face.b, edge2, edge1);
      ShrinkWrapper.newFace(newFaces, face.c, edge3, edge2);
    }

    // Overwrite old arrays
    this.geometry.vertices = newVertices;
    this.geometry.vertexNormals = newVerticiesNormals;
    this.geometry.faces = newFaces;
  }

  /**
   * @param vert {Vector3}
   * @param dir {Vector3}
   * @param step {Number}
   */
  debugProjectPint(vert, dir, step = 0) {
    // shot a ray from this vertex up util hit a point
    const raycaster = new THREE.Raycaster();

    // flip the direction if the previous try missed.
    let direction = dir;
    if (step !== 0) {
      direction = dir.clone().negate();
    }

    raycaster.set(vert, direction);

    const intersects = raycaster.intersectObject(this.target);

    // Toggle rotation bool for meshes that we clicked
    if (intersects.length > 0) {
      // this.debugShowPoint(intersects[0].point, 0xFF0000);
      // console.log(intersects[0].distance);
      this.output.push(intersects[0].distance);

      return intersects[0].point;

    } else {
      // If not found, shot a ray in opposite direction
      if (step === 0) {
        return this.debugProjectPint(vert, dir, step + 1);
      } else {
        return null;
      }
    }
  }
}

function main() {
  let wrapper = new ShrinkWrapper(mesh);
  wrapper.modify(1);
  console.log(`Processed ${wrapper.output.length} vertices.`);
}

const wrapped = performance.timerify(main);

const obs = new PerformanceObserver((list) => {
  console.log(list.getEntries()[0].duration);
  obs.disconnect();
});
obs.observe({entryTypes: ['function']});

// A performance timeline entry will be created
wrapped();




