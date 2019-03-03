'use strict';

const THREE = require('three');

const NaiveBox = require('./naive_box');

class ShrinkWrapper {

  /**
   * @param target {Mesh}
   * @param text {string}
   */
  constructor(target, text) {

    this.subdividLevel = 5;

    /**
     * @type {Mesh}
     */
    this.target = target;

    let cornerText = ShrinkWrapper.generateCornerPoints(text);

    /**
     * @type {NaiveBox}
     */
    this.geometry = new NaiveBox(cornerText);

    /**
     * @type {MeshBasicMaterial}
     */
    this.material = new THREE.MeshNormalMaterial();

    /**
     * @type {Mesh}
     */
    this.mesh = new THREE.Mesh(this.geometry, this.material);

    /**
     * @type {Array.<Number>}
     */
    this.output = [];
  }

  /**
   * Should generate the corner points according to the paper. However I am
   * simply reading the given data from the author here.
   */
  static generateCornerPoints(text) {
    return text.split('\n').slice(0, 8).join('\n');
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
    this.subdividLevel = repeats;

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
    let oldVertices, oldFaces, oldVerticesNormal;
    let newVertices, newFaces, newVerticesNormals;

    let i, il;

    // new stuff.
    let sourceEdges, newEdgeVertices;

    oldVertices = this.geometry.vertices; // { x, y, z}
    oldVerticesNormal = this.geometry.vertexNormals;
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
    newVerticesNormals = [];
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
        point = this.shootRay(tmp, tmpNormal);
      }


      if (point) {
        newEdge.add(point);
      }

      currentEdge.newEdge = newEdgeVertices.length;

      newEdgeVertices.push(newEdge);
      newVerticesNormals.push(tmpNormal);

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
    newVerticesNormals = oldVerticesNormal.concat(newVerticesNormals);
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
    this.geometry.vertexNormals = newVerticesNormals;
    this.geometry.faces = newFaces;
  }

  /**
   * shot a ray from this vertex util hit a point on the mesh.
   * @param vert {Vector3} from
   * @param dir {Vector3} toward
   * @param invert {boolean}
   * @return {Vector3} the hit point vector
   */
  shootRay(vert, dir, invert = false) {
    const raycaster = new THREE.Raycaster();

    // flip the direction if the previous try missed.
    const direction = invert ? dir.clone().negate() : dir;

    raycaster.set(vert, direction);
    const intersects = raycaster.intersectObject(this.target);

    if (intersects.length > 0) {
      let delta = intersects[0].distance;
      if (invert) delta = -delta;
      this.output.push(delta);

      return intersects[0].point;
    } else {
      // If not found, shot a ray in opposite direction
      return !invert ? this.shootRay(vert, dir, true) : null;
    }
  }
}

module.exports = ShrinkWrapper;
