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

    scene.add(this.mesh);

    /**
     *          7____6
     *        3/___2/|
     *       | 5__|_4
     *      1/___0/
     * @type {Map<string, Vector3>}
     */
    this.predefinedNormals = new Map([
      ['0_1', new THREE.Vector3(0, -1, -1)], // -z, -y
      ['0_2', new THREE.Vector3(-1, 0, -1)], // -x, -z
      ['1_2', new THREE.Vector3(0, 0, -1)], // face: (SUBJECT TO CHANGE) -z
      ['1_3', new THREE.Vector3(1, 0, -1)], // x, -z
      ['2_3', new THREE.Vector3(0, 1, -1)], // -z. y
      ['4_6', new THREE.Vector3(-1, 0, 1)], // -x. z
      ['0_6', new THREE.Vector3(-1, 0, 0)], // face: (SUBJECT TO CHANGE) -x
      ['0_4', new THREE.Vector3(-1, -1, 0)], // -x. -y
      ['2_6', new THREE.Vector3(-1, 1, 0)], // -x. -y
      ['4_7', new THREE.Vector3(0, 0, 1)], // face: (SUBJECT TO CHANGE) z
      ['6_7', new THREE.Vector3(0, 1, 1)], // y,z
      ['5_7', new THREE.Vector3(1, 0, 1)], // x,z
      ['4_5', new THREE.Vector3(0, -1, 1)], // -y,z
      ['1_5', new THREE.Vector3(1, -1, 0)], // x,-y
      ['3_5', new THREE.Vector3(1, 0, 0)], // face: (SUBJECT TO CHANGE) x
      ['3_7', new THREE.Vector3(1, 1, 0)], // x,y
      ['3_6', new THREE.Vector3(0, 1, 0)], // face: (SUBJECT TO CHANGE) y
      ['0_5', new THREE.Vector3(0, -1, 0)], // face: (SUBJECT TO CHANGE) -y

      // ['0_1', new THREE.Vector3(0, -1, -1)], // -z, -y
      // ['0_2', new THREE.Vector3(-1, 0, -1)], // -x, -z
      // ['0_3', new THREE.Vector3(0, 0, -1)], // face: (SUBJECT TO CHANGE) -z
      // ['1_3', new THREE.Vector3(1, 0, -1)], // x, -z
      // ['2_3', new THREE.Vector3(0, 1, -1)], // -z. y
      // ['4_6', new THREE.Vector3(-1, 0, 1)], // -x. z
      // ['2_4', new THREE.Vector3(-1, 0, 0)], // face: (SUBJECT TO CHANGE) -x
      // ['0_4', new THREE.Vector3(-1, -1, 0)], // -x. -y
      // ['2_6', new THREE.Vector3(-1, 1, 0)], // -x. -y
      // ['5_6', new THREE.Vector3(0, 0, 1)], // face: (SUBJECT TO CHANGE) z
      // ['6_7', new THREE.Vector3(0, 1, 1)], // y,z
      // ['5_7', new THREE.Vector3(1, 0, 1)], // x,z
      // ['4_5', new THREE.Vector3(0, -1, 1)], // -y,z
      // ['1_5', new THREE.Vector3(1, -1, 0)], // x,-y
      // ['1_7', new THREE.Vector3(1, 0, 0)], // face: (SUBJECT TO CHANGE) x
      // ['3_7', new THREE.Vector3(1, 1, 0)], // x,y
      // ['2_7', new THREE.Vector3(0, 1, 0)], // face: (SUBJECT TO CHANGE) y
      // ['1_4', new THREE.Vector3(0, -1, 0)], // face: (SUBJECT TO CHANGE) -y

    ]);

    this.predefinedNormals.forEach((v, key) => v.normalize());

    this.initHelpers();
  }

  initHelpers() {
    this.recreateEdgeHelper();
  }

  /**
   * @private
   */
  recreateEdgeHelper() {
    if (this.edgeHelper) {
      scene.remove(this.edgeHelper);
    }

    const edges = new THREE.EdgesGeometry(this.geometry, 0); // show all edge

    /**
     * @type {LineSegments}
     */
    this.edgeHelper = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({color: 0x0000ff}));
    scene.add(this.edgeHelper);
  }

  /**
   * @private
   * @param a
   * @param b
   * @param vertices
   * @param map
   * @param face
   */
  processEdge(a, b, vertices, map, face) {
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
   * @param vertices
   * @param faces
   * @param edges
   */
  generateLookups(vertices, faces, edges) {
    faces.forEach(face => {
      this.processEdge(face.a, face.b, vertices, edges, face);
      this.processEdge(face.b, face.c, vertices, edges, face);
      this.processEdge(face.c, face.a, vertices, edges, face);
    });
  }

  /**
   * Main function to shrink wrap
   */
  modify() {
    let repeats = 5;

    while (repeats-- > 0) {
      this.shrink();
    }

    delete this.geometry.__tmpVertices;

    this.geometry.computeFaceNormals();
    this.geometry.computeVertexNormals();

    this.recreateEdgeHelper();
  }


  shrink2() {
    let oldVertices = this.geometry.vertices; // { x, y, z}
    let oldFaces = this.geometry.faces; // { a, b, c }
    let oldQuads = this.geometry.quads;
    let sourceEdges = {}; // Edge => { oldVertex1, oldVertex2, faces[]  }

    // Preprocess
    oldQuads.forEach(quads => {
      // this.processEdge(quads.a, quads.b, oldVertices, sourceEdges, )
      // this.processEdge(face.a, face.b, vertices, edges, face);
      // this.processEdge(face.b, face.c, vertices, edges, face);
      // this.processEdge(face.c, face.a, vertices, edges, face);
    });

    console.log(sourceEdges);
  }

  /**
   * @private
   */
  shrink() {
    let oldVertices, oldFaces, oldVerticiesNormal;
    let newVertices, newFaces, newVerticiesNormals; // newUVs = [];

    let i, il;
    let metaVertices;

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

    metaVertices = new Array(oldVertices.length);
    sourceEdges = {}; // Edge => { oldVertex1, oldVertex2, faces[]  }

    this.generateLookups(oldVertices, oldFaces, sourceEdges);

    console.log(metaVertices);
    console.log(sourceEdges);

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
      // scene.add(new THREE.ArrowHelper(normalA, currentEdge.a, 1, 0xffff00));
      // scene.add(new THREE.ArrowHelper(normalB, currentEdge.b, 1, 0xffff00));

      // calculate normal
      let tmpNormal = new THREE.Vector3();
      tmpNormal.set(0, 0, 0);
      tmpNormal.addVectors(normalA, normalB).divideScalar(2).normalize();

      if (tmpNormal) {
        // Subject to remove
        // const arrowHelper = new THREE.ArrowHelper(tmpNormal, tmp, 1, 0xffff00);
        // scene.add(arrowHelper);

        point = this.debugProjectPint(tmp, tmpNormal);
      }

      // this.debugShowPoint(tmp);

      if (point) {
        newEdge.add(point);
      }

      currentEdge.newEdge = newEdgeVertices.length;

      newEdgeVertices.push(newEdge);
      newVerticiesNormals.push(tmpNormal);

      console.log(i, currentEdge, newEdge);
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

      edge1 = this.getEdge(face.a, face.b, sourceEdges).newEdge + sl;
      edge2 = this.getEdge(face.b, face.c, sourceEdges).newEdge + sl;
      edge3 = this.getEdge(face.c, face.a, sourceEdges).newEdge + sl;

      // console.log(`${edge1.a}_${edge1.b}`);
      // console.log(this.getEdge(face.a, face.b, sourceEdges));

      // create 4 faces.

      this.newFace(newFaces, edge1, edge2, edge3);
      this.newFace(newFaces, face.a, edge1, edge3);
      this.newFace(newFaces, face.b, edge2, edge1);
      this.newFace(newFaces, face.c, edge3, edge2);
    }

    // Overwrite old arrays
    this.geometry.vertices = newVertices;
    this.geometry.vertexNormals = newVerticiesNormals;
    this.geometry.faces = newFaces;

    // this.geometry.verticesNeedUpdate = true;
    // console.log(newFaces);
  }

  /**
   * @private
   * @param newFaces
   * @param a
   * @param b
   * @param c
   */
  newFace(newFaces, a, b, c) {
    newFaces.push(new THREE.Face3(a, b, c));
  }

  /**
   * @private
   * @param a
   * @param b
   * @param map
   * @return {*}
   */
  getEdge(a, b, map) {
    let vertexIndexA = Math.min(a, b);
    let vertexIndexB = Math.max(a, b);
    let key = `${vertexIndexA}_${vertexIndexB}`;
    return map[key];
  }

  /**
   * @param pos {Vector3}
   * @param color {Number}
   */
  debugShowPoint(pos, color = 0xFACADE) {
    const dotGeometry = new THREE.Geometry();
    dotGeometry.vertices.push(pos);
    const dotMaterial = new THREE.PointsMaterial({
      size: 0.1,
      color: color,
    });
    const dot = new THREE.Points(dotGeometry, dotMaterial);
    scene.add(dot);
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

      return intersects[0].point;

    } else {
      // If not found, shot a ray in opposite direction
      console.log('missed');
      if (step === 0) {
        return this.debugProjectPint(vert, dir, step + 1);
      } else {
        return null;
      }
    }


  }
}

class QuadBox extends THREE.Geometry {
  constructor() {
    super();

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

    /**
     * @type {Array.<{a,b,c,d}>}
     */
    this.quads = [
      {a: 3, b: 1, c: 0, d: 2},
      {a: 2, b: 0, c: 4, d: 6},
      {a: 6, b: 4, c: 5, d: 7},
      {a: 7, b: 5, c: 1, d: 3},
      {a: 7, b: 3, c: 2, d: 6},
      {a: 1, b: 5, c: 4, d: 0},
    ]; // {a, b, c, d}

    this.quads.forEach(quad => this.drawQuadFace(quad));

    this.computeFaceNormals();
    this.computeVertexNormals();
  }

  drawQuadFace({a, b, c, d}) {

    /*
     *  a - d
     *  | \ |
     *  b - c
     */
    this.drawTriFace(a, b, c);
    this.drawTriFace(c, d, a);
  }

  drawTriFace(a, b, c) {
    this.faces.push(new THREE.Face3(a, b, c));
  }


}

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

    // this.vertexNormals = this.vertexNormals.map(v => v.normalize());
    this.vertexNormals.forEach((v, key) => v.normalize());

    // this.vertices.push(
    //   new THREE.Vector3(-0.746645, 0.235, -1.965),
    //   new THREE.Vector3(0.746641, 0.235, -1.965),
    //   new THREE.Vector3(-0.755, 0.735, -1.77438),
    //   new THREE.Vector3(0.755, 0.735, -1.77438),
    //   new THREE.Vector3(-0.791225, 0.295, 1.925),
    //   new THREE.Vector3(0.791223, 0.295, 1.925),
    //   new THREE.Vector3(-0.745, 0.942911, 1.895),
    //   new THREE.Vector3(0.745, 0.942911, 1.895),
    // );

    // this.colors.push(
    //   new THREE.Color(0xFF0000), // red
    //   new THREE.Color(0x00FF00), // green
    //   new THREE.Color(0x0000FF), // blue
    //   new THREE.Color(0xFF00FF), // purple
    //   new THREE.Color(0x00FFFF), // aqua
    //   new THREE.Color(0xFFFF00), // yellow
    //   new THREE.Color(0xFACADE), // pink
    //   new THREE.Color(0xFFFFFF), // white
    // );

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

    // this.computeVertexNormals();
    // this.computeFaceNormals();
    // this.computeBoundingBox();
  }
}
