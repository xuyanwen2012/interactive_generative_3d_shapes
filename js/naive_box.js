class ShrinkWrapper {

  /**
   * @param target {Mesh}
   */
  constructor(target) {

    /**
     * @type {Mesh}
     */
    this.target = target;

    /**
     * @type {NaiveBox}
     */
    this.geometry = new NaiveBox();

    /**
     * @type {MeshBasicMaterial}
     */
    this.material = new THREE.MeshBasicMaterial({
      color: 0xFFFFFF,
    });

    /**
     * @type {Mesh}
     */
    this.mesh = new THREE.Mesh(this.geometry, this.material);

    scene.add(this.mesh);


    /*



    */
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
      // ['1_2', new THREE.Vector3(, ,)], // face
      ['1_3', new THREE.Vector3(1, -1, 0)], // x, -y
      ['2_3', new THREE.Vector3(0, 1, -1)], // -z. y
      ['4_6', new THREE.Vector3(1, 0, 1)], // -x. z
      // ['0_6', new THREE.Vector3()], // face
      ['0_4', new THREE.Vector3(-1, -1, 0)], // -x. -y
      ['2_6', new THREE.Vector3(-1, 1, 0)], // -x. -y
      // ['4_7', new THREE.Vector3()], // face
      ['6_7', new THREE.Vector3(0, 1, 1)], // y,z
      ['5_7', new THREE.Vector3(1, 0, 1)], // x,z
      ['4_5', new THREE.Vector3(0, -1, 1)], // -y,z
      ['1_5', new THREE.Vector3(1, -1, 0)], // x,-y
      // ['1_7', new THREE.Vector3()], // face
      ['3_7', new THREE.Vector3(1, 1, 0)], // x,y
      // ['2_7', new THREE.Vector3(1, 1, 0)], // face
      // ['0_5', new THREE.Vector3(1, 1, 0)], // face
    ]);

    this.predefinedNormals.values().forEach(v => v.normalize());

    this.initHelpers();
  }

  initHelpers() {
    this.createEdgeHelper();
    this.createVertexNormalHelper();
  }

  /**
   * @private
   */
  createEdgeHelper() {
    const edges = new THREE.EdgesGeometry(this.geometry);
    const line = new THREE.LineSegments(edges, new THREE.LineBasicMaterial({color: 0x0000ff}));
    scene.add(line);
  }

  /**
   * @private
   */
  createVertexNormalHelper() {
    const helper = new THREE.VertexNormalsHelper(this.mesh, 2, 0x00ff00, 1);
    scene.add(helper);
  }

  /**
   * @private
   * @param a
   * @param b
   * @param vertices
   * @param map
   * @param face
   * @param metaVertices
   */
  processEdge(a, b, vertices, map, face, metaVertices) {

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
        // aIndex: a, // numbered reference
        // bIndex: b,
        faces: [] // pointers to face

      };

      map[key] = edge;

    }

    edge.faces.push(face);

    metaVertices[a].edges.push(edge);
    metaVertices[b].edges.push(edge);
  }

  /**
   * @private
   * @param vertices
   * @param faces
   * @param metaVertices
   * @param edges
   */
  generateLookups(vertices, faces, metaVertices, edges) {

    let i, il, face, edge;

    for (i = 0, il = vertices.length; i < il; i++) {

      metaVertices[i] = {edges: []};

    }

    for (i = 0, il = faces.length; i < il; i++) {
      face = faces[i];

      this.processEdge(face.a, face.b, vertices, edges, face, metaVertices);
      this.processEdge(face.b, face.c, vertices, edges, face, metaVertices);
      this.processEdge(face.c, face.a, vertices, edges, face, metaVertices);
    }
  }

  /**
   * Main function
   */
  shrink() {
    const ABC = ['a', 'b', 'c'];
    // let tmp = new THREE.Vector3();

    let oldVertices, oldFaces;
    let newVertices, newFaces; // newUVs = [];

    let n, l, i, il, j, k;
    let metaVertices;

    // new stuff.
    let sourceEdges, newEdgeVertices, newSourceVertices;

    oldVertices = this.geometry.vertices; // { x, y, z}
    oldFaces = this.geometry.faces; // { a: oldVertex1, b: oldVertex2, c: oldVertex3 }

    /******************************************************
     *
     * Step 0: Preprocess Geometry to Generate edges Lookup
     *
     *******************************************************/

    metaVertices = new Array(oldVertices.length);
    sourceEdges = {}; // Edge => { oldVertex1, oldVertex2, faces[]  }

    this.generateLookups(oldVertices, oldFaces, metaVertices, sourceEdges);

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
    let other, currentEdge, newEdge, face;
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

      console.log(i, currentEdge, newEdge);
      this.debugShowPoint(tmp);
      this.debugProjectPintUp(tmp);
    }


    /******************************************************
     *
     *  Step 3.
     *  Generate Faces between source vertecies
     *  and edge vertices.
     *
     *******************************************************/
  }

  /**
   * @param pos {Vector3}
   * @param color {Number}
   */
  debugShowPoint(pos, color = 0xFACADE) {
    const dotGeometry = new THREE.Geometry();
    dotGeometry.vertices.push(pos);
    const dotMaterial = new THREE.PointsMaterial({
      size: 0.5,
      color: color,
    });
    const dot = new THREE.Points(dotGeometry, dotMaterial);
    scene.add(dot);
  }

  /**
   * @param vert {Vector3}
   */
  debugProjectPintUp(vert) {
    // shot a ray from this vertex up util hit a point
    const raycaster = new THREE.Raycaster();
    raycaster.set(vert, new THREE.Vector3(0, 1, 0));

    const intersects = raycaster.intersectObject(this.target);

    // Toggle rotation bool for meshes that we clicked
    if (intersects.length > 0) {
      console.log('hit');
      this.debugShowPoint(intersects[0].point, 0xFF0000);
    } else {
      console.log('missed');
    }
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
    this.faces.push(new THREE.Face3(4, 6, 0));
    this.faces.push(new THREE.Face3(2, 0, 6));
    this.faces.push(new THREE.Face3(6, 4, 7));
    this.faces.push(new THREE.Face3(5, 7, 4));
    this.faces.push(new THREE.Face3(5, 1, 7));
    this.faces.push(new THREE.Face3(3, 7, 1));
    this.faces.push(new THREE.Face3(3, 2, 7));
    this.faces.push(new THREE.Face3(6, 7, 2));
    this.faces.push(new THREE.Face3(4, 0, 5));
    this.faces.push(new THREE.Face3(1, 5, 0));

    this.computeVertexNormals();
    this.computeFaceNormals();
    this.computeBoundingBox();
  }
}
