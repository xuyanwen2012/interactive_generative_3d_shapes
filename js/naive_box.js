class NaiveBox extends THREE.Geometry {
  constructor() {
    super();

    this.initBox();
  }

  initBox() {
    // Temp
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

    this.colors.push(
      new THREE.Color(0xFF0000), // red
      new THREE.Color(0x00FF00), // green
      new THREE.Color(0x0000FF), // blue
      new THREE.Color(0xFF00FF), // purple
      new THREE.Color(0x00FFFF), // aqua
      new THREE.Color(0xFFFF00), // yellow
      new THREE.Color(0xFACADE), // pink
      new THREE.Color(0xFFFFFF), // white
    );

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

    this.smooth();
  }

  processEdge(a, b, vertices, map, face, metaVertices) {

    let vertexIndexA = Math.min(a, b);
    let vertexIndexB = Math.max(a, b);

    let key = vertexIndexA + "_" + vertexIndexB;

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

  smooth() {
    const ABC = ['a', 'b', 'c'];
    // let tmp = new THREE.Vector3();

    let oldVertices, oldFaces;
    let newVertices, newFaces; // newUVs = [];

    let n, l, i, il, j, k;
    let metaVertices;

    // new stuff.
    let sourceEdges, newEdgeVertices, newSourceVertices;

    oldVertices = this.vertices; // { x, y, z}
    oldFaces = this.faces; // { a: oldVertex1, b: oldVertex2, c: oldVertex3 }

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
    let edgeVertexWeight, adjacentVertexWeight, connectedFaces;

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

      // console.log(currentEdge, newEdge);
      console.log(tmp);
      this.debugShowPoint(tmp);
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
   */
  debugShowPoint(pos) {
    const dotGeometry = new THREE.Geometry();
    dotGeometry.vertices.push(pos);
    const dotMaterial = new THREE.PointsMaterial({
      size: 0.5,
      color: 0xFACADE,
      // sizeAttenuation: false
    });
    const dot = new THREE.Points(dotGeometry, dotMaterial);
    // dot.position.set(pos);
    scene.add(dot);
  }
}
