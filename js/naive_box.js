class NaiveBox extends THREE.Geometry {
  constructor(...args) {
    super();

    // const indices = new Uint16Array([
    //   0, 1,
    //   1, 2,
    //   2, 3,
    //   3, 0,
    //   4, 5,
    //   5, 6,
    //   6, 7,
    //   7, 4,
    //   0, 4,
    //   1, 5,
    //   2, 6,
    //   3, 7
    // ]);
    //

    /*
        7____6
      3/___2/|
      | 5__|_4
      1/___0/
    */

    const vertices = new Float32Array([
      -0.695, 0.125, -2.04054,
      0.695, 0.125, -2.04054,
      -0.705, 0.691878, -1.845,
      0.705, 0.691878, -1.845,
      -0.665, 0.285, 2.03903,
      0.665, 0.285, 2.03903,
      -0.578869, 0.975, 2.045,
      0.57887, 0.975, 2.045,
    ]);

    // this.setIndex(new THREE.BufferAttribute(indices, 1));
    // this.addAttribute('position', new THREE.BufferAttribute(vertices, 3));

    this.initBox();
  }

  initBox() {
    // // Temp
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
  }

  smooth() {

  }
}
