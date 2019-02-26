class NaiveBox extends THREE.Geometry {
  constructor(...args) {
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

    this.colors.push(
      new THREE.Color(0xFF0000),
      new THREE.Color(0x00FF00),
      new THREE.Color(0x0000FF),
      new THREE.Color(0xFF00FF),
      new THREE.Color(0x00FFFF),
      new THREE.Color(0xFFFF00),
      new THREE.Color(0xFACADE),
      new THREE.Color(0xFFFFFF),
    );

    // this.faces.push(new THREE.Face3(0, 1, 2));
    // this.faces.push(new THREE.Face3(0, 1, 2));
    // this.faces.push(new THREE.Face3(0, 1, 2));
    // this.faces.push(new THREE.Face3(0, 1, 2));
    // this.faces.push(new THREE.Face3(0, 1, 2));
    // this.faces.push(new THREE.Face3(0, 1, 2));


  }

  smooth() {

  }
}
