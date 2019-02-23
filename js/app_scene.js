class AppScene extends THREE.Scene {

  constructor(...args) {
    super(...args);

    this.background = new THREE.Color(0x999999);

    camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 1, 500);
    camera.position.set(3, 3, -8);

    const light = new THREE.AmbientLight(0x999999);
    const grid = new THREE.GridHelper(10, 10, 0xffffff, 0x555555);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    controls.addEventListener('change', render);
    controls.target.set(0, 0, 0);
    controls.update();

    this.add(light);
    this.add(camera);
    this.add(grid);
  }
}
