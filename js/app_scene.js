class AppScene extends THREE.Scene {

  /**
   * @param args
   */
  constructor(...args) {
    super(...args);

    this.background = new THREE.Color(0x000000);

    camera = new THREE.OrthographicCamera(
      frustumSize * aspect / -2,
      frustumSize * aspect / 2,
      frustumSize / 2,
      frustumSize / -2,
      0.01,
      20,
    );
    camera.position.set(0, 4, 0);
    camera.lookAt(new THREE.Vector3(0, 0, 0));

    this.setupScene();
  }

  /**
   * Initialize scene objects
   */
  setupScene() {
    const light = new THREE.AmbientLight(0x999999);
    const controls = new THREE.OrbitControls(camera, renderer.domElement);

    /**
     * @type {GridHelper}
     */
    this.gridHelper = new THREE.GridHelper(10, 10, 0xffffff, 0x555555);

    controls.addEventListener('change', render);
    controls.target.set(0, 0, 0);
    controls.update();

    this.add(light);
    this.add(camera);
    this.add(this.gridHelper);
  }

  /**
   * @param group {Group}
   */
  addModel(group) {
    this.add(group);

    /**
     * @type {BoundingBox}
     */
    this.box = new BoundingBox(group, 0xffff00);
    this.add(this.box);
  }

  hideHelpers() {
    this.gridHelper.visible = false;
    this.box.visible = false;

    render();
  }

  showHelpers() {
    this.gridHelper.visible = true;
    this.box.visible = true;

    render();
  }
}
