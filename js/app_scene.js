class AppScene extends THREE.Scene {

  /**
   * @param args
   */
  constructor(...args) {
    super(...args);

    this.background = new THREE.Color(0x000000);

    this.setupPerspectiveCamera();
    // this.setupOrthoCamera();
    this.setupScene();
  }

  setupPerspectiveCamera() {
    camera = new THREE.PerspectiveCamera(35, aspect, 0.01, 20);
    camera.position.set(0, 4, 0);
    camera.lookAt(new THREE.Vector3(0, 0, 0));
  }

  setupOrthoCamera() {
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

    /**
     * @type {AxesHelper}
     */
    this.axesHelper = new THREE.AxesHelper();

    controls.addEventListener('change', render);
    controls.target.set(0, 0, 0);
    controls.update();

    this.add(light);
    this.add(camera);
    this.add(this.gridHelper);

    this.add(this.axesHelper);
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

    /**
     * @type {VertexNormalsHelper}
     */
    this.vertexNormal = new THREE.VertexNormalsHelper(group, 2, 0xffff00, 1);

    this.add(this.box);
    this.add(this.vertexNormal);
  }

  hideHelpers() {
    this.gridHelper.visible = false;
    this.box.visible = false;
    this.vertexNormal.visible = false;
    this.axesHelper.visible = false;

    render();
  }

  showHelpers() {
    this.gridHelper.visible = true;
    this.box.visible = true;
    this.vertexNormal.visible = true;
    this.axesHelper.visible = true;

    render();
  }
}
