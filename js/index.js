'use strict';

// Global variables
let camera, scene, renderer;

const frustumSize = 6;
const canvasWidth = 600;
const canvasHeight = 600;
const aspect = 1.0;

/**
 * @type {THREE.OBJLoader}
 */
const loader = new THREE.OBJLoader();

init();

function initGUI() {
  const gui = new dat.GUI();

  let obj = {
    show: () => scene.showHelpers(),
    hide: () => scene.hideHelpers()
  };

  gui.add(obj, 'show');
  gui.add(obj, 'hide');
}

function init() {
  initGUI();

  // Renderer
  renderer = new THREE.WebGLRenderer({
    antialias: true,
    preserveDrawingBuffer: true
  });

  renderer.setPixelRatio(1.0);
  renderer.setSize(canvasWidth, canvasHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new AppScene();

  // OBJ
  loadModel('./models/1abeca7159db7ed9f200a72c9245aee7.obj');

  window.addEventListener('resize', onWindowResize, false);
  document.addEventListener('mousemove', onDocumentMouseMove, false);
}


/**
 * @param path {string}
 * @param pos {THREE.Vector3}
 */
function loadModel(path, pos = new THREE.Vector3(0, 0, 0)) {

  loader.load(path, (group) => {
    let mesh = group.children[0];

    console.log(mesh.geometry.attributes);

    mesh.material = new NaiveDepthGenerator().material;
    mesh.position.set(pos.x, pos.y, pos.z);
    scene.addModel(group);


    let points = generatePointCloudFromGeo(new THREE.Color(1, 0, 0), mesh.geometry);
    scene.add(points);

    points.position.set(pos.x + 3, pos.y, pos.z);

    render();
  });
}

function render() {
  renderer.clear();
  renderer.render(scene, camera);
}

function onDocumentMouseMove(event) {
  event.preventDefault();
}

function onWindowResize() {
  renderer.setSize(canvasWidth, canvasHeight);

  camera.aspect = aspect;

  camera.left = -frustumSize * aspect / 2;
  camera.right = frustumSize * aspect / 2;
  camera.top = frustumSize / 2;
  camera.bottom = -frustumSize / 2;
  camera.updateProjectionMatrix();

  camera.lookAt(new THREE.Vector3(0, 0, 0));

  render();
}

function screenShot() {
  const gl = renderer.domElement.getContext('webgl');
  const width = canvasWidth;
  const height = canvasHeight;
  const pixels = new Uint8Array(width * height * 4);

  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);

  console.log(pixels); // Uint8Array
}

