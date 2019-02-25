'use strict';

// Global variables
let camera, scene, renderer;

// Some constants
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

    screenShot();
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

/**
 * temporary
 */
function screenShot() {
  const gl = renderer.domElement.getContext('webgl');

  const unit = 100;
  const width = 2 * unit;
  const height = 6 * unit;
  const pixels = new Uint8Array(width * height * 4);

  gl.readPixels(
    200,
    0,
    width,
    height,
    gl.RGBA,
    gl.UNSIGNED_BYTE,
    pixels,
  );

  createHeightSurface(pixels);
}

/**
 * @param pixels {Uint8Array}
 */
function createHeightSurface(pixels) {
  console.log(pixels); // Uint8Array

  const numPixels = pixels.length / 4;

  // const skip = numPixels


  // const geometry = new THREE.PlaneGeometry(2, 6, 200, 600);
  const geometry = new THREE.BufferGeometry();
  const w = 200;
  const h = 600;
  let arr = [];
  for (let x = 0; x < w; x++) {
    for (let z = 0; z < h; z++) {
      let red = pixels[z * (w * 4) + x * 4];
      arr.push(x / 100.0, red / 128.0, z / 100.0);
    }
  }
  let vertices = new Float32Array(arr);
  geometry.addAttribute('position', new THREE.BufferAttribute(vertices, 3));
  geometry.computeBoundingBox();

  const material = new THREE.MeshBasicMaterial({
    color: 0xffff00,
    wireframe: true,
  });
  const plane = new THREE.Mesh(geometry, material);

  plane.position.set(-3, 1, 0);

  console.log(geometry);
  scene.add(plane);
}
