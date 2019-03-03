'use strict';

// Global variables
let camera, scene, renderer;
let raycaster = new THREE.Raycaster();
let mouse = new THREE.Vector2();

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
    hide: () => scene.hideHelpers(),
    shrink: () => doShrinkWrap()
  };

  gui.add(obj, 'show');
  gui.add(obj, 'hide');
  gui.add(obj, 'shrink');
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
  window.addEventListener('mousemove', onMouseMove, false);
}

let loadedModel;

function doShrinkWrap() {
  const jjj = new ShrinkWrapper(loadedModel);
  jjj.modify(2);
  console.log(jjj.output);
}

/**
 * @param path {string}
 * @param pos {THREE.Vector3}
 */
function loadModel(path, pos = new THREE.Vector3(0, 0, 0)) {

  loader.load(path, (group) => {
    let mesh = group.children[0];

    // console.log(mesh.geometry.attributes);

    mesh.material = new NaiveDepthGenerator().material;
    mesh.position.set(pos.x, pos.y, pos.z);
    // scene.addModel(group);

    depth_map_mesh = mesh;
    render();
    // screenShot();
    loadedModel = mesh;

  });
}

function render() {
  renderer.clear();
  renderer.render(scene, camera);

  if (!depth_map_mesh) {
    return;
  }

  raycaster.setFromCamera(mouse, camera);
  // See if the ray from the camera into the world hits one of our meshes
  const intersects = raycaster.intersectObject(depth_map_mesh);
  // Toggle rotation bool for meshes that we clicked
  if (intersects.length > 0) {
    helper.position.set(0, 0, 0);
    helper.lookAt(intersects[0].face.normal);
    helper.position.copy(intersects[0].point);
  }
}

// Temp
let helper;
createRaycast();

function createRaycast() {
  const geometry = new THREE.ConeBufferGeometry(0.1, 0.25, 3);
  geometry.translate(0, 0, 0);
  geometry.rotateX(Math.PI / 2);
  helper = new THREE.Mesh(geometry, new THREE.MeshNormalMaterial());
  scene.add(helper);
}

function onMouseMove(event) {
  event.preventDefault();
  mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
  mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
}

function onWindowResize() {
  renderer.setSize(canvasWidth, canvasHeight);

  camera.aspect = aspect;
  // updateOrthoCamera();

  render();
}

function updateOrthoCamera() {
  camera.left = -frustumSize * aspect / 2;
  camera.right = frustumSize * aspect / 2;
  camera.top = frustumSize / 2;
  camera.bottom = -frustumSize / 2;
  camera.updateProjectionMatrix();
  camera.lookAt(new THREE.Vector3(0, 0, 0));
}

/**
 * temporary
 */
function screenShot() {
  scene.hideHelpers();

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

  createDepthSurface(pixels);
  // createDepthSurface(pixels)
}


let depth_map_mesh;

/**
 * @param pixels {Uint8Array}
 */
function createDepthSurface(pixels) {
  // console.log(pixels); // Uint8Array

  const geometry = new THREE.Geometry();
  const width = 600;
  const depth = 200;
  for (let x = 0; x < depth; x++) {
    for (let z = 0; z < width; z++) {
      // let red = pixels[z * (w * 4) + x * 4];
      let yValue = pixels[z * (depth * 4) + x * 4] / 128.0;
      // let yValue = pixels[z * 4 + (depth * x * 4)] / 128.0;
      let vertex = new THREE.Vector3(x / 100.0, yValue, z / 100.0);
      geometry.vertices.push(vertex);
    }
  }

  // we create a rectangle between four vertices, and we do
  // that as two triangles.
  for (let z = 0; z < depth - 1; z++) {
    for (let x = 0; x < width - 1; x++) {
      // we need to point to the position in the array
      // a - - b
      // |  x  |
      // c - - d
      const a = x + z * width;
      const b = (x + 1) + (z * width);
      const c = x + ((z + 1) * width);
      const d = (x + 1) + ((z + 1) * width);
      const face1 = new THREE.Face3(a, b, d);
      const face2 = new THREE.Face3(d, c, a);
      face1.color = new THREE.Color(1, 1, 1);
      face2.color = new THREE.Color(1, 1, 1);
      geometry.faces.push(face1);
      geometry.faces.push(face2);
    }
  }

  geometry.computeVertexNormals(true);
  geometry.computeFaceNormals();
  geometry.computeBoundingBox();

  const material = new THREE.MeshNormalMaterial();
  const plane = new THREE.Mesh(geometry, material);

  plane.rotation.y = Math.PI;
  plane.position.set(3, 0, 3);
  // const helper = new THREE.VertexNormalsHelper(plane, 2, 0x00ff00, 1);
  // scene.add(helper);
  scene.add(plane);
  // depth_map_mesh = plane;
}

