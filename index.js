'use strict';

let camera, scene, renderer;

const loader = new THREE.OBJLoader();

init();

/**
 * @param object {THREE.Object3D}
 */
function OnObjLoad(object) {
  let mesh = object.children[0];
  mesh.material = new THREE.MeshNormalMaterial();
  mesh.material.flatShading = true;
  scene.add(object);

  let geo = mesh.geometry;

  let vnh = new THREE.VertexNormalsHelper(mesh, 0.05);
  // scene.add(vnh);

  let box = new THREE.BoxHelper(object, 0xffff00);
  scene.add(box);

  render();
}

function initGUI() {
  const gui = new dat.GUI();
}

function init() {
  initGUI();

  // Renderer
  renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // Scene
  scene = new AppScene();

  // OBJ
  loadModel('./models/1b5b5a43e0281030b96212c8f6cd06e.obj');

  window.addEventListener('resize', onWindowResize, false);
  document.addEventListener('mousemove', onDocumentMouseMove, false);
}

function loadModel(path, position) {
  let model = loader.load(path, OnObjLoad);

  console.log(model);
}

function render() {
  renderer.render(scene, camera);
}

function onDocumentMouseMove(event) {
  event.preventDefault();
}

function onWindowResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();

  renderer.setSize(window.innerWidth, window.innerHeight);

  render();
}

