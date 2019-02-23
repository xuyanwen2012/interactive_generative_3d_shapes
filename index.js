'use strict';

let camera, scene, renderer;

const loader = new THREE.OBJLoader();

init();


class Demo {

}


function init() {
  // Scene
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x999999);
  scene.add(new THREE.AmbientLight(0x999999));

  // Camera
  camera = new THREE.PerspectiveCamera(35, window.innerWidth / window.innerHeight, 1, 500);

  // camera.up.set(0, 0, 1); // Z is up for objects intended to be 3D printed.
  camera.position.set(3, 3, -8);
  camera.add(new THREE.PointLight(0xffffff, 0.8));
  scene.add(camera);

  // Grid Helper
  const grid = new THREE.GridHelper(10, 10, 0xffffff, 0x555555);
  // grid.rotateOnAxis(new THREE.Vector3(1, 0, 0), 90 * (Math.PI / 180));
  scene.add(grid);

  // Renderer
  renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(window.innerWidth, window.innerHeight);
  document.body.appendChild(renderer.domElement);

  // OBJ
  loader.load('./models/1abeca7159db7ed9f200a72c9245aee7.obj', (obj) => {
    scene.add(obj);
    obj.material = new THREE.MeshNormalMaterial();
    render();
  });

  // Control
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.addEventListener('change', render);
  controls.target.set(0, 0, 0);
  controls.update();

  window.addEventListener('resize', onWindowResize, false);
  document.addEventListener('mousemove', onDocumentMouseMove, false);
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

