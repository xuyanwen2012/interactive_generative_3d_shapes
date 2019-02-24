'use strict';

let camera, scene, renderer;

/**
 * @type {THREE.OBJLoader}
 */
const loader = new THREE.OBJLoader();

init();

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
  // function readJSON(path) {
  //   const xhr = new XMLHttpRequest();
  //   xhr.open('GET', path, true);
  //   xhr.responseType = 'json';
  //   xhr.onload = (e) => {
  //     if (e.status === 200) {
  //       const file = new File([this.response], '_index.json');
  //       const fileReader = new FileReader();
  //       fileReader.addEventListener('load', () => {
  //         console.log(file);
  //       });
  //       fileReader.readAsText(file);
  //     }
  //   };
  //   xhr.send();
  // }
  // readJSON('./models/_index.json');
  loadModel('./models/1abeca7159db7ed9f200a72c9245aee7.obj');
  // loadModel('./models/1acfbda4ce0ec524bedced414fad522f.obj', new THREE.Vector3(2, 0, 0));
  // loadModel('./models/1ae530f49a914595b491214a0cc2380.obj', new THREE.Vector3(4, 0, 0));
  // loadModel('./models/1aef0af3cdafb118c6a40bdf315062da.obj', new THREE.Vector3(-2, 0, 0));
  // loadModel('./models/1b5b5a43e0281030b96212c8f6cd06e.obj', new THREE.Vector3(-4, 0, 0));

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

    // mesh.material = new THREE.MeshNormalMaterial();
    // mesh.material.flatShading = true;
    // mesh.material.needsUpdate = true;

    mesh.material = new NaiveDepthGenerator().material;

    mesh.position.set(pos.x, pos.y, pos.z);
    scene.add(group);

    let points = generatePointCloudFromGeo(new THREE.Color(1, 0, 0), mesh.geometry);
    scene.add(points);

    points.position.set(pos.x + 3, pos.y, pos.z);

    let box = new BoundingBox(group, 0xffff00);
    scene.add(box);

    render();
  });
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

