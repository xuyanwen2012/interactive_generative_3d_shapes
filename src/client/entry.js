import {PerspectiveCamera, Scene, Vector3, WebGLRenderer} from 'three';
import MainScene from './main_scene';

const renderer = new WebGLRenderer({antialias: true});
const camera = new PerspectiveCamera();
const scene = new Scene();
const mainScene = new MainScene(camera, renderer);

scene.add(mainScene);

// camera
camera.position.set(6, 3, -10);
camera.lookAt(new Vector3(0, 0, 0));

// renderer
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x7ec0ee, 1);

// render loop
const onAnimationFrameHandler = (timeStamp) => {
  renderer.render(scene, camera);
  mainScene.update && mainScene.update(timeStamp);
  window.requestAnimationFrame(onAnimationFrameHandler);
};
window.requestAnimationFrame(onAnimationFrameHandler);

// resize
const windowResizeHandler = () => {
  const {innerHeight, innerWidth} = window;
  renderer.setSize(innerWidth, innerHeight);
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
};

windowResizeHandler();
window.addEventListener('resize', windowResizeHandler);

// dom
document.body.style.margin = '0';
document.body.appendChild(renderer.domElement);
