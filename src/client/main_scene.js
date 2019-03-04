import {
  AmbientLight,
  AxesHelper,
  GridHelper,
  Group,
  MeshNormalMaterial
} from 'three';
import OrbitControls from 'three-orbitcontrols';

import BoundingBox from './bouding_box';
import {loader} from '../obj_parser';

export default class MainScene extends Group {
  constructor(camera, renderer) {
    super();

    // Essentials
    const light = new AmbientLight(0x404040, 0.66);
    const controls = new OrbitControls(camera, renderer.domElement);

    // Helpers
    const gridHelper = new GridHelper(10, 10, 0xffffff, 0x555555);
    const axesHelper = new AxesHelper();
    const boxHelper = new BoundingBox();

    /**
     * @type {LineSegments[]}
     */
    this.helpers = [
      gridHelper,
      axesHelper,
      boxHelper,
    ];

    this.loadModel();

    this.add(light, controls);
    this.add(...this.helpers);
  }

  update(timeStamp) {
  }

  loadModel(filename = 'models/1abeca7159db7ed9f200a72c9245aee7.obj') {
    loader.load(filename, (group) => {
      let mesh = group.children[0];

      mesh.material = new MeshNormalMaterial();
      this.add(mesh);
    });
  }

  /**
   * @private
   */
  hideHelpers() {
    this.helpers.forEach(helper => helper.visible = false);
  }

  /**
   * @private
   */
  showHelpers() {
    this.helpers.forEach(helper => helper.visible = true);
  }
}
