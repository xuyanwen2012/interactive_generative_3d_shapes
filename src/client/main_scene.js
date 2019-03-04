import {AmbientLight, AxesHelper, GridHelper, Group} from 'three';
import OrbitControls from 'three-orbitcontrols';

import BoundingBox from './bouding_box';

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

    this.add(light, controls);
    this.add(...this.helpers);
  }

  update(timeStamp) {
  }

  hideHelpers() {
    this.helpers.forEach(helper => helper.visible = false);
  }

  showHelpers() {
    this.helpers.forEach(helper => helper.visible = true);
  }
}
