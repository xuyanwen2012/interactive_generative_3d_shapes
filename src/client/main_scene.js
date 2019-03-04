import {
  AmbientLight,
  AxesHelper,
  DoubleSide,
  FileLoader,
  GridHelper,
  Group,
  MeshNormalMaterial,
} from 'three';
import OrbitControls from 'three-orbitcontrols';
import * as dat from 'dat.gui';

import ShrinkWrapper from '../shrink_wrapper';
import BoundingBox from './bouding_box';

const parser = require('../obj_parser');

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

    /**
     * @type {Mesh}
     */
    this.carMesh = null;

    /**
     * @type {string}
     */
    this.objText = null;

    this.loadModel();
    this.initializeGUI();

    this.add(light);
    this.add(...this.helpers);
  }

  initializeGUI() {
    const gui = new dat.GUI();

    const helpers = {
      gridHelpers: true,
      carModel: true,
    };

    this.options = {
      subdivisions: 5,
      shrink: () => this.doAlgorithm(this.subdivisions),
    };

    gui.add(helpers, 'gridHelpers').onChange(newValue => this.updateHelpers(newValue));
    gui.add(helpers, 'carModel').onChange(newValue => this.updateModel(newValue));

    gui.add(this.options, 'subdivisions').min(0).max(5).step(1);
    gui.add(this.options, 'shrink');
  }

  update(timeStamp) {
  }

  loadModel(filename = 'models/1abeca7159db7ed9f200a72c9245aee7.obj') {
    let fileLoader = new FileLoader();

    fileLoader.load(filename, (text) => {
      const mesh = parser.parseModel(text);
      this.add(mesh);
      mesh.material = new MeshNormalMaterial({
        side: DoubleSide,
      });
      this.carMesh = mesh;
      this.objText = text;
    });
  }

  /**
   * @private
   */
  doAlgorithm(levels) {
    const wrapper = new ShrinkWrapper(this.carMesh, this.objText);

    this.add(wrapper.mesh);

    wrapper.modify(levels);
  }

  /**
   * @private
   */
  updateHelpers(newValue) {
    this.helpers.forEach(helper => helper.visible = newValue);
  }

  /**
   * @private
   */
  updateModel(newValue) {
    this.carMesh.translateX(newValue ? 3 : -3);
  }
}
