import {
  AmbientLight,
  AxesHelper,
  DoubleSide,
  EdgesGeometry,
  FileLoader,
  GridHelper,
  Group,
  LineBasicMaterial,
  LineSegments,
  MeshNormalMaterial,
} from 'three';
import OrbitControls from 'three-orbitcontrols';
import * as dat from 'dat.gui';

import ShrinkWrapper from '../core/shrink_wrapper';
import BoundingBox from './bouding_box';

const parser = require('../core/obj_parser');

export default class MainScene extends Group {
  constructor(camera, renderer) {
    super();

    // Setup Essentials
    const light = new AmbientLight(0x404040, 0.66);
    const controls = new OrbitControls(camera, renderer.domElement);

    /**
     * @type {LineSegments[]}
     */
    this.helpers = [
      new GridHelper(10, 10, 0xffffff, 0x555555),
      new AxesHelper(),
      new BoundingBox(),
    ];

    this.edgeHelper = null;

    /**
     * @type {Mesh}
     */
    this.carMesh = null;

    /**
     * @type {string}
     */
    this.objText = '';

    /**
     * @type {Mesh}
     */
    this.wrapper = null;

    this.loadModel();
    this.initializeGUI();

    this.add(light);
    this.add(...this.helpers);
  }

  initializeGUI() {
    const gui = new dat.GUI();

    const helpers = {
      gridHelpers: true,
      edgeHelper: false,
      carModel: true,
    };

    this.options = {
      subdivisions: 5,
      shrink: () => this.doAlgorithm(this.options.subdivisions),
    };

    gui.add(helpers, 'gridHelpers').onChange(newValue => this.updateHelpers(newValue));
    gui.add(helpers, 'edgeHelper').onChange(newValue => this.updateEdgeHelper(newValue));
    gui.add(helpers, 'carModel').onChange(newValue => this.updateModel(newValue));

    gui.add(this.options, 'subdivisions').min(0).max(5).step(1);
    gui.add(this.options, 'shrink');
  }

  update(timeStamp) {
  }

  loadModel(filename = 'models/1b94aad142e6c2b8af9f38a1ee687286.obj') {
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
    this.remove(this.wrapper);
    const wrapper = new ShrinkWrapper(this.carMesh, this.objText);

    this.add(wrapper.mesh);
    wrapper.modify(levels);

    this.wrapper = wrapper.mesh;
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

  /**
   * @private
   */
  updateEdgeHelper(newValue) {
    if (newValue && this.wrapper) {
      this.remove(this.edgeHelper);
      const edges = new EdgesGeometry(this.wrapper.geometry, 0); // show all edge
      this.edgeHelper = new LineSegments(edges, new LineBasicMaterial({color: 0x0000ff}));
      this.add(this.edgeHelper);
    }

    if (this.edgeHelper) {
      this.edgeHelper.visible = newValue;
    }
  }
}
