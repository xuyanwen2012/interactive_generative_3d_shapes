import {
  AmbientLight,
  DoubleSide,
  EdgesGeometry,
  FileLoader,
  Group,
  LineBasicMaterial,
  LineSegments,
  MeshNormalMaterial,
  Vector3,
} from 'three';
import OrbitControls from 'three-orbitcontrols';

import ShrinkWrapper from '../core/shrink_wrapper';

const parser = require('../core/obj_parser');

export default class SubdivisionScene extends Group {
  constructor(camera, renderer) {
    super();

    // Setup Essentials
    const light = new AmbientLight(0x404040, 0.66);
    const controls = new OrbitControls(camera, renderer.domElement);

    /**
     * @type {LineSegments[]}
     */
    this.helpers = [
      // new GridHelper(20, 20, 0xffffff, 0x555555),
      // new AxesHelper(),
      // new BoundingBox(),
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
     * @type {Array.<Mesh>}
     */
    this.wrappers = [];

    this.loadModel();

    this.add(light);
    this.add(...this.helpers);
  }

  loadModel(filename = 'models/1b94aad142e6c2b8af9f38a1ee687286.obj', position = new Vector3(0, 0, 0)) {
    let fileLoader = new FileLoader();

    fileLoader.load(filename, (text) => {
      const mesh = parser.parseModel(text);
      // this.add(mesh);
      mesh.material = new MeshNormalMaterial({
        side: DoubleSide,
      });
      this.carMesh = mesh;
      this.objText = text;

      mesh.position.set(position.x, position.y, position.z);

      const numLevels = 5;

      for (let i = 0; i <= numLevels; i++) {
        this.doAlgorithm(i);
      }

      const unit = 4;
      const offset = numLevels * unit / 2;
      for (let i = 0; i < 3; i++) {
        this.wrappers[i].position.set(0, 0, i * unit - offset);
        this.wrappers[i].rotation.y = Math.PI / 3;
        this.createEdgeHelper(this.wrappers[i])
      }

      for (let i = 3; i < 6; i++) {
        this.wrappers[i].position.set(8, 0, (i - 3) * unit - offset);
        this.wrappers[i].rotation.y = Math.PI / 3;
        this.createEdgeHelper(this.wrappers[i])
      }

    });
  }


  /**
   * @private
   */
  doAlgorithm(levels) {
    const wrapper = new ShrinkWrapper(this.carMesh, this.objText);

    this.add(wrapper.mesh);
    wrapper.modify(levels);

    this.wrappers.push(wrapper.mesh)
  }

  createEdgeHelper(mesh) {
    const edges = new EdgesGeometry(mesh.geometry, 0); // show all edge
    const edgeHelper = new LineSegments(edges, new LineBasicMaterial({
      linewidth: 0.025,
      color: 0x0000ff
    }));
    edgeHelper.position.copy(mesh.position);
    edgeHelper.rotation.copy(mesh.rotation);
    this.add(edgeHelper);
  }
}
