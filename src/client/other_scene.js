import {
  AmbientLight,
  BoxGeometry,
  EdgesGeometry,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshNormalMaterial,
  VertexNormalsHelper,
} from 'three';
import OrbitControls from 'three-orbitcontrols';

const parser = require('../core/obj_parser');

export default class OtherScene extends Group {
  constructor(camera, renderer) {
    super();

    // Setup Essentials
    const light = new AmbientLight(0x404040, 0.66);
    const controls = new OrbitControls(camera, renderer.domElement);


    const geometry = new BoxGeometry(1, 1, 1);
    // geometry.computeVertexNormals();

    const material = new MeshNormalMaterial();
    const cube = new Mesh(geometry, material);
    const helper = new VertexNormalsHelper(cube, 2, 0x00ff00, 1);


    this.add(cube);
    this.add(helper);
    this.add(light);
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
