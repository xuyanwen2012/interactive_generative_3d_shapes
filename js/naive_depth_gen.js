class NaiveDepthGenerator {
  constructor() {

    /**
     * @type {THREE.ShaderMaterial}
     */
    this.material = new THREE.ShaderMaterial({
      uniforms: {},
      vertexShader: vertex_shader,
      fragmentShader: fragment_shader,
      // side: THREE.DoubleSide,
      transparent: true,
    });
  }
}
