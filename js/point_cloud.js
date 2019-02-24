const pointSize = 0.05;

/**
 * @param color {THREE.Color}
 * @param geo {THREE.BufferGeometry}
 */
function generatePointCloudFromGeo(color, geo) {
  const geometry = new THREE.BufferGeometry();
  const numPoints = geo.attributes.position.count;
  const positions = new Float32Array(numPoints * 3);
  const colors = new Float32Array(numPoints * 3);

  for (let i = 0; i < numPoints; i++) {
    positions[3 * i] = geo.attributes.position.array[3 * i];
    positions[3 * i + 1] = geo.attributes.position.array[3 * i + 1];
    positions[3 * i + 2] = geo.attributes.position.array[3 * i + 2];

    const y = positions[3 * i + 1];
    const intensity = (y + 0.1) * 0.75;
    colors[3 * i] = color.r * intensity;
    colors[3 * i + 1] = color.g * intensity;
    colors[3 * i + 2] = color.b * intensity;
  }

  console.log(numPoints);

  geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingBox();

  const material = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: THREE.VertexColors
  });

  return new THREE.Points(geometry, material);
}
