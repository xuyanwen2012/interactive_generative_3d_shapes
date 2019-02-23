const pointSize = 0.05;
const width = 80;
const length = 160;

function generatePointCloudGeometry(color, width, length) {
  const geometry = new THREE.BufferGeometry();
  const numPoints = width * length;
  const positions = new Float32Array(numPoints * 3);
  const colors = new Float32Array(numPoints * 3);

  let k = 0;
  for (let i = 0; i < width; i++) {
    for (let j = 0; j < length; j++) {
      const u = i / width;
      const v = j / length;
      const x = u - 0.5;
      const y = (Math.cos(u * Math.PI * 4) + Math.sin(v * Math.PI * 8)) / 20;
      const z = v - 0.5;
      positions[3 * k] = x;
      positions[3 * k + 1] = y;
      positions[3 * k + 2] = z;
      const intensity = (y + 0.1) * 5;
      colors[3 * k] = color.r * intensity;
      colors[3 * k + 1] = color.g * intensity;
      colors[3 * k + 2] = color.b * intensity;
      k++;
    }
  }

  geometry.addAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.addAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingBox();
  return geometry;
}

function generatePointcloud(color, width, length) {
  const geometry = generatePointCloudGeometry(color, width, length);
  const material = new THREE.PointsMaterial({
    size: pointSize,
    vertexColors: THREE.VertexColors
  });
  return new THREE.Points(geometry, material);
}
