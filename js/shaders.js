/**
 * @constant
 * @type {string}
 */
const vertex_shader = `
  precision mediump float;
  precision mediump int;
  
	varying vec3 vPosition;
	varying vec3 vNormal;
	
	void main()	{
		vPosition = position;
		vNormal = normal;
		gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
	}
`;

/**
 * @constant
 * @type {string}
 */
const fragment_shader = `
  precision mediump float;
	precision mediump int;

	varying vec3 vPosition;
	varying vec3 vNormal;

	void main()	{
	  // float depth = 1.0 - (2.0 - vPosition.y) / 2.0;
	  float depth = 1.0 - (vPosition.z + 3.0) / 6.0;
	
		vec4 color = vec4(depth, depth, depth, 1.0 );
		// vec4 color = vec4(vNormal, 1.0 );
		// vec4 color = vec4(vPosition.x / 2.0, depth, vPosition.z / 6.0, 1.0 );
		gl_FragColor = color;
	}
`;
