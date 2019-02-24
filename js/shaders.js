/**
 * @constant
 * @type {string}
 */
const vertex_shader = `
  precision mediump float;
  precision mediump int;
  
	varying vec3 vPosition;
	
	void main()	{
		vPosition = position;
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
	
	void main()	{
	  float depth = 1.0 - (2.0 - vPosition.y) / 2.0;
	
		vec4 color = vec4(vPosition.x / 2.0, depth, vPosition.z / 6.0, 1.0 );
		gl_FragColor = color;
	}
`;
