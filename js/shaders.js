/**
 * @constant
 * @type {string}
 */
const vertex_shader = `
  precision mediump float;
  precision mediump int;
  
	attribute vec3 position;
	attribute vec4 color;
	
	varying vec3 vPosition;
	varying vec4 vColor;
	
	void main()	{
		vPosition = position;
		vColor = color;
		gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
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
	varying vec4 vColor;
	
	void main()	{
		vec4 color = vec4( vColor );
		gl_FragColor = color;
	}
`;
