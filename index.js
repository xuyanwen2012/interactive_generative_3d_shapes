#!/usr/bin/env node
'use strict';

const meow = require('meow');
const preprocess = require('./src/main');

const cli = meow(`
	Usage
	  $ preprocess <input>

	Examples
	  $ preprocess 1abeca7159db7ed9f200a72c9245aee7.obj
	  🌈 unicorns 🌈
`);

preprocess(cli.input[0]);
