'use strict';

const fs = require('fs');
const path = require('path');
const Reconstructor = require('./reconstructor');

const data = JSON.parse(fs.readFileSync(path.join(__dirname, '../output/1abeca7159db7ed9f200a72c9245aee7.json'), 'utf8'));

let reconstructor = new Reconstructor(data);

reconstructor.modify();
