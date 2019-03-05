'use strict';
const fs = require('fs');
const path = require('path');
const glob = require('glob');
module.exports = locateFiles;

function isDirectory (file) {
    return fs.existsSync(file) ?
        fs.lstatSync(file).isDirectory() :
        file.indexOf('.') < 0;
}

function locateFiles (args) {
    if (typeof(args.input) !== 'string') {
        throw new Error(`Invalid input argument '${args.input}'`);
    }
    // console.log(`${args.output} isDir? ${isDirectory(args.output)}`)
    // console.log(`${args.output} isDir? ${isDirectory(args.output)}`)
    let mapFiles = args.output ? isDirectory(args.output) ?
        (file) => {
            return {
                input: file,
                output: path.join(args.output, 
                    path.basename(file)
                    .replace(args.inputExt, args.outputExt))
            }
        } :
        (file) => { 
             return { 
                input: file, 
                output: file.replace(args.inputExt, args.outputExt) 
            }; 
        } :
        (file) => { return { input: file }; };

    if (args.input.indexOf('*') > -1) {
        return glob.sync(args.input).map(mapFiles);
    }
    if (isDirectory(args.input)) {
        return locateFiles({
            input: path.join(args.input, '**' + args.inputExt),
            output: args.output,
            inputExt: args.inputExt,
            outputExt: args.outputExt,
        });
    }
    return [ mapFiles(args.input) ];
}
