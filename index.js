#!/usr/bin/env node
'use strict';
const fs = require('fs');
const ArgumentParser = require('argparse').ArgumentParser;
const locateFiles = require('./src/utils/locate_files');

const OBJ_INPUT_EXT = '.obj';
const DATA_PARAM_EXT = '.json';
const OBJ_GEN_EXT = '.gen.obj';
const DEFAULT_OBJ_MODEL_DIR = 'models';
const DEFAULT_DATA_PARAM_DIR = 'output';
const DEFAULT_OBJ_GEN_DIR = 'output';

main();

function main () {
    const parser = new ArgumentParser({
        version: '0.0.1',
        addHelp: true,
        description: "tbd (shape-gen tools)"
    });
    const subparsers = parser.addSubparsers({ title: 'command', dest: 'command' })
    const commands = {};
    function addSubcommand (name, addArgs, callback) {
        const parser = subparsers.addParser(name, { addHelp: true });
        addArgs(parser);
        commands[name] = callback;
    }
    function enforceFileExists (file) {
        if (!fs.existsSync(file)) {
            console.warn(`file does not exist: '${file}'`);
            process.exit();
        }
    }
    // Commands

    // Get info on model / etc data
    // Currently being used to test glob()
    addSubcommand('info', (parser) => {
        parser.addArgument('input');
        parser.addArgument('--iext', { defaultValue: '.obj' });
        parser.addArgument('--oext', { defaultValue: '.json' });
    }, (args) => {
        const files = locateFiles({
            input: args.input, output: "output",
            inputExt: args.iext, outputExt: args.oext,
        });
        console.dir(files);
    });

    // view a model, json parameterization, or directory
    addSubcommand('view', (parser) => {
        parser.addArgument('input');
    }, (args) => {
        enforceFileExists(args.input);
        console.warn("TBD: view");
    });

    // process an obj model => json parameterization
    addSubcommand('process', (parser) => {
        parser.addArgument('input'); 
        parser.addArgument('output'); 
        parser.addArgument([ '-l', '--levels' ], { type: Number, defaultValue: 5 });
        parser.addArgument([ '--limit' ], { type: Number, defaultValue: 0 });
        parser.addArgument([ '-r', '--rebuild' ], { action: 'storeTrue' });
    }, (args) => {
        args.inputExt = OBJ_INPUT_EXT;
        args.outputExt = DATA_PARAM_EXT;
        args.output = args.output || DEFAULT_DATA_PARAM_DIR;

        let fileArgs = locateFiles(args);
        if (!args.rebuild) {
            console.dir(fileArgs);
            fileArgs = fileArgs.filter((file) => {
                return !fs.existsSync(file.output) 
            });
            console.dir(fileArgs);
        }
        if (args.limit) {
            console.log(`reducing ${fileArgs.length} to ${args.limit}`);
            args.limit = Math.min(args.limit, fileArgs.length);
            fileArgs = fileArgs.slice(0, args.limit);
        }
        fileArgs.forEach((fargs) => {
            fargs.__proto__ = args;
            enforceFileExists(fargs.input);
            console.log(`processing ${fargs.input} => ${fargs.output}, levels = ${fargs.levels}`);
            require('./src/process_file')(fargs);
        })
    });

    // reconstruct a json parameterization => obj model
    addSubcommand('reconstruct', (parser) => {
        parser.addArgument('input'); 
        parser.addArgument('output');
        parser.addArgument([ '-l', '--levels' ], { type: Number, defaultValue: 5 });
        parser.addArgument([ '--limit' ], { type: Number, defaultValue: 0 });
        parser.addArgument([ '-r', '--rebuild' ], { action: 'storeTrue' });
    }, (args) => {
        args.inputExt = DATA_PARAM_EXT;
        args.outputExt = OBJ_GEN_EXT;
        args.output = args.output || DEFAULT_OBJ_GEN_DIR;

        let fileArgs = locateFiles(args);
        if (!args.rebuild) {
            console.dir(fileArgs);
            fileArgs = fileArgs.filter((file) => {
                return !fs.existsSync(file.output) 
            });
            console.dir(fileArgs);
        }
        if (args.limit) {
            console.log(`reducing ${fileArgs.length} to ${args.limit}`);
            args.limit = Math.min(args.limit, fileArgs.length);
            fileArgs = fileArgs.slice(0, args.limit);
        }
        console.dir(fileArgs);
        fileArgs.forEach((fargs) => {
            fargs.__proto__ = args;
            enforceFileExists(fargs.input);
            console.log(`processing ${fargs.input} => ${fargs.output}, levels = ${fargs.levels}`);
            require('./src/reconstruct_file')(fargs);
        });
    });

   const args = parser.parseArgs();
   // console.dir(args);
   commands[args.command](args);
   process.exit();
}
