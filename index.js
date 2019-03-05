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

    const getAllFileArgumentsAndRunParallel = (baseArgs, each) => (args) => {
        console.dir(args);
        args.__proto__ = baseArgs;             // provide defaults for optional args 
        let tasks = locateFiles(args);
        if (!args.rebuild) {
            // console.dir(tasks);
            tasks = tasks.filter((file) => {
                return !fs.existsSync(file.output) 
            });
            // console.dir(tasks);
        }
        if (args.limit) {
            console.log(`reducing ${tasks.length} to ${args.limit}`);
            args.limit = Math.min(args.limit, tasks.length);
            tasks = tasks.slice(0, args.limit);
        }
        args.workers = Math.min(args.workers, tasks.length);
        if (args.workers > 1) {
            console.log(`Launching ${args.workers} cluster workers...`);
            require('./src/utils/processing_worker')({
                numWorkers: args.workers,
                job: baseArgs.job,
                tasks: tasks
            });
        } else {
            tasks.forEach((argInstance, i) => {
                console.log(`${i} / ${tasks.length}`);
                argInstance.__proto__ = args;     // forward parent args
                each(argInstance);
            });
        }
    };

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
        parser.addArgument('output', { defaultValue: DEFAULT_DATA_PARAM_DIR }); 
        parser.addArgument([ '-l', '--levels' ], { type: Number, defaultValue: 5 });
        parser.addArgument([ '-J', '--workers' ], { type: Number, defaultValue: 1 });
        parser.addArgument([ '--limit' ], { type: Number, defaultValue: 0 });
        parser.addArgument([ '-r', '--rebuild' ], { action: 'storeTrue' });
    }, getAllFileArgumentsAndRunParallel({
        job: 'process',
        inputExt: OBJ_INPUT_EXT,
        outputExt: DATA_PARAM_EXT,
    }, (args) => {
        enforceFileExists(args.input);
        console.log(`processing ${args.input} => ${args.output}, levels = ${args.levels}`);
        require('./src/process_file')(args);
    }));

    // reconstruct a json parameterization => obj model
    addSubcommand('reconstruct', (parser) => {
        parser.addArgument('input'); 
        parser.addArgument('output', { defaultValue: DEFAULT_OBJ_GEN_DIR });
        parser.addArgument([ '-l', '--levels' ], { type: Number, defaultValue: 5 });
        parser.addArgument([ '-J', '--workers' ], { type: Number, defaultValue: 1 });
        parser.addArgument([ '--limit' ], { type: Number, defaultValue: 0 });
        parser.addArgument([ '-r', '--rebuild' ], { action: 'storeTrue' });
    }, getAllFileArgumentsAndRunParallel({
        job: 'reconstruct',
        inputExt: DATA_PARAM_EXT,
        outputExt: OBJ_GEN_EXT,
    }, (args) => {
        enforceFileExists(args.input);
        console.log(`processing ${args.input} => ${args.output}, levels = ${args.levels}`);
        require('./src/reconstruct_file')(args);
    }));

   const args = parser.parseArgs();
   // console.dir(args);
   commands[args.command](args);
   process.exit();
}
