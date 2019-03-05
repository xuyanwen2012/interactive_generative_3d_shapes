#!/usr/bin/env node
'use strict';

const ArgumentParser = require('argparse').ArgumentParser;
const fs = require('fs');

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
    }, (args) => {
        const glob = require('glob');
        const files = { 
            'obj': [],
            'json': [],
            'gen.obj': []
        };
        const extCounts = {};
        glob.sync(args.input).forEach((file) => {
            const ext = file.split('.').slice(1).join('.');
            if (files[ext] !== undefined) {
                files[ext].push(file);
            }
        });
        function listFiles (ext) {
            console.log(`${files[ext].length} ${ext} file(s):\n\t${files[ext].join('\n\t')}`);
        }
        [ 'obj', 'json', 'gen.obj' ].map(listFiles);
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
    }, (args) => {
        enforceFileExists(args.input);
        require('./src/process_file')(args);
    });

    // reconstruct a json parameterization => obj model
    addSubcommand('reconstruct', (parser) => {
        parser.addArgument('input'); 
        parser.addArgument('output');
        parser.addArgument([ '-l', '--levels' ], { type: Number, defaultValue: 5 });
    }, (args) => {
        enforceFileExists(args.input);
        require('./src/reconstruct_file')(args);
    });

   const args = parser.parseArgs();
   console.dir(args);
   commands[args.command](args);
   process.exit();
}
