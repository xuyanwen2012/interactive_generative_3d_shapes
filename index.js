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
        require('./src/main')(args);
    });

    // reconstruct a json parameterization => obj model
    addSubcommand('reconstruct', (parser) => {
        parser.addArgument('input'); 
        parser.addArgument('output');
    }, (args) => {
        enforceFileExists(args.input);
        console.warn("TBD: reconstruct");
    });

   const args = parser.parseArgs();
   console.dir(args);
   commands[args.command](args);
   process.exit();
}
