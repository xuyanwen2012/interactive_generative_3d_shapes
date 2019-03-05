const {
  Worker,
  isMainThread,
  parentPort,
  workerData
} = require('worker_threads');

if (isMainThread) {
  module.exports = (args) => {
    workers = [];
    for (i = 1; i < args.numWorkers; ++i) {
      console.log(`starting worker ${i}`);
      let worker = new Worker(__filename, {
        numWorkers: args.numWorkers,
        id: i,
        job: args.job,
        tasks: args.tasks
      });
      workers.push(worker);
      // worker.on('message', resolve);
      // worker.on('error', reject);
      worker.on('exit', (code) => {
        if (code !== 0) console.log(new Error(`Worker stopped with exit code ${code}`));
      });
    }
    runTasks({
      id: 0, numWorkers: args.numWorkers, job: args.job, tasks: args.tasks
    });
  }
} else {
  console.log(`Launched worker ${worker.id}`);
  runTasks(workerData);
}
function runTasks (args) {
  const run = {
    process: () => require('../process_file'),
    reconstruct: () => require('../reconstruct_file')
  }[args.job]();
  args.tasks.forEach((task, i) => {
    if ((i % args.numWorkers) == args.id) {
      console.log(`Worker ${args.id} running task ${i}`);
      console.dir(task);
      // run(task);
    }
  });

}
