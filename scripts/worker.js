const {Worker, isMainThread, workerData} = require('worker_threads');

let currentVal = 0;
let intervals = [100, 1000, 500];

function counter(id, i) {
  console.log('[', id, ']', i);
  return i;
}

if (isMainThread) {
  console.log('this is the main thread');
  for (let i = 0; i < 2; i++) {
    let w = new Worker(__filename, {workerData: i});
  }

  setInterval((a) => currentVal = counter(a, currentVal + 1), intervals[2], "MainThread");
} else {

  console.log('this isn\'t');

  setInterval((a) => currentVal = counter(a, currentVal + 1), intervals[workerData], workerData);

}
