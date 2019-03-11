const tf = require('@tensorflow/tfjs-node');

const fs = require('fs');
const path = require('path');
const obj = JSON.parse(fs.readFileSync('output/1b94aad142e6c2b8af9f38a1ee687286.json', 'utf8'));
const reconstruct = require('../reconstruct_file.js');


function createModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.dense({units: 1000, activation: 'relu', inputShape: [6162]}),
      tf.layers.dropout({rate: 0.2}),
      tf.layers.dense({units: 10, activation: 'relu'}),
      tf.layers.dropout({rate: 0.2}),
      tf.layers.dense({units: 1000, activation: 'relu'}),
      tf.layers.dropout({rate: 0.2}),
      tf.layers.dense({units: 6162, activation: 'linear'}),
      tf.layers.dropout({rate: 0.2}),
    ]
  });

  let decoder = tf.sequential();
  for (let i = 4; i < 8; i++) {
    decoder.add(model.layers[i]);
  }

  model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
  model.summary();

  decoder.summary();
  const data = tf.randomNormal([10, 6162]);

  model.fit(data, data, {
    epochs: 100,
    batchSize: 50,
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`),
      onTrainEnd: () => saveModel(),
    }
  });

  return {model, decoder}
}

async function saveModel() {
  let {model, decoder} = createModel();
  const modelResult = await model.save('file:///Users/Chivalry/WebstormProjects/threeProject/trained-models');
  // const decoderResult = await decoder.save('file:///Users/Chivalry/WebstormProjects/threeProject/trained-models');
  console.log(modelResult);
  // console.log(decoderResult);
}

async function loadModel() {
  const model = await tf.loadLayersModel('file:///Users/Chivalry/WebstormProjects/threeProject/trained-models/model.json');
  console.log(model);
  return model;
}

/**
 * @param model
 * @param data
 * @return {Promise<void>}
 */
async function makePredict(model, data) {
  let rawData = await model.predict(data).data();

  let newData = Array.from(rawData);

  reconstruct({
    data: newData,
    levels: 5,
    output: 'tmmp.obj',
  })
}

loadModel().then((model) => {
  model.summary();

  const data = tf.randomNormal([1, 6162]);
  makePredict(model, data);
}).catch((err) => {
  console.log(err);
  // saveModel();
});
