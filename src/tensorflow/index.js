const tf = require('@tensorflow/tfjs-node');

const fs = require('fs');
const obj = JSON.parse(fs.readFileSync('output/1b94aad142e6c2b8af9f38a1ee687286.json', 'utf8'));

const input = tf.input({shape: [6162]});
const encoderA = tf.layers.dense({
  units: 1000,
  activation: 'sigmoid'
}).apply(input);

const encoderB = tf.layers.dense({
  units: 10,
  activation: 'sigmoid'
}).apply(encoderA);

const decoderA = tf.layers.dense({
  units: 1000,
  activation: 'sigmoid'
}).apply(encoderB);

const decoderB = tf.layers.dense({
  units: 6162,
  activation: 'linear'
}).apply(decoderA);

const model = tf.model({inputs: input, outputs: decoderB});
model.compile({optimizer: 'adam', loss: 'meanSquaredError'});
model.summary();

const data = tf.randomNormal([1000, 6162]);

model.fit(data, data, {
  epochs: 100,
  batchSize: 50,
  callbacks: {
    onEpochEnd: (epoch, log) => console.log(`Epoch ${epoch}: loss = ${log.loss}`)
  }
});
