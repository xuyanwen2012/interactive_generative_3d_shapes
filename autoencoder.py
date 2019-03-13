#!/usr/bin/env python3
"""run / test / train autoencoder.

Usage:
  autoencoder.py train <num_epochs>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--autosave <autosave_frequency>]
    [--autosave-path <autosave_path>]
    [--snapshot <snapshot_frequency>]
    [--snapshot-path <snapshot_path>]
    [--batch-size <batch_size>]
    [--train-test-split <split_ratio>]
  autoencoder.py summarize-runs <model_path>
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py test
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py repredict <output> <count>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py repredict snapshot <snapshot> <count>
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py interpolate <key1> <key2> <interp>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py gen-random <output> <count>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py gen-latent-codes <output>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py gen-latent-models <output>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py interpolate <key1> <key2>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py add-features <key1> <key2>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py remix <original> <add> <subtract>
    [--model <model_path>]
    [--use-dataset <dataset_path>]
    [--train-test-split <split_ratio>]
  autoencoder.py list-keys

Options:
  -h --help                         show this screen
  --use-dataset <dataset_path>      use a specific dataset (should be a URL)
  --model <model_path>              select model path to load from  [default: model]
  --autosave <autosave_frequency>   set autosave frequency          [default: 10]
  --autosave-path <autosave_path>   set autosave path               [default: model]
  --snapshot <snapshot_frequency>   set snapshot frequency          [default: 50]
  --snapshot-path <snapshot_path>   set snapshot path               [default: model/snapshots]
  --batch-size <batch_size>         set training batch size         [default: 32]
  --train-test-split <split_ratio>  set train / test split ratio    [default: 0.8]
"""
from urllib.request import urlopen
import pickle as pkl
import numpy as np
import json
import sys
import os
import shutil
import subprocess
from docopt import docopt
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, LeakyReLU
from keras.models import Model, load_model
from keras.losses import mean_squared_error
import keras

""" Helper functions """


def makedirs(path):
    """ Builds out all directories for an arbitrary file path. Call this before writing files. """
    basedir, file = os.path.split(path)
    if basedir and not os.path.exists(basedir):
        os.makedirs(basedir)


""" D enforcement pattern """


def enforce(condition, fmt, *args, exception=Exception):
    if not condition:
        raise exception(fmt % args)


""" Load dataset """


def load_dataset(dataset_url):
    """ Loads a processed dataset (shrinkwrapped params as a flat JSON array) from a URL.

    Also caches said file to a '.cache/<filename>' temp file as a caching layer.
    """
    print("Loading dataset")
    file = os.path.split(dataset_url)[1]
    cached_path = os.path.join('.cache', file)

    if os.path.exists(cached_path):
        print("loading cached '%s'" % cached_path)
        with open(cached_path, 'rb') as f:
            return pkl.load(f)

    if dataset_url.startswith('https://'):
        print("fetching %s..." % dataset_url)
        data = urlopen(dataset_url).read()
        print("done; caching locally as '%s'" % cached_path)
        makedirs(cached_path)
        with open(cached_path, 'wb') as f:
            f.write(data)
        return pkl.loads(data)
    raise Exception("Unable to load dataset from '%s'" % dataset_url)


def validate_and_split_data(dataset, train_test_split=0.75):
    # Validate parameters...
    enforce(train_test_split > 0.0 and train_test_split <= 1.0, "invalid train / test split: %s", train_test_split)
    enforce(type(dataset) == dict, "Invalid dataset object: got %s (%s)!", dataset, type(dataset))
    enforce(set(dataset.keys()) == set(['data', 'keys']), "Invalid dataset format! (has keys %s)", set(dataset.keys()))

    # Load data, keys
    data, keys = dataset['data'], dataset['keys']

    enforce(type(data) == np.ndarray and type(keys) == list, "Invalid types!: data %s, keys %s", type(data), type(keys))
    enforce(len(data.shape) == 2 and data.shape[1] == 6162, "Invalid shape! %s", data.shape)
    enforce(len(keys) == data.shape[0], "# keys (%s) does not match # data elements (%s)!", len(keys), data.shape[1])

    # Calculate train / test split
    num_train = int(data.shape[0] * train_test_split)
    num_test = data.shape[0] - num_train

    enforce(num_train > 0,
            "must have at least 1 training sample; got %s train, %s test from %s elements, %s train / test split",
            num_train, num_test, data.shape[0], train_test_split)

    # Split data
    x_train, x_test = np.split(data, [num_train], 0)
    print("split data %s => x_train %s, x_test %s with train / test split of %s" % (
        data.shape, x_train.shape, x_test.shape, train_test_split))

    return x_train, x_test

class AutoencoderModel:
    def __init__(
            self,
            dataset,
            train_test_split=0.75,
            autoload_path='model',
            autosave_path='model',
            autosave_frequency=10,
            model_snapshot_path='model/snapshots',
            model_snapshot_frequency=100,
            input_size=6162,
            hidden_layer_size=1000,
            encoding_size=10):
        """ Constructs an Autoencoder with persistent data for iterative training.

        input_size, hidden_layer_size, encoding_size:
            Parameters for the autoencoder's layer sizes. Don't touch these unless you have good reason to.
            Also, these will get ignored if the model is loaded from a file.

        autoload_path, autosave_path:
            Directory to automatically save / load the model to / from.
            In most cases these should be set to the same thing.
            The one exception is you could set autoload_path to independently to load from a model snapshot,
                ie. 'model/snapshots/1200/', or something

            The model will be autoloaded only from __init__, and this parameter is not saved.

            autosave_path is saved, however, as the model will be autosaved (if this is set) after:
                1) build() is called   (will be called automatically iff autoload is set but there isn't any persistent model data yet)
                2) training epochs, after every model_snapshot_frequency epochs

        model_snapshot_path, model_snapshot_frequency:
            If set, autosaves model snapshots to this directory – specifically, a subdirectory.
            Snapshots are saved every model_snapshot_frequency epochs.

            With default parameters it would save to:
                models/model.h5, model_state.json                   (current state)
                models/snapshots/100/model.h5, model_state.json     (model after 100 epochs)
                models/snapshots/200/model.h5, model_state.json     (model after 200 epochs)
                models/snapshots/300/model.h5, model_state.json     (model after 300 epochs)
                ...
                models/snapshots/N/model.h5, model_state.json       (model after N epochs)
        """
        self.dataset = dataset
        self.keys = dataset['keys']
        self.data = validate_and_split_data(dataset, train_test_split)
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.encoding_size = encoding_size
        self.autosave_path = autosave_path
        self.autosave_frequency = autosave_frequency
        self.model_snapshot_path = model_snapshot_path
        self.model_snapshot_frequency = model_snapshot_frequency
        self.current_epoch = 0

        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.encoding_size = encoding_size

        if autoload_path:
            if not self.load(autoload_path):
                self.build()

    def load_model (self, path):
        model_path = os.path.join(path, 'model.h5')
        state_path = os.path.join(path, 'model_state.json')
        if os.path.exists(model_path):
            enforce(os.path.exists(state_path), "could not load model state from %s"%state_path)
            with open(state_path, 'r') as f:
                state = json.loads(f.read())
            autoencoder = load_model(model_path)
            encoder, decoder = self.get_encoder_and_decoder(autoencoder)
            return {
                'epoch': state['current_epoch'],
                'autoencoder': autoencoder,
                'encoder': encoder,
                'decoder': decoder,
            }
        return None

    def load(self, path=None):
        """ Loads keras model and other data from a directory, given by path.

        If path is not given, defaults to self.autoload_path.
        This method may fail. If neither given, returns False.

        Loads the following files:
            <path>/model.h5
            <path>/model_state.json

        model_state.json is additional persistent data we need, like the
        current training epoch.
        """
        path = path or self.autoload_path
        result = self.load_model(path)
        if result:
            print("Loaded model from %s"%path)
            self.current_epoch = result['epoch']
            self.autoencoder = result['autoencoder']
            self.encoder = result['encoder']
            self.decoder = result['decoder']
            return True
        print("Can't load model from '%s', file does not exist" % path)
        return False

    def save(self, path=None, save_summary=True):
        """ Saves keras model and other data to a directory, given by path.

        If path is not specified, defaults to self.autosave_path.
        If neither specified, raises an exception.

        Saves the following files:
            <path>/model.h5
            <path>/model_state.json

        model_state.json is additional persistent data we need, like the
        current training epoch.
        """

        # Use autosave_path if path not specified
        path = path or self.autosave_path
        if not path:
            # If neither specified, we have no idea where to save the model, so raise an error
            raise Exception("Cannot save, no path specified (self.autosave_path = %s)" % self.autosave_path)

        # Save keras model
        model_path = os.path.join(path, 'model.h5')
        makedirs(model_path)
        print("Saving as '%s'" % model_path)
        self.autoencoder.save(model_path)

        # Save additional persistent state (current_epoch, etc)
        state_path = os.path.join(path, 'model_state.json')
        with open(state_path, 'w') as f:
            f.write(json.dumps({
                'current_epoch': self.current_epoch,
            }))

        # Summarize model
        if save_summary:
            self.save_model_summary(path, 
                self.summarize_model(path, 
                    data=self.data, 
                    autoencoder=self.autoencoder,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    epoch=self.current_epoch))        

    def summarize_model (self, model_path, data, autoencoder, encoder, decoder, epoch):
        print("summarizing '%s'"%model_path)
        x_train, x_test = data
        z_train, z_test = map(encoder.predict, (x_train, x_test))
        y_train, y_test = map(decoder.predict, (z_train, z_test))
        train_loss = autoencoder.evaluate(x_train, x_train)
        test_loss  = autoencoder.evaluate(x_test, x_test)
        print("train_loss: %s, test_loss: %s, test/train loss %0.1f%%, z-var %s, %s, y/x var %0.1f%%, %0.1f%%"%(
            train_loss, test_loss, test_loss / train_loss * 100,
            np.var(z_train), np.var(z_test), 
            np.var(y_train) / np.var(x_train) * 100,
            np.var(y_test) / np.var(x_test) * 100))
        summary = { 
            'epoch':        epoch,  
            'train_loss':   train_loss, 
            'test_loss':    test_loss 
        }
        def summarize_distribution (name, x):
            summary[name] = {
                'min':      float(np.min(x)),
                'max':      float(np.max(x)),
                'mean':     float(np.mean(x)),
                'var':      float(np.var(x)),
            }
        summarize_distribution('x_train', x_train)
        summarize_distribution('x_test', x_test)
        summarize_distribution('y_train', y_train)
        summarize_distribution('y_test', y_test)
        summarize_distribution('z_train', z_train)
        summarize_distribution('z_test', z_test)

        # print(z_train.shape)
        for i in range(10):
            # print(z_train[:,i].shape)
            summarize_distribution('z_train[%d]'%i, z_train[:,i])
        return summary

    def load_model_summary (self, model_path, data=None, rebuild=False):
        data = data or self.data
        summary_path = model_path and os.path.join(model_path, 'summary.json')
        if not rebuild and os.path.exists(summary_path):
            print("loading '%s'"%summary_path)
            with open(summary_path, 'r') as f:
                return json.loads(f.read())

        print("no snapshot for %s, rebuilding..."%model_path)
        model = self.load_model(model_path)
        if model is None:
            print("couldn't load model from %s! aborting"%model)
            return None
        summary = self.summarize_model(data=data, model_path=model_path, **model)
        self.save_model_summary(model_path, summary)
        return summary

    def load_this_model_summary (self, model_path):
        summary_path = model_path and os.path.join(model_path, 'summary.json')
        if os.path.exists(summary_path):
            print("loading '%s'"%summary_path)
            with open(summary_path, 'r') as f:
                return json.loads(f.read())

        summary = self.summarize_model(os.path.join(self.model_snapshot_path, str(self.current_epoch)), 
            data=self.data, 
            autoencoder=self.autoencoder,
            encoder=self.encoder,
            decoder=self.decoder,
            epoch=self.current_epoch)
        self.save_model_summary(
            os.path.join(self.model_snapshot_path, 
            str(self.current_epoch)), 
            summary)
        return summary

    def save_model_summary (self, model_path, summary=None):
        summary_path = model_path and os.path.join(model_path, 'summary.json')
        print("saving '%s'"%summary_path)
        makedirs(summary_path)
        with open(summary_path, 'w') as f:
            f.write(json.dumps(summary))

    def summarize_snapshots (self, model_path, rebuild=False):
        summaries = []
        print("summarizing...")

        snapshot_path = os.path.join(model_path, 'snapshots')
        snapshots = list(os.listdir(snapshot_path))
        for i, snapshot in enumerate(snapshots):
            if not snapshot.isnumeric():
                continue
            path = os.path.join(snapshot_path, snapshot)
            summary = self.load_model_summary(path, rebuild)
            if summary is None:
                print("Failed to load '%s', skipping"%path)
            else:
                summaries.append(self.load_model_summary(path, rebuild))
            print("%s / %s"%(i+1, len(snapshots)))
        summaries.sort(key=lambda x: x['epoch'])

        def csv_header (summary):
            for key, value in summary.items():
                if type(value) == dict:
                    for k, v in value.items():
                        yield '%s.%s'%(key, k)
                else:
                    yield key

        def csv_values (summary):
            for value in summary.values():
                if type(value) == dict:
                    for value in value.values():
                        yield value
                else:
                    yield value

        # print(list(csv_header(summaries[0])))
        # for summary in summaries:
        #     print(set(map(type, csv_values(summary))))
        #     print(list(csv_values(summary)))
        csv_data = '\n'.join([ ', '.join(csv_header(summaries[0])) ] + [
            ', '.join(map(str, csv_values(summary)))
            for summary in summaries
        ])
        path = os.path.join('summary', '%s.csv'%model_path.split('/')[0])
        makedirs(path)
        print("saving '%s'"%path)
        with open(path, 'w') as f:
            f.write(csv_data)

    def build(self):
        """ Builds a new model.

        Called automatically by Model's constructor iff autoload path set but there are no files to load from.
        Otherwise, you could disable autoload and call this explicitely to construct a new model.

        Additionally, if self.autosave_path is set this will autosave after constructing the model.
        """
        print("Building model")
        self.autoencoder = Sequential([
            Dense(self.hidden_layer_size, input_shape=(self.input_size,)),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(self.encoding_size),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(self.hidden_layer_size),
            LeakyReLU(alpha=0.1),
            Dropout(0.2),
            Dense(self.input_size),
            Activation('linear')
        ])
        print("compiling...")
        self.autoencoder.compile(optimizer='adam', loss='mean_squared_error')

        print("Built autoencoder:")
        self.autoencoder.summary()
        self.encoder, self.decoder = self.get_encoder_and_decoder(self.autoencoder)

        if self.autosave_path:
            self.save()

    def get_encoder_and_decoder(self, model):
        # model.summary()
        enforce(len(model.layers) in (8, 11, 12),
                "autoencoder model has changed, expected 8, 11, or 12 layers but got %s:\n\t%s",
                len(model.layers),
                '\n\t'.join(['%s: %s' % values for values in enumerate(model.layers)]))

        if len(model.layers) == 8:
            encoder_layers, decoder_layers = 4, 4
        elif len(model.layers) == 11:
            encoder_layers, decoder_layers = 6, 5
        elif len(model.layers) == 12:
            encoder_layers, decoder_layers = 6, 6

        print("encoder:")
        encoder_input = Input(shape=(self.input_size,))
        encoder = encoder_input
        for layer in model.layers[0:encoder_layers]:
            encoder = layer(encoder)
        encoder = Model(encoder_input, encoder)

        print("decoder:")
        decoder_input = Input(shape=(self.encoding_size,))
        decoder = decoder_input
        for layer in model.layers[encoder_layers:encoder_layers+decoder_layers]:
            decoder = layer(decoder)
        decoder = Model(decoder_input, decoder)
        return encoder, decoder

    def train(self, epochs, batch_size=32):

        enforce(self.autosave_frequency > 0, "autosave frequency must be > 0, got %s", self.autosave_frequency)
        enforce(batch_size > 0, "batch size must be > 0, got %s", batch_size)
        x_train, x_test = self.data

        """ Train model """
        print("Training model for %s epochs (epochs %s -> %s)" % (
        epochs, self.current_epoch, self.current_epoch + epochs))
        if self.model_snapshot_frequency and self.model_snapshot_path:
            next_snapshot = (self.current_epoch // self.model_snapshot_frequency + 1) * self.model_snapshot_frequency
            print("Next snapshot at epoch %s" % next_snapshot)
        else:
            next_snapshot = None

        last_saved_epoch = self.current_epoch

        while epochs > 0:
            print("Training on epoch %s -> %s" % (self.current_epoch, self.current_epoch + self.autosave_frequency))
            self.autoencoder.fit(x_train, x_train, validation_data=(x_test, x_test),
                epochs=self.autosave_frequency, batch_size=batch_size)
            epochs -= self.autosave_frequency
            self.current_epoch += self.autosave_frequency

            if next_snapshot and self.current_epoch >= next_snapshot:
                print("Saving snapshot at epoch %s" % (self.current_epoch))
                self.save(os.path.join(self.model_snapshot_path, str(self.current_epoch)))
                next_snapshot = (
                                            self.current_epoch // self.model_snapshot_frequency + 1) * self.model_snapshot_frequency
                print("Next snapshot at epoch %s" % next_snapshot)

            self.save_model_summary(os.path.join(self.model_snapshot_path, str(self.current_epoch)), 
                self.summarize_model(os.path.join(self.model_snapshot_path, str(self.current_epoch)), 
                    data=self.data, 
                    autoencoder=self.autoencoder,
                    encoder=self.encoder,
                    decoder=self.decoder,
                    epoch=self.current_epoch))

            print("Autosaving...")
            self.save(save_summary=False)
            last_saved_epoch = self.current_epoch

        if last_saved_epoch != self.current_epoch:
            print("Autosaving...")
            self.save()

    def evaluate_using_test_data(self):
        x_train, x_test = self.data

        print("evaluation using test data (shape %s) TBD" % (x_test.shape,))

        # TODO: compare these using loss function
        y_train = self.autoencoder.predict(x_train)
        y_test = self.autoencoder.predict(x_test)

    def repredict(self, count, output_path):
        x_train, x_test = self.data
        y_train = self.autoencoder.predict(x_train)
        y_test = self.autoencoder.predict(x_test)

        print("repredicting input data")
        train_path = os.path.join(output_path, 'params', 'train')
        test_path = os.path.join(output_path, 'params', 'test')
        makedirs(os.path.join(train_path, 'foo.json'))
        makedirs(os.path.join(test_path, 'foo.json'))

        print("saving to %s, %s" % (train_path, test_path))
        for i in range(count):
            j = i + y_train.shape[0]
            print("%s / %s" % (i + 1, count))
            with open(os.path.join(train_path, '%s.output.json' % self.dataset['keys'][i]), 'w') as f:
                f.write(json.dumps([float(value) for value in y_train[i]]))
            with open(os.path.join(train_path, '%s.input.json' % self.dataset['keys'][i]), 'w') as f:
                f.write(json.dumps([float(value) for value in x_train[i]]))
            with open(os.path.join(test_path, '%s.output.json' % self.dataset['keys'][j]), 'w') as f:
                f.write(json.dumps([float(value) for value in y_test[i]]))
            with open(os.path.join(test_path, '%s.input.json' % self.dataset['keys'][j]), 'w') as f:
                f.write(json.dumps([float(value) for value in x_test[i]]))

        print("building objs...")
        obj_train_path = os.path.join(output_path, 'obj', 'train')
        obj_test_path = os.path.join(output_path, 'obj', 'test')
        makedirs(os.path.join(obj_train_path, 'foo.obj'))
        makedirs(os.path.join(obj_test_path, 'foo.obj'))

        subprocess.run(['node', 'index.js', 'reconstruct', train_path, obj_train_path])
        subprocess.run(['node', 'index.js', 'reconstruct', test_path, obj_test_path])
        print("done")

    def generate_interpolated (self, key1, key2, interpolations):
        x_train, x_test = self.data
        idx1 = [ i for i, key in enumerate(self.keys) if key == key1 ]
        idx2 = [ i for i, key in enumerate(self.keys) if key == key2 ]
        enforce(len(idx1) > 0, "invalid key %s", key1); idx1 = idx1[0]
        enforce(len(idx2) > 0, "invalid key %s", key2); idx2 = idx2[0]

        x1 = x_train[idx1] if idx1 < x_train.shape[0] else x_test[idx1 - x_train.shape[0]]
        x2 = x_train[idx2] if idx2 < x_train.shape[0] else x_test[idx2 - x_train.shape[0]]
        print(x1.shape)
        z1 = self.encoder.predict(np.array([ x1 ]))[0]
        z2 = self.encoder.predict(np.array([ x2 ]))[0]

        path = os.path.join('interpolated', '%s-%s'%(key1, key2))
        if not os.path.exists(path):
            os.makedirs(path)

        print("writing to %s"%path)
        for interp in interpolations:
            zinterp = z1 * (1 - interp) + z2 * interp
            yinterp = self.decoder.predict(np.array([ zinterp ]))[0]
            with open(os.path.join(path, '%s.json'%(interp)), 'w') as f:
                f.write(json.dumps([ float(value) for value in yinterp ]))
        subprocess.run([ 'node', 'index.js', 'reconstruct', path, path ])
        for file in os.listdir(path):
            if file.endswith('.json'):
                os.remove(os.path.join(path, file))

    def generate_add_features (self, key1, key2, interpolations):
        x_train, x_test = self.data
        idx1 = [ i for i, key in enumerate(self.keys) if key == key1 ]
        idx2 = [ i for i, key in enumerate(self.keys) if key == key2 ]
        enforce(len(idx1) > 0, "invalid key %s", key1); idx1 = idx1[0]
        enforce(len(idx2) > 0, "invalid key %s", key2); idx2 = idx2[0]

        x1 = x_train[idx1] if idx1 < x_train.shape[0] else x_test[idx1 - x_train.shape[0]]
        x2 = x_train[idx2] if idx2 < x_train.shape[0] else x_test[idx2 - x_train.shape[0]]
        print(x1.shape)
        z1 = self.encoder.predict(np.array([ x1 ]))[0]
        z2 = self.encoder.predict(np.array([ x2 ]))[0]

        path = os.path.join('added_feature', '%s-%s'%(key1, key2))
        if not os.path.exists(path):
            os.makedirs(path)

        print("writing to %s"%path)
        for interp in interpolations:
            zinterp = z1 + z2 * interp
            yinterp = self.decoder.predict(np.array([ zinterp ]))[0]
            with open(os.path.join(path, '%s.json'%(interp)), 'w') as f:
                f.write(json.dumps([ float(value) for value in yinterp ]))
        subprocess.run([ 'node', 'index.js', 'reconstruct', path, path ])
        for file in os.listdir(path):
            if file.endswith('.json'):
                os.remove(os.path.join(path, file))

    def generate_remix (self, origin_key, add_key, sub_key, interpolations):
        key1, key2, key3 = origin_key, add_key, sub_key
        x_train, x_test = self.data
        idx1 = [ i for i, key in enumerate(self.keys) if key == key1 ]
        idx2 = [ i for i, key in enumerate(self.keys) if key == key2 ]
        idx3 = [ i for i, key in enumerate(self.keys) if key == key3 ]
        enforce(len(idx1) > 0, "invalid key %s", key1); idx1 = idx1[0]
        enforce(len(idx2) > 0, "invalid key %s", key2); idx2 = idx2[0]
        enforce(len(idx3) > 0, "invalid key %s", key3); idx3 = idx3[0]

        x1 = x_train[idx1] if idx1 < x_train.shape[0] else x_test[idx1 - x_train.shape[0]]
        x2 = x_train[idx2] if idx2 < x_train.shape[0] else x_test[idx2 - x_train.shape[0]]
        x3 = x_train[idx3] if idx3 < x_train.shape[0] else x_test[idx3 - x_train.shape[0]]
        print(x1.shape)
        z1 = self.encoder.predict(np.array([ x1 ]))[0]
        z2 = self.encoder.predict(np.array([ x2 ]))[0]
        z3 = self.encoder.predict(np.array([ x3 ]))[0]

        path = os.path.join('interpolated', '%s-%s'%(key1, key2))
        if not os.path.exists(path):
            os.makedirs(path)

        print("writing to %s"%path)
        for interp in interpolations:
            zinterp = z1 + z2 * (1 - interp) + z3 * interp 
            yinterp = self.decoder.predict(np.array([ zinterp ]))[0]
            with open(os.path.join(path, '%s.json'%(interp)), 'w') as f:
                f.write(json.dumps([ float(value) for value in yinterp ]))
        subprocess.run([ 'node', 'index.js', 'reconstruct', path, path ])
        for file in os.listdir(path):
            if file.endswith('.json'):
                os.remove(os.path.join(path, file))

    def list_keys (self):
        x_train, x_test = self.data
        print("%s keys (%s train, %s test"%(
            len(self.keys),
            x_train.shape[0],
            x_test.shape[0],
        ))
        for i, key in enumerate(self.keys):
            print("%s %s"%(
                'TRAIN' if i < x_train.shape[0] else 'TEST',
                key
            ))

    def generate_random (self, output_path, count):
        json_path = os.path.join(output_path, 'json')
        if not os.path.exists(json_path):
            os.makedirs(json_path)

        print("generating...")
        x_train, x_test = self.data
        z_train = self.encoder.predict(x_train)
        z_mean = np.mean(z_train)
        z_stdev = np.var(z_train) ** 0.5
        z_samples = np.random.normal(loc=z_mean, scale=z_stdev, size=(count, 10))
        y_samples = self.decoder.predict(z_samples)
        # print(z_samples.shape, y_samples.shape, y_samples[0].shape)

        for i in range(count):
            with open(os.path.join(json_path, '%s.json'%i), 'w') as f:
                f.write(json.dumps(list(map(float, y_samples[i]))))
        print("writing obj files...")
        subprocess.run([ 'node', 'index.js', 'reconstruct', json_path, output_path, '--rebuild' ])
        shutil.rmtree(json_path)

    def generate_latent_codes (self, model_path, output_path):
        pass

    def generate_latent_models (self, model_path, output_path):
        print("generating...")
        gencount = 0
        json_path = os.path.join(output_path, 'json')
        def save_model(kind, label, z):
            nonlocal gencount
            gencount += 1
            path = os.path.join(json_path, label, kind)
            makedirs(path)
            y = self.decoder.predict(np.array([ z ]))
            with open('%s.json'%path, 'w') as f:
                f.write(json.dumps(list(map(float, y[0]))))

        summary = self.load_this_model_summary(model_path)
        z_train, z_test = summary['z_train'], summary['z_test']
        z_nonzero = np.array([ 1 if summary['z_train[%s]'%i]['mean'] != 0 else 0 for i in range(10) ])
        z_allzero = np.ones(10) - z_nonzero

        min_, max_, mean, stdev = z_train['min'], z_train['max'] or 1, z_train['mean'] or 5, z_train['var'] ** 0.5 or 0.25
        save_model('all-0', 'all', np.ones(10) * 0)
        save_model('all-1', 'all', np.ones(10))
        save_model('all-half', 'all', np.ones(10) * 0.5)
        save_model('all-2', 'all', np.ones(10) * 2)
        if z_train['mean'] != 0:
            save_model('mean', 'all', np.ones(10) * z_train['mean'])
            save_model('mean-minus-1', 'all', np.ones(10) * (z_train['mean'] - z_train['var'] ** 0.5))
            save_model('mean-minus-2', 'all', np.ones(10) * (z_train['mean'] - z_train['var'] ** 0.5 * 2))
            save_model('mean-plus-1', 'all', np.ones(10) * (z_train['mean'] + z_train['var'] ** 0.5))
            save_model('mean-plus-2', 'all', np.ones(10) * (z_train['mean'] + z_train['var'] ** 0.5 * 2))
        if z_train['min'] != 0:
            save_model('min', 'all', np.ones(10) * z_train['min'])
        if z_train['max'] != 0:
            save_model('max', 'all', np.ones(10) * z_train['max'])
            save_model('max-2', 'all', np.ones(10) * z_train['max'] * 2)
            save_model('max-10', 'all', np.ones(10) * z_train['max'] * 10)

        if np.any(z_nonzero):
            save_model('all-1', 'all-nonzero', z_nonzero)
            save_model('all-half', 'all-nonzero', z_nonzero * 0.5)
            save_model('all-2', 'all-nonzero', z_nonzero * 2)
            if z_train['mean'] != 0:
                save_model('mean', 'all-nonzero', z_nonzero * z_train['mean'])
                save_model('mean-minus-1', 'all-nonzero', z_nonzero * (z_train['mean'] - z_train['var'] ** 0.5))
                save_model('mean-minus-2', 'all-nonzero', z_nonzero * (z_train['mean'] - z_train['var'] ** 0.5 * 2))
                save_model('mean-plus-1', 'all-nonzero', z_nonzero * (z_train['mean'] + z_train['var'] ** 0.5))
                save_model('mean-plus-2', 'all-nonzero', z_nonzero * (z_train['mean'] + z_train['var'] ** 0.5 * 2))
            if z_train['min'] != 0:
                save_model('min', 'all-nonzero', z_nonzero * z_train['min'])
            if z_train['max'] != 0:
                save_model('max', 'all-nonzero', z_nonzero * z_train['max'])
                save_model('max-2', 'all-nonzero', z_nonzero * z_train['max'] * 2)
                save_model('max-10', 'all-nonzero', z_nonzero * z_train['max'] * 10)

        if np.any(z_allzero):
            save_model('all-1', 'all-zero', z_allzero)
            save_model('all-half', 'all-zero', z_allzero * 0.5)
            save_model('all-2', 'all-zero', z_allzero * 2)
            if z_train['mean'] != 0:
                save_model('mean', 'all-zero', z_allzero * z_train['mean'])
                save_model('mean-minus-1', 'all-zero', z_allzero * (z_train['mean'] - z_train['var'] ** 0.5))
                save_model('mean-minus-2', 'all-zero', z_allzero * (z_train['mean'] - z_train['var'] ** 0.5 * 2))
                save_model('mean-plus-1', 'all-zero', z_allzero * (z_train['mean'] + z_train['var'] ** 0.5))
                save_model('mean-plus-2', 'all-zero', z_allzero * (z_train['mean'] + z_train['var'] ** 0.5 * 2))
            if z_train['min'] != 0:
                save_model('min', 'all-zero', z_allzero * z_train['min'])
            if z_train['max'] != 0:
                save_model('max', 'all-zero', z_allzero * z_train['max'])
                save_model('max-2', 'all-zero', z_allzero * z_train['max'] * 2)
                save_model('max-10', 'all-zero', z_allzero * z_train['max'] * 10)

        for i in range(10):
            latent = summary['z_train[%s]'%i]
            z = np.array([ 0 if i != j else 1 for j in range(10) ])
            max_, mean, stdev = latent['max'] or 1, latent['mean'] or 5, latent['var'] ** 0.5 or 0.25
            minmaxrange = max_ - latent['min']
            if latent['min']:
                save_model('min', 'non-zero-%s'%i, z * latent['min'])

            if latent['mean']:
                for k in range(10+1):
                    save_model('interp-%d'%k, 'non-zero-%s'%i, z * k / 10.0 * minmaxrange + latent['min'])
                save_model('mean', 'non-zero-%s'%i, z * mean)
                save_model('mean-minus-1', 'non-zero-%s'%i, z * (mean - stdev))
                save_model('mean-minus-2', 'non-zero-%s'%i, z * (mean - stdev * 2))
                save_model('mean-plus-1', 'non-zero-%s'%i, z * (mean + stdev))
                save_model('mean-plus-2', 'non-zero-%s'%i, z * (mean + stdev * 2))
                save_model('max', 'non-zero-%s'%i, z * max_)
                save_model('max-2', 'non-zero-%s'%i, z * max_ * 2)
                save_model('max-10', 'non-zero-%s'%i, z * max_ * 10)
            else:
                for k in range(10+1):
                    save_model('interp-%d'%k, 'zero-%s'%i, z * k / 10.0)
                save_model('mean', 'zero-%s'%i, z * mean)
                save_model('mean-minus-1', 'zero-%s'%i, z * (mean - stdev))
                save_model('mean-minus-2', 'zero-%s'%i, z * (mean - stdev * 2))
                save_model('mean-plus-1', 'zero-%s'%i, z * (mean + stdev))
                save_model('mean-plus-2', 'zero-%s'%i, z * (mean + stdev * 2))
                save_model('max', 'zero-%s'%i, z * max_)
                save_model('max-2', 'zero-%s'%i, z * max_ * 2)
                save_model('max-10', 'zero-%s'%i, z * max_ * 10)

        print("generated %s files"%gencount)
        print("building objs...")
        for dir in os.listdir(json_path):
            json_files = os.path.join(json_path, dir)
            target_path = os.path.join(output_path, dir)
            if not os.path.exists(target_path):
                os.makedirs(target_path)
            subprocess.run(['node', 'index.js', 'reconstruct', json_files, target_path])
            shutil.rmtree(json_files)


DEFAULT_DATASET = 'https://raw.githubusercontent.com/SeijiEmery/shape-net-data/master/datasets/training-lv5.pkl'


class ArgumentParsingException(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


if __name__ == '__main__':
    args = docopt(__doc__)
    # print(args)

    """ Validate arguments """
    enforce_arg = lambda cond, fmt, *args: enforce(cond, fmt, *args, exception=ArgumentParsingException)


    def parse_arg(T, key, min_bound=None, max_bound=None):
        try:
            value = T(args[key])
            if min_bound is not None:
                enforce_arg(value >= min_bound, "%s must be >= %s, got %s", key, min_bound, args[key])
            if max_bound is not None:
                enforce_arg(value <= max_bound, "%s must be <= %s, got %s", key, max_bound, args[key])
            return value
        except ValueError:
            enforce_arg(False, "%s should be %s, got '%s'", key, str(T), args[key])

    try:
        data_url = args['--use-dataset'] or DEFAULT_DATASET
        model_path = args['--model']
        autosave_path = args['--autosave-path']
        autosave_freq = parse_arg(int, '--autosave', min_bound=0)
        snapshot_path = args['--snapshot-path']
        snapshot_freq = parse_arg(int, '--snapshot', min_bound=0)
        batch_size = parse_arg(int, '--batch-size', min_bound=1)
        train_test_split = parse_arg(float, '--train-test-split', min_bound=0.0, max_bound=1.0)
        # enforce_arg(os.path.exists(model_path), "model_path '%s' does not exist", model_path)

        if args['train']:
            num_epochs = parse_arg(int, '<num_epochs>', min_bound=1)
        elif args['test']:
            pass
        elif args['repredict'] or args['gen-latent-codes'] or args['gen-latent-models'] or args['gen-random']:
            output_path = args['<output>'] or 'repredicted'
            if args['repredict'] or args['gen-random']:
                count = parse_arg(int, '<count>', min_bound=1)
                if args['snapshot']:
                    snapshot_id = parse_arg(str, '<snapshot>')
                    model_path = os.path.join(model_path, 'snapshots', snapshot_id)
                    enforce_arg(os.path.exists(model_path), "no snapshot %s at %s", snapshot_id, model_path)
                    output_path = os.path.join(output_path, snapshot_id)
        elif args['interpolate'] or args['add-features']:
            key1 = parse_arg(str, '<key1>')
            key2 = parse_arg(str, '<key2>')
            interp = args['<interp>']
            if interp is not None:
                interp = parse_arg(float, '<interp>', min_bound=0, max_bound=1)

        elif args['remix']:
            origin_key = parse_arg(str, '<original>')
            add_key = parse_arg(str, '<add>')
            sub_key = parse_arg(str, '<subtract>')
            interp = args['<interp>']
            if interp is not None:
                interp = parse_arg(float, '<interp>', min_bound=0, max_bound=1)

        elif args['summarize-runs']:
            model_path = args['<model_path>']
            snapshot_path = os.path.join(model_path, 'snapshots')

    except ArgumentParsingException as e:
        print("Invalid argument: %s" % e)
        sys.exit(-1)

    """ Run everything """
    dataset = load_dataset(data_url)
    autoencoder = AutoencoderModel(
        dataset=dataset,
        train_test_split=train_test_split,
        autoload_path=model_path,
        autosave_path=autosave_path,
        autosave_frequency=autosave_freq,
        model_snapshot_path=snapshot_path,
        model_snapshot_frequency=snapshot_freq)

    if args['train']:
        autoencoder.train(
            epochs=num_epochs,
            batch_size=batch_size)

    elif args['summarize-runs']:
        autoencoder.summarize_snapshots(model_path)

    elif args['test']:
        autoencoder.evaluate_using_test_data()

    elif args['repredict']:
        autoencoder.repredict(
            count=count,
            output_path=output_path)

    elif args['gen-latent-models']:
        autoencoder.generate_latent_models(
            model_path=model_path, 
            output_path=output_path)

    elif args['gen-latent-codes']:
        autoencoder.generate_latent_codes(
            model_path=model_path,
            output_path=output_path)

    elif args['interpolate']:
        if interp is not None:
            autoencoder.generate_interpolated(
                key1=key1, key2=key2, interpolations=[ interp ])
        else:
            autoencoder.generate_interpolated(
                key1=key1, key2=key2, interpolations=[ 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.8, 0.9, 1.0 ])

    elif args['add-features']:
        if interp is not None:
            autoencoder.generate_add_features(
                key1=key1, key2=key2, interpolations=[ interp ])
        else:
            autoencoder.generate_add_features(
                key1=key1, key2=key2, interpolations=[ 
                    -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.8, 0.9, 1.0 ])

    elif args['remix']:
        if interp is not None:
            autoencoder.generate_remix(
                origin_key=origin_key, add_key=add_key, sub_key=sub_key, interpolations=[ interp ])
        else:
            autoencoder.generate_remix(
                origin_key=origin_key, add_key=add_key, sub_key=sub_key, interpolations=[ 
                    -1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1,
                    0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.6, 0.8, 0.9, 1.0 ])

    elif args['gen-random']:
        autoencoder.generate_random(output_path=output_path, count=count)

    elif args['list-keys']:
        autoencoder.list_keys()
