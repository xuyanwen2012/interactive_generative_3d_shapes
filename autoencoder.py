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
from docopt import docopt
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model


""" Helper functions """
def makedirs (path):
    """ Builds out all directories for an arbitrary file path. Call this before writing files. """
    basedir, file = os.path.split(path)
    if basedir and not os.path.exists(basedir):
        os.makedirs(basedir)

""" D enforcement pattern """
def enforce (condition, fmt, *args, exception=Exception):
    if not condition:
        raise exception(fmt%args)

""" Load dataset """
def load_dataset(dataset_url):
    """ Loads a processed dataset (shrinkwrapped params as a flat JSON array) from a URL.

    Also caches said file to a '.cache/<filename>' temp file as a caching layer.
    """
    print("Loading dataset")
    file = os.path.split(dataset_url)[1]
    cached_path = os.path.join('.cache', file)

    if os.path.exists(cached_path):
        print("loading cached '%s'"%cached_path)
        with open(cached_path, 'rb') as f:
            return pkl.load(f)

    if dataset_url.startswith('https://'):
        print("fetching %s..."%dataset_url)
        data = urlopen(url).read()
        print("done; caching locally as '%s'"%cached_path)
        makedirs(cached_path)
        with open(cached_path, 'wb') as f:
            f.write(data)
        return pkl.loads(data)
    raise Exception("Unable to load dataset from '%s'"%dataset_url)


def validate_and_split_data (dataset, train_test_split=0.75):
    # Validate parameters...
    enforce(train_test_split > 0.0 and train_test_split <= 1.0, "invalid train / test split: %s", train_test_split)
    enforce(type(dataset) == dict, "Invalid dataset object: got %s (%s)!", dataset, type(dataset))
    enforce(set(dataset.keys()) == set([ 'data', 'keys' ]), "Invalid dataset format! (has keys %s)", set(dataset.keys()))

    # Load data, keys
    data, keys = dataset['data'], dataset['keys']

    enforce(type(data) == np.ndarray and type(keys) == list, "Invalid types!: data %s, keys %s", type(data), type(keys))
    enforce(len(data.shape) == 2 and data.shape[1] == 6162, "Invalid shape! %s", data.shape)
    enforce(len(keys) == data.shape[0], "# keys (%s) does not match # data elements (%s)!", len(keys), data.shape[1])
    
    # Calculate train / test split
    num_train = int(data.shape[0] * train_test_split)
    num_test = data.shape[0] - num_train

    enforce(num_train > 0, "must have at least 1 training sample; got %s train, %s test from %s elements, %s train / test split",
        num_train, num_test, data.shape[0], train_test_split)

    # Split data
    x_train, x_test = np.split(data, [ num_train ], 0)
    print("split data %s => x_train %s, x_test %s with train / test split of %s"%(
        data.shape, x_train.shape, x_test.shape, train_test_split))

    return x_train, x_test


class AutoencoderModel:
    def __init__ (
            self,
            dataset,
            train_test_split = 0.75,
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
        self.data = validate_and_split_data(dataset, train_test_split)
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.encoding_size = encoding_size
        self.autosave_path = autosave_path
        self.autosave_frequency = autosave_frequency
        self.model_snapshot_path = model_snapshot_path
        self.model_snapshot_frequency = model_snapshot_frequency
        self.current_epoch = 0

        if autoload_path:
            if not self.load(autoload_path):
                self.build()

    def load (self, path=None):
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
        model_path = os.path.join(path, 'model.h5')
        state_path = os.path.join(path, 'model_state.json')

        if os.path.exists(model_path):
            # Model state: dict w/ persistent elements like current_epoch, saved persistently in additon to the keras model
            if os.path.exists(state_path):
                with open(state_path, 'r') as f:
                    state = json.loads(f.read())
                    self.current_epoch = state['current_epoch']
            else:
                print("Could not load model state ('%s' missing)"%state_path)

            print("Loading model from '%s'"%model_path)
            self.autoencoder = load_model(model_path)
            print("Loaded autoencoder:")
            self.autoencoder.summary()
            return True
        else:
            print("Can't load model from '%s', file does not exist"%path)
            return False

    def save (self, path=None):
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
            raise Exception("Cannot save, no path specified (self.autosave_path = %s)"%self.autosave_path)

        # Save keras model
        model_path = os.path.join(path, 'model.h5')
        makedirs(model_path)
        print("Saving as '%s'"%model_path)
        self.autoencoder.save(model_path)

        # Save additional persistent state (current_epoch, etc)
        state_path = os.path.join(path, 'model_state.json')
        with open(state_path, 'w') as f:
            f.write(json.dumps({ 
                'current_epoch': self.current_epoch,
            }))

    def build (self):
        """ Builds a new model. 

        Called automatically by Model's constructor iff autoload path set but there are no files to load from.
        Otherwise, you could disable autoload and call this explicitely to construct a new model.

        Additionally, if self.autosave_path is set this will autosave after constructing the model.
        """
        print("Building model")
        self.autoencoder = Sequential([
            Dense(self.hidden_layer_size, input_shape=(self.input_size,)),
            Activation('sigmoid'),
            Dense(self.encoding_size),
            Activation('sigmoid'),
            Dense(self.hidden_layer_size),
            Activation('linear'),
            Dense(self.input_size),
            Activation('linear')
        ])
        print("compiling...")
        self.autoencoder.compile(optimizer='adadelta', loss='mse')
        
        print("Loaded autoencoder:")
        self.autoencoder.summary()

        if self.autosave_path:
            self.save()

    def train (self, epochs, batch_size=32):
        
        enforce(self.autosave_frequency > 0, "autosave frequency must be > 0, got %s", self.autosave_frequency)
        enforce(batch_size > 0, "batch size must be > 0, got %s", batch_size)
        x_train, x_test = self.data

        """ Train model """
        print("Training model for %s epochs (epochs %s -> %s)"%(epochs, self.current_epoch, self.current_epoch + epochs))
        if self.model_snapshot_frequency and self.model_snapshot_path:
            next_snapshot = (self.current_epoch // self.model_snapshot_frequency + 1) * self.model_snapshot_frequency
            print("Next snapshot at epoch %s"%next_snapshot)
        else:
            next_snapshot = None

        last_saved_epoch = self.current_epoch

        while epochs > 0:
            print("Training on epoch %s -> %s"%(self.current_epoch, self.current_epoch + self.autosave_frequency))
            self.autoencoder.fit(x_train, x_train, epochs=self.autosave_frequency, batch_size=batch_size)

            epochs -= self.autosave_frequency
            self.current_epoch += self.autosave_frequency

            if next_snapshot and self.current_epoch >= next_snapshot:
                print("Saving snapshot at epoch %s"%(self.current_epoch))
                self.save(os.path.join(self.model_snapshot_path, str(self.current_epoch)))
                next_snapshot = (self.current_epoch // self.model_snapshot_frequency + 1) * self.model_snapshot_frequency
                print("Next snapshot at epoch %s"%next_snapshot)

            print("Autosaving...")
            self.save()
            last_saved_epoch = self.current_epoch

        if last_saved_epoch != self.current_epoch:
            print("Autosaving...")
            self.save()

DEFAULT_DATASET = 'https://raw.githubusercontent.com/SeijiEmery/shape-net-data/master/datasets/training-lv5.pkl'

class ArgumentParsingException (Exception):
    def __init__ (self, *args, **kwargs):
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
        if args['train']:
            num_epochs = parse_arg(int, '<num_epochs>', min_bound=1)

        data_url = args['--use-dataset'] or DEFAULT_DATASET
        model_path = args['--model']
        autosave_path = args['--autosave-path']
        autosave_freq = parse_arg(int, '--autosave', min_bound=0)
        snapshot_path = args['--snapshot-path']
        snapshot_freq = parse_arg(int, '--snapshot', min_bound=0)
        batch_size = parse_arg(int, '--batch-size', min_bound=1)
        train_test_split = parse_arg(float, '--train-test-split', min_bound=0.0, max_bound=1.0)
        # enforce_arg(os.path.exists(model_path), "model_path '%s' does not exist", model_path)
    except ArgumentParsingException as e:
        print("Invalid argument: %s"%e)
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

