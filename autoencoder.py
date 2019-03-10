#!/usr/bin/env python3
from urllib.request import urlopen
import pickle as pkl
import numpy as np
import json
import sys
import os
import argparse
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
from keras.models import Model, load_model


""" Helper functions """
def makedirs (path):
    """ Builds out all directories for an arbitrary file path. Call this before writing files. """
    basedir, file = os.path.split(path)
    if basedir and not os.path.exists(basedir):
        os.makedirs(basedir)

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


class AutoencoderModel:
    def __init__ (
            self, 
            autoload_path='model', 
            autosave_path='model', 
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
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.encoding_size = encoding_size
        self.autosave_path = autosave_path
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

        # Build directories if they don't exist
        makedirs(path)

        # Save keras model
        print("Saving as '%s'"%model_path)
        model_path = os.path.join(path, 'model.h5')
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
        
    
if __name__ == '__main__':
    dataset = load_dataset('https://raw.githubusercontent.com/SeijiEmery/shape-net-data/master/datasets/training-lv5.pkl')
    print("shape: %s"%(dataset['data'].shape,))
    print("keys: %s"%(len(dataset['keys'])))

    autoencoder = AutoencoderModel()
