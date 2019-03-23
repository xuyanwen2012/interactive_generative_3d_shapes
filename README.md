# Interactive Generative 3D Shapes
> A Three.js implementation of ShrinkWrap algorithm for 3D shape parameterization. 


## Install

    npm install
    
## Usage

Currently the project is set in a wired structure, which are subjects to refactor. There are actually two programs.

### Visualization tool

Use the following comand to run WebPack development mode for visualizations.

```
$ npm run start
```
### OBJ Preprocessor

#### Processing models:

To process models (obj => shrinkwrap params), run

    $ node index.js process <input-model-directory> <output-directory>

There are some nice options, like `--limit <N>` (only runs the processor on N files), `--levels <N>` (set subdivision levels), and `--rebuild` (by default, the preprocessor will skip reprocessing files that already exist in the output directory, but this can be overridden with `--rebuild`). 

For the full list, run `node index.js process --help`. 

Warning: `--workers` is currently broken and should not be used.

#### Reconstructing models:

To reconstruct models (shrinkwrap params => obj), run

    $ node index.js reconstruct <input-param-directory> <output-obj-directory>

Again, there are several options you can list using `node index.js reconstruct --help`.

#### Other options:

    $ node index.js info <directory> --iext <input-file-ext> --oext <output-file-ext>

can be used to list all the files in a directory w/ a matching file extension (and corresponding output files w/ output file extensions), and exists for development and debugging purposes.

    $ node index.js view

currently does not do anything. To launch the visualization tool run `npm start`.

## Running offline training / etc with autoencoder.py

#### Run this first:

	chmod +x autoencoder.py
	npm install
	
All commands have the following format

	./autoencoder.py <command> <args...>
	
### Training

To train the autoencoder (this will automatically create a new model, or train from an existing model):

	./autoencoder.py train <num-epochs>

Where <num-epochs> is the number of epochs that the model will train for. ie if the model has been trained for 100 epochs, training it for 50 will train it from 100 -> 150 epochs.

#### Autosaving

By default, the model is autosaved to `model`, and is autosaved to `<autosave-path>/model.h5` and `<autosave-path>/model_state.json` every `<autosave-frequency>` epochs. It's fine to kill the training process (so long as it isn't in the middle of autosaving), as it will by default just restore from the latest autosave. You can change the autosave directory, and the autosave frequency, with the arguments

	--autosave-path <autosave-path>
	--autosave <autosave-frequency>	
	
By defaut the path is `model` and the frequency is `10`. The frequency is the frequency that it saves between epochs (10 => will save the model state every 10 epochs).

#### Snapshots

The model also saves snapshots every n epochs into a separate directory. By default this is `model/snapshots`, and if you change autosave-path you will need to set this explicitly (and should be set to `<model-path>/snapshots`). You can change this with

	--snapshot <snapshot_frequency>
	--snapshot-path <snapshot_path>


#### Loading from another model directory

To load from a specific model, use

	--model <model_path>
	
Note that this only specifies that you want to -load- from that path and will need to set autosave-path and snapshot-path if you want to write back to it. Mostly, this is useful to resume training from snapshots, ie `--model model/snapshots/20`, and can be used for most other commands.

#### Other options

You can set the dataset source (expects a URL) with

	--use-dataset <dataset-path>
	
By default, this is https://raw.githubusercontent.com/SeijiEmery/shape-net-data/master/datasets/training-lv5.pkl, and should be a pickled dictionary with two entries: `data`, a 2d numpy array of `samples (from obj files)` x `data (see our parameterization)`, and `keys`, a list of the source files corresponding to data. `len(keys)` should match `# samples`, ie `data.shape[0]`. If for whatever reason you wanted to use another dataset, there is a bunch of validation code that will check all of this stuff for you, and emit helpful error messages if anything is incorrect.

Lastly, you can set the train / test split with

	--train-test-split <split_ratio>
	
This is expected to be a value on (0, 1), with eg. 0.8 => 80% train, 20% test.

### Summarize (get logs)

	./autoencoder.py summarize-runs <model-path>
	
This will summarize each snapshot in `<model-path>/snapshots`, and build a combined .csv summary in `summary/<model-path-name>.csv`. This gives you a lot of data, including distribution info (mean, min, max, variance) for all inputs / outputs. You can visualize this with `visualize_summaries.ipynb`.

This feature was added somewhat late, and, as with most features here, was a bit of a hack. Summarization was built in a backwards compatible way, ie. summary data for each epoch is built and cached into model directories as `<model-path>/snapshots/**/model_summary.json`. When this is present, it's read, and when not present, the respective `model.h5` file is loaded and a summary is generated from that, then saved. To... complicate things, a nice 'feature' was added so that while snapshots are only saved every epochs, snapshot summaries are saved (to snapshot directories) whenever the model autosaves. Note that this was because the way that the model autosaves was also a bit of a hack – for details, see the implementation of AutoencoderModel.train(). TLDR; summary datapoints are limited by the autosave frequency, and thus if you want datapoints every epoch you must set this to 1. Obviously this will make the model train slower and has a lot of overhead. A refactor was planned to fix all this stuff, but was cancelled (ran out of time). What we have still fundamentally works, but is just a bit inefficient and can be somewhat annoying to work with (you must pass --model <path> everywhere to commands (see below), etc).

You can find a sample summary at <https://github.com/SeijiEmery/shape-net-data/tree/master/archived_models/mse-relu-epoch-633-summary>

### Evaluating the model

#### Repredict (ie. use the trained autoencoder to reproduce N test + N train obj meshes)

	./autoencoder.py repredict <output-directory> <num-output-train-and-test-meshes> [--model <path-to-model-snapshot>]

You can also run

	./autoencoder.py repredict snapshot <epoch> <num-outputs>
	
Which is equivalent to

	./autoencoder.py repredict repredicted <num-outputs> --model model/snapshots/<epoch>
	
Note that this command (and all of the following commands) internally runs `node index.js reconstruct <data-path> <output-mesh-path>` many times to actually reconstruct the obj models. If you're getting errors you probably need to run `npm install`.
	
#### Gen-random (ie. use the trained decoder to build N random models by sampling randomly from the encoder's latent space)

	./autoencoder.py gen-random <output-directory> <num-outputs> [--model <path-to-model-snapshot>]
	
#### Generate mesh representations of encoding values / latent space
	
	./autoencoder.py gen-latent-models <output> [--model <path-to-model-snapshot>]
	
#### Linearlly combine existing models

	./autoencoder.py interpolate <key1 (from)> <key2 (to)> <interp>
	./autoencoder.py add-features <key1 (add features to)> <key2 (add features from)>
	./autoencoder.py remix <key1 (original)> <key2 (add features)> <key3 (subtract features)>
	
Keys correspond to model names in the source dataset. You can list all of them with

	./autoencoder.py list-keys
	
All of these will do some variation of 'add / subtract features of rebuilt obj models through autoencoder latent space'. Interpolate will morph from one model to another. `<interp>` specifies the step size, ie. 0.1 will step from 0.0, 0.1, 0.2, ..., 0.9, 1.0, and generate obj meshes for each of those. This works, and works as well as `repredict` does for the models you're passing in.

`add-features` will implicitely do that (using 0.1 interp steps), but just instead of interpolating will just add one model's features to another. This was experimental and does not work for the model we've trained (ie. does not produce recognizable car models).

`remix` will do the same, but adding one model's features and subtracting another's. This was an improvement on `add-features`, and does work. Sort of. You can see how well it works for yourself.

#### Data repository

You can find samples of all of the above at <https://github.com/SeijiEmery/shape-net-data/tree/master/archived_models>. ie. you can download a trained model from there and evaluate and run the above commands on it.

## References 

<b>ShapeNet (2015)</b> [[Link]](https://www.shapenet.org/)
<br>3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated.
<br>ShapeNetCore [[Link]](http://shapenet.cs.stanford.edu/shrec16/): 51300 models for 55 categories.
<p align="center"><img width="50%" src="http://msavva.github.io/files/shapenet.png" /></p>

:gem: <b>Exploring Generative 3D Shapes Using Autoencoder Networks (Autodesk 2017)</b> [[Paper]](https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Exploring%20Generative%203D%20Shapes%20Using%20Autoencoder%20Networks.jpeg" /></p>