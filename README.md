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


### Offline neural net (autoencoder) training

To run:

    $ python3 autoencoder.py train <num-epochs>
    $ python3 autoencoder.py train <num-epochs> --autosave <autosave_frequency> --snapshot <snapshot_frequency>

By default, the latest keras model is saved to `model/model.h5` and snapshots are saved to `model/snapshots/<epoch>/model.h5`. You can run from past model with

    $ python3 autoencoder.py train <num-epochs> --model model/snapshots/<epoch>

The output directories, save frequencies, batch size and dataset source are all configurable via commandline arguments, see

    $ python3 autoencoder.py --help

## References 

<b>ShapeNet (2015)</b> [[Link]](https://www.shapenet.org/)
<br>3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated.
<br>ShapeNetCore [[Link]](http://shapenet.cs.stanford.edu/shrec16/): 51300 models for 55 categories.
<p align="center"><img width="50%" src="http://msavva.github.io/files/shapenet.png" /></p>

:gem: <b>Exploring Generative 3D Shapes Using Autoencoder Networks (Autodesk 2017)</b> [[Paper]](https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Exploring%20Generative%203D%20Shapes%20Using%20Autoencoder%20Networks.jpeg" /></p>

