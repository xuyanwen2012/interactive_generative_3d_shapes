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

I am still setting up this cli application right now. So use the following code for temporory access of the preprocessor. The OBJ file should placed under ```models```.

```
$ node index.js [filename]

# example:
# node index.js 1abeca7159db7ed9f200a72c9245aee7.obj
# => 
#    OBJLoader: 40.886ms
#    Processed 6138 vertices.
#    8797.551085
#    The file 1abeca7159db7ed9f200a72c9245aee7.json has been saved!

```

### OBJ Reconstruction

Still under developing. But you can see a sample at ```src/temp.js```.
    
## References 

<b>ShapeNet (2015)</b> [[Link]](https://www.shapenet.org/)
<br>3Million+ models and 4K+ categories. A dataset that is large in scale, well organized and richly annotated.
<br>ShapeNetCore [[Link]](http://shapenet.cs.stanford.edu/shrec16/): 51300 models for 55 categories.
<p align="center"><img width="50%" src="http://msavva.github.io/files/shapenet.png" /></p>

:gem: <b>Exploring Generative 3D Shapes Using Autoencoder Networks (Autodesk 2017)</b> [[Paper]](https://www.autodeskresearch.com/publications/exploring_generative_3d_shapes)
<p align="center"><img width="50%" src="https://github.com/timzhang642/3D-Machine-Learning/blob/master/imgs/Exploring%20Generative%203D%20Shapes%20Using%20Autoencoder%20Networks.jpeg" /></p>

