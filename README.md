# [Python Black Hole Raytracer](http://spiro.fisica.unipd.it/~antonell/starless/)

Starless is a CPU black hole raytracer in numpy suitable for both informative diagrams and decent wallpaper material.

It is still in development and will be presented on my site along with [my real time black hole visualizer](http://spiro.fisica.unipd.it/~antonell/schwarzschild). Starless, instead, is not real time but offers greater functionality, performing actual raytracing (while the mentioned WebGL applet uses a precomputed lookup texture for deflection).

## Features

- Full geodesic raytracing in Schwarzschild geometry
- Predicts distortion of arbitrary objects defined implicitly
- Alpha-blended accretion disk
- Sky distortion
- Dust
- Bloom postprocessing
- Completely parallel - renders chunks of the image using numpy arrays arithmetic
- Easy debugging by saving masks and intermediate results as images

## Dependencies

python, numpy, scipy

## Textures

My textures are too heavy to put on Github. To use them in the program, you need to create a `textures` folder and put a `bgedit.png` texture for the sky (a skysphere, x = lon, y = lat) and an `adisk.jpg` for the accretion disk (x = angle, y = radius).

[Example adisk.jpg](http://i.imgur.com/eUR6ytQ.jpg)
[Example bgedit.png](http://svs.gsfc.nasa.gov/vis/a000000/a003500/a003572/TychoSkymapII.t5_04096x02048.jpg)

Texture sizes are irrelevant (these texture will never end up in the GPU). You can use your own texture names or formats by editing the source.

Texture files are not needed if rendering with modes that don't use them.

## Installation

Clone with git:

```
$ git clone https://github.com/rantonels/starless.git
```

**or** download this [zip file](https://github.com/rantonels/starless/archive/master.zip) and extract.

## How to run

Write a .scene file. Examine the provided .scenes and model your file on them.

Some options will lie under the 'lofi' and 'hifi' sections. This are respectively the rendering specs for tests and the final image.

To run a test:

```
$ python tracer.py -d yourscene.scene
```

and wait. The rendered image will be in the `tests` folder under the name `out.png`, along with some other useful images.

To run the full render, just omit the `-d` option. The results will still be saved in `tests`.

## Command line usage

`tracer.py` accepts the following command line options:

`-d`: run test (render with [lofi] settings)

`--no-display`: do not open matplotlib preview window.

`--no-shuffle`: do not shuffle pixels before chunking. This, in practice, means that instead of being rendered as a gradually densening cloud, the image is raytraced progressively starting from the top. The end result is identical, but with shuffling the preview window might give an idea of the render sooner. However, **disabling shuffling** provides a nice speed improvement for larger images, because:

1) copying rendered data to the large final image buffer is slightly faster if it's contiguous
2) per chunk, the raytracer performs full calculations relative to an object (disc, horizon, etc) if and only if at least one ray of it its the object. So, if chunks are actually contiguous, there is a certain probability that some of them will never hit certain objects and many computations will be skipped. Shuffled chunks almost surely hit every object in the scene.

The (single) scene filename can be placed anywhere, and is recognized as such if it doesn't start with the `-` character. If omitted, `scenes/default.scene` is rendered.

## Writing .scene files

Please refer to the `scenes/default.scene` file for a commented overview of all options.

Many options have default values and can be omitted, but I make no guarantees on the existence and values of such defaults. To be sure, include all options from `scenes/default.scene`.
