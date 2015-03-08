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
- Completely parallel
- Easy debugging by saving masks and intermediate results as images

## Dependencies

python, numpy, scipy

## Textures

My textures are too heavy to put on Github. To use them in the program, you need to create a `textures` folder and put a `bgedit.png` texture for the sky (a skysphere, x = lon, y = lat) and an `adisk.jpg` for the accretion disk (x = angle, y = radius).

Texture sizes are irrelevant (these texture will never end up in the GPU). You can use your own texture names or formats by editing the source.

Texture files are not needed if rendering with modes that don't use them.

## Installation

Clone with git:

```
git clone https://github.com/rantonels/starless.git
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

## Writing .scene files

Please refer to the `scenes/default.scene` file for a commented overview of all options.
