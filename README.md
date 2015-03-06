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

## How to run

Create the `tests` folder, if it doesn't exist yet.

Write a .scene file. Examine the provided .scenes and model your file on them.

Some options will lie under the 'lofi' and 'hifi' sections. This are respectively the rendering specs for tests and the final image.

To run a test:

```
$ python tracer.py -d yourscene.scene
```

and wait. The rendered image will be in the `tests` folder under the name `out.png`, along with some other useful images.

To run the full render, just omit the `-d` option. The results will still be saved in `test`.
