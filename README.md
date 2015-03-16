# [Python Black Hole Raytracer](http://rantonels.github.io/starless)

Starless is a CPU black hole raytracer in numpy suitable for both informative diagrams and decent wallpaper material.

It is still in development and will be presented on my site along with [my real time black hole visualizer](http://spiro.fisica.unipd.it/~antonell/schwarzschild). Starless, instead, is not real time but offers greater functionality, performing actual raytracing (while the mentioned WebGL applet uses a precomputed lookup texture for deflection).

## Features

- Full geodesic raytracing in Schwarzschild geometry
- Predicts distortion of arbitrary objects defined implicitly
- Alpha-blended accretion disk
- Optional blackbody mode for accretion disk with realistic redshift (doppler + gravitational)
- Sky distortion
- Dust
- Bloom postprocessing by Airy disk convolution with spectral dependence
- Completely parallel - renders chunks of the image using numpy arrays arithmetic
- Multicore (with multiprocessing)
- Easy debugging by saving masks and intermediate results as images

## (Possible) future features

- Non-stationary observers, aberration, render from inside the event horizon
- Kerr geometry (rotating black holes with frame dragging)
- Integration of time variable, time dependent objects (e.g. orbiting sphere with retarded time rendering) with worldtube-null geodesic intersection
- Generic redshift framework for solid colour objects

## Installation and usage

Please refer to the [Wiki](https://github.com/rantonels/starless/wiki).
