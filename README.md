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
- Bloom postprocessing
- Completely parallel - renders chunks of the image using numpy arrays arithmetic
- Multicore (with multiprocessing)
- Easy debugging by saving masks and intermediate results as images

## (Possible) future features

- Non-stationary observers, aberration, render from inside the event horizon
- Kerr geometry (rotating black holes with frame dragging)
- Integration of time variable, time dependent objects (e.g. orbiting sphere with retarded time rendering) with worldtube-null geodesic intersection
- Generic redshift framework for solid colour objects

## Dependencies

python, numpy, scipy

## Textures

**a default `adisk.jpg` is now included.**

My sky texture is too heavy to put on Github. To use it in the program, you need to place a `bgedit.png` texture in the `textures/` folder for the sky (a skysphere, x = lon, y = lat) and optionally replace `adisk.jpg` for the accretion disk (x = angle, y = radius).

[Example adisk.jpg](http://i.imgur.com/eUR6ytQ.jpg)

[Example bgedit.png](http://svs.gsfc.nasa.gov/vis/a000000/a003500/a003572/TychoSkymapII.t5_04096x02048.jpg)

Texture sizes are irrelevant (these texture will never end up in the GPU). You can use your own texture names or formats by editing the source.

Texture files are not needed if rendering with modes that don't use them.

## Installation

### Linux

Download the [latest release](https://github.com/rantonels/starless/releases).

If you want the absolute bleeding edge version, clone with git:

```
$ git clone https://github.com/rantonels/starless.git
```

I make absolutely no guarantees about the stability of anything, but generally you should be safer with the release.

### Windows

Hilarious.

## How to run

Write a .scene file. Examine the provided .scenes and model your file on them.

Some options will lie under the 'lofi' and 'hifi' sections. This are respectively the rendering specs for tests and the final image.

To run a test:

```
$ python tracer.py -d yourscene.scene
```

and wait. The rendered image will be in the `tests` folder under the name `out.png`, along with some other useful images.

To run the full render, just omit the `-d` option. The results will still be saved in `tests`.

## Multiprocessing

The raytracer is multicore. It partitions the viewport in chunks, then partitions again the list of chunks to distribute them to processes. By default, 4 processes are created. This number can and should be changed with the option `-jN`. 

Best results should arise when the number of processes equals the number of cores of the machine, or that number minus one. However, your mileage may vary - experiment with different values of `-j` and see what works best for you. It's entirely possible you don't get any speedup from multiprocessing; in that case use `-j1` to avoid overhead.

Gains might be modest. As an example, a 4 min render on single core runs in ~2 min 40 seconds using 7 cores on my 8 cores laptop.

## Command line usage

`tracer.py` accepts the following command line options:

`-d`: run test (render with [lofi] settings)

`-o` or `--no-bs`: a shortcut for the `--no-display`, `--no-shuffle`, `--no-graph` options. Disables all gadgets to concentrate on performance. Useful for heavy renders.

`--no-display`: do not open matplotlib preview window. This is a **huge improvement** in speed for large images.

`--no-shuffle`: do not shuffle pixels before chunking. This, in practice, means that instead of being rendered as a gradually densening cloud, the image is raytraced progressively in linear chunks (though the order of the chunks themselves is still shuffled). The end result is identical, but with shuffling the preview window might give an idea of the render sooner. However, **disabling shuffling** provides a nice speed improvement for larger images, because:

1) copying rendered data to the large final image buffer is much faster if it's contiguous
2) per chunk, the raytracer performs full calculations relative to an object (disc, horizon, etc) if and only if at least one ray of it its the object. So, if chunks are actually contiguous, there is a certain probability that some of them will never hit certain objects and many computations will be skipped. Shuffled chunks almost surely hit every object in the scene.

`-jN`: use N processes (tip: use N = number of cores). Default is 4.

`--no-graph`: By default Starless generates a matplotlib schematic graph, `graph.png`, of the sizes and position of the black hole, the accretion disc and the camera projected on the z-x plane (this is still a work in progress, and not complete). This option disables the generation of the graph.

The (single) scene filename can be placed anywhere on the command string, and is recognized as such if it doesn't start with the `-` character. If omitted, `scenes/default.scene` is rendered.

## Writing .scene files

Please refer to the `scenes/default.scene` file for a commented overview of all options.

Many options have default values and can be omitted, but I make no guarantees on the existence and values of such defaults. To be sure, include all options from `scenes/default.scene`.

Some general rules:

* Precision and render time are obviously massively affected by the `Iterations` and `Stepsize` options. As a rule of thumb, in `[lofi]` fix `Stepsize` to a large value, such as `0.08`, then render multiple times with `-d` progressively decreasing `Iterations`. You need to find the smallest value of `Iterations` for which no significant parts of the image are clipped. When you've fixed the `Iterations` number, decrease the step size to a desirable value (`0.02` is basically high-quality, `0.04` or `0.08` is enough for most purposes) *and simultaneously increase* `Iterations` *by the same factor* (e.g.: halve step size, double iterations) so as to keep clipping distance **approximately fixed**. When satisfied, copy these values to the `[hifi]` section. This procedure minimizes render time.
