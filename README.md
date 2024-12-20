# texrelenv: TexRel-like environments for emergent language experiments

A Python package for generating data sets based on [Hugh Perkins' TexRel](https://arxiv.org/abs/2105.12804).

![3 example images of the data generated by this library](example.png "Examples")

## Getting started

You can install this package from pypi using

```
pip install texrelenv
```

See example usage below for how to use the package to generate data.

## Example usage

Here is an example of how to generate 100 images based on some config options specified in a `config.toml` file.

Let's say you have a `config.toml` file that looks like the below. "grid" refers to the overall canvas of the image while "things" refers to the coloured objects that appear in images. The config options shown below are exhausted and are explained in the (**Config options**)[#Config options] section below

```
[grid]
grid_size = 16
hard_boundary = true
objects_can_overlap = false

[things]
thing_size = 4
distinct_shapes = 9
distinct_colours = 9
fix_colour = false
avoid_sharing_colours = true

[environment]
things_per_image = 5
rotate = false
flip = false

[split]
held_out_images = 0.2
held_out_shapes = 0
held_out_colours = 0
```

A dataset can be created from the above config in the following way

```
from texrelenv import DataSet

data = DataSet(config_file="config.toml")
```

You could also pass all the config arguments as arguments when you instantiate the `DataSet` (they are all named the same).

Having instantiated the data set, you can generate some train images like this:

```
data.sample(100, 'train')
```

Or some test images like this:

```
data.sample(100, 'test')
```

## Config options

| Option   | Type | Meaning|
|----------|------|--------|
| **grid_size** | int  | blah   |
| **hard_boundary** | bool  | blah   |
| **objects_can_overlap** | bool  | blah   |
| **thing_size** | int  | blah   |
| **distinct_shapes** | int  | blah   |
| **distinct_colours** | int  | blah   |
| **fix_colour** | bool  | blah   |
| **avoid_sharing_colours** | bool  | blah   |
| **things_per_image** | int  | blah   |
| **rotate** | bool  | blah   |
| **flip** | bool  | blah   |
| **held_out_images** | float  | blah   |
| **held_out_shapes** | float  | blah   |
| **held_out_colours** | float  | blah   |

## Contributing

To contribute to the project, please make sure you have poetry installed and before you start working on the code, set up a virtual environment, activate it and then run

```
poetry install
```

to install all dependencies including dependencies only required for development. Then update precommit hooks with

```
pre-commit install
pre-commit autoupdate
```

Pre-commit will help you to keep your code style in keeping with the rest of the project.

Please write tests for your code, store them in `test/`, and only commit code that passes the unit tests that already exist. You can run unit tests with

```
python -m unittest
```

(precommit will not run unit tests!)

## Acknowledgements

We are hugely thankful to Hugh Perkins for [coming up with the idea of TexRel](https://arxiv.org/abs/2105.12804) in the first place, for creating the first version of PyTorch, and for doing various other cool stuff.
