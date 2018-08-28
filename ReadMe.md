# MLDisasm

## Introduction

MLDisasm is a machine learning model trained to simulate the behaviour of a disassembler, i.e. it converts machine code into human-readable assembly code. MLDisasm is based around the long short-term memory (LSTM) cell, a type of recurrent neural network.

So far, MLDisasm has been trained to generate x86 assembly using both the AT&T and Intel syntaxes, and could be extended to other assembly languages, such as ARM, in the future.

A current limitation of the disassembler is that it is unable to find the boundaries of instructions on its own: x86 instructions may be 2 to 15 bytes long, and there may be many equally valid interpretations of any 15 byte block. This problem will be addressed in the next version (0.2).

The following is proposed as a solution:

* Given a 15 byte block, use a sliding window to produce every possible combination i.e. disassemble bytes 1-2, 1-3, ..., 1-15, and pass the results to another model which chooses the best interpretation.

## Dependencies

### Training Set Generation

The following are needed to generate the training set:

* `GNU` coreutils (`bash`, etc.)

* `GNU` binutils (`objdump`, for AT&T syntax or non-Intel assembly languages)

* `NASM` (`ndisasm`, for Intel syntax)

* `Python 3.6`

### Training and Running the model

The following were used to run the model.

#### Necessary dependencies

The following are vital and the program won't run without them. Different versions may work though. These packages can all be installed using the Python pacakge manager, `pip`.

* `Python 3.6`

* `tensorflow-gpu 1.10.0` &ndash; machine learning backend (NB: there is a patch to be applied (see next section) which only applies to `TensorFlow 1.10.0`, consequently other versions of TensorFlow **will not work**)

* `numpy 1.14.5` &ndash; for CPU-based mathematics

* `h5py 2.8.0` &ndash; for saving and loading Keras models

#### Optional dependencies

The following are optional but recommended: the program will run without them but might not perform as well. Different versions might work. These packages can also be installed with `pip`.

* `psutil 5.4.6` &ndash; for profiling memory during execution

* `colorama 0.3.9` &ndash; for formatted log output

* `ujson 1.35` &ndash; for better I/O performance

## Configuration

MLDisasm is configured using a JSON file stored in `data/config.json`.

Here is an example configuration file:

`data/config.json`

```json
{
    // The maximum number of records to use during training.
    "max_records":          2000000,
    // The maximum number of records to use during gridsearch.
    "gs_records":           10000,
    // The length of a sequence. Sequences
    "seq_len":              50,
    // Mask value: input vectors which contain only this value will be skipped.
    "mask_value":           -1,
    // Model configuration. Parameters here can be overridden by the "grid" object.
    "model": {
        // Whether to use a GRU (true) or LSTM (false) for recurrent units.
        "gru_mode":         false,
        // Metrics to measure training progress by. Currently only "accuracy" is supported.
        "metrics":          ["accuracy"],
        // How many records to train on at once. Larger values are more efficient (up to a point, dependent on the size
        // of video memory) but smaller values (again, to a point) may be better for training.
        "batch_size":       100,
        // How many training epochs to use. Small values are recommended because the training set can be very large.
        "epochs":           10,
        // How many units in each LSTM or GRU layer.
        "hidden_size":      256,
        // Dimensionality of the output. Currently ignored -- the value is computed internally.
        "output_size":      1148,
        // How many LSTM or GRU layers to use.
        "lstm_layers":      1,
        // The LSTM activation function (tanh is recommended).
        "lstm_activation":  "tanh",
        // Whether to use bias vectors (recommended).
        "lstm_use_bias":    true,
        // Whether to use bias in the LSTM forget gate (recommended).
        "lstm_forget_bias": true,
        // Whether to use dropout in the output to prevent overfitting (very, very small values recommended).
        "lstm_dropout":     0.000001,
        // Whether to use dropout in the hidden state to prevent overfitting (very, very small values recommended).
        "lstm_r_dropout":   0,
        // What activation function to use in the dense layer, which maps from LSTM space to the output space (sigmoid
        // recommended).
        "dense_activation": "sigmoid",
        // What loss function to use (categorical cross-entropy recommended).
        "loss":             "categorical_crossentropy",
        // Whether to use a softmax layer (recommended).
        "use_softmax":      true,
        // What optimizer to use.
        "optimizer":        "Adadelta",
        // Parameters for the optimizer, such as learning rate (lr).
        "opt_params":{
            "lr":           1.0
        },
        // Whether to use the mask_value.
        "use_masking":      true
    },
    // Override parameter values above during gridsearch. Values must be arrays. Each combination of the values below
    // will be searched which is why it's often better to search unrelated parameters separately.
    "grid": {
        "hidden_size": [64,128,256,512],
        "lstm_layers": [1,2,3]
    }
}
```

Here is an example of an out-of-line grid (see "Select hyperparameters without training"):

`data/grids/00_hidden_size.json`

```json
{
    "hidden_size": [64,128,256,512],
    "lstm_layers": [1,2,3]
}
```

## Invocation

### Patching TensorFlow

As of version 1.10.0, TensorFlow and its implementation of Keras contain a bug which affects Python 3.6. If TensorFlow is to be installed from source, apply the patch as follows:

```shell
export PATCH_PATH="$(pwd)/patches/tensorflow.patch"
cd /path/to/tensorflow
git apply "${PATCH_PATH}"
```

(Replace paths as appropriate)

If TensorFlow is already installed (e.g. via pip), locate its directory in the Python site-packages directory, apply the patch like this:

```shell
export PATCH_PATH="$(pwd)/patches/tensorflow.patch"
cd ~/.local/lib/python3.6/site-packages/tensorflow
git apply "${PATCH_PATH}"
```

(Replace paths as appropriate)

### Generating a training set

Use the following command to generate a training set with the OBJDUMP disassembler:

```shell
tools/gen-train-auto -o <model name>
```

Use the following command to generate a training set with the NDISASM disassembler:

```shell
tools/gen-train-auto -n <model name>
```

### Automate the full training process

Use the following command to run the three stages of the training process (hyperparameter search, training and validation):

```shell
./autotrain <model name>
```

Note: If you don't have grids for gridsearch, see the next section for how to create them.

### Select hyperparameters without training

First create the file `data/config.json` with some default values. You can then create a parameter grid. This can be done in two ways.

You can skip this step if you are comfortable using the default values provided.

#### 1. Whole grid in one file with `tune`

Write the parameter grid in `data/config.json` and run the following command:

```shell
./tune <model name>
```

This is useful when you have only a few parameters to search, or all your parameters are related and should be searched in one go.

#### 2. Separate grids with `autotune`

Create `.json` files (with the executable permission) in a directory called `data/grids`. The contents of executable `.json` files will be inserted into `data/config.json` (each file must therefore contain a valid JSON object) one at a time and used as a parameter grid. The grids will be searched in lexicographical order, so `00_batch_size.json` is searched before `01_hidden_size.json`.

After creating grids, tune the configuration using the following command:

```shell
./autotune <model name>
```

This is useful when there are many, unrelated parameters to search, and searching every combination of them at once would take too long.

### Train the model

After selecting hyperparameters you can train the model using the following command:

```shell
./train <model name>
```

### Validate the model

After training, you can validate the model on unseen data using the following command:

```shell
python src/validator.py <model name>
```

### Run unit tests and benchmarks

You can run unit tests using the pre-commit hook:

```shell
./tools/pre-commit
```

You can run benchmarks using the following command:

```shell
python3 src/bench-runner.py
```

## License

The program is licensed under the terms of the MIT license. See the file `COPYING` for details.
