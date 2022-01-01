# Transformer Implemented in Tensorflow 2 (2.7)

## Requirements

```shell
pip install hydra-core==1.1.1 tensorflow==2.7.0 tensorflow-text==2.7.3
```

CUDA/CuDNN is optional as the model is relatively small.

## Problem

The Transformer in this repository aims at solving algebraic problems within 1000 and with only addition and subtraction.

The problem is setup in an unstructured style, which can be potentially challenging and be a good experiment for Transformer. Some examples are like:

```
Q: 7 2 6 subtract 2 4 7 equals 4 0 8
A: Wrong. It"s four hundred and seventy nine

Q: 3 6 8 add 4 8 0 equals 8 4 8
A: Correct. It"s eight hundred and forty eight
```

The question is in digits, and it makes a statement about `A +/- B = C`.

The answer firstly gives whether the statement is correct, then it gives the correct answer in English expression.

The input is simply tokenized by splitting by white space.

## Run

The configuration files of dataset, model and training are in `cfg/`, which is parsed by [Hydra](https://hydra.cc/).

### Generate Dataset

In the default setting, `100,000` pairs for training data and `10,000` pairs for test data.

- Wrong and correct samples are evenly distributed.

- Subtract and addition are evenly distributed.
- All numbers in the statement are greater than or equal to zero and less than 1000.

Generate the dataset by:

```
python gen_dataset.py
```

### Train

The default setting for the model and the training hyper-parameters:

```
d_model: 64
num_layers: 2
num_heads: 4
dff: 256
```

```
batch_size: 512
epoch: 50
```

The meanings of the parameters can be found in the paper of Transformer.

Run the training by:

```
python train.py
```

The models are saved in `checkpoints/` by default.

### Evaluation

Two evaluation metrics for the results are used here:

1. Accuracy of `Wrong` and `Correct`.
2. Accuracy of the exact whole answers.

The results are simply inferred by greedily selecting the token with highest probability in each step.

> The first step is limited in `Correct.` and `Wrong.`, and the second step is limited in `It"s`.
>
> The performance can be improved by using beam search, but it's not the point for this repository.

The results on the metrics are decent:

```
Accuracy of Correctness: 0.9693333333333334
Accuracy of Exact Match: 0.9686666666666667
```

Run the evaluation by:

```
python evaluation.py
```

## Reference

1. Tutorial of Transformer: https://tensorflow.google.cn/tutorials/text/transformer
2. Paper of Transformer: https://arxiv.org/abs/1706.03762

## Appendix

### Visualize Attention Weights

### Training Logs





