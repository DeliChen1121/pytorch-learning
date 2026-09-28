# pytorch-learning

Personal notes and code from learning PyTorch.

I run most of this on a Mac (MPS) and sometimes on a shared GPU server. Notebooks live in `my_code/`. Papers I'm reading sit in `learning-materials/`.

## Notebooks (`my_code/`)

### `MNIST.ipynb`
Train a small MLP on MNIST end to end. Load data, train/eval loop, track loss and accuracy, plot history.

### `Imagenet_CIFAR10.ipynb`
Move from MNIST to CIFAR-10. Start with a flat MLP (it struggles), then switch to CNNs with data aug, validation split, and a learning-rate scheduler.

### `ResNet_learning.ipynb`
Build residual blocks by hand and train a small ResNet-style model on CIFAR-10. Focus is on how skip connections keep gradients flowing when the net gets deeper.

### `RNN_learning.ipynb`
Quick scratch pad for `nn.RNN` and `nn.LSTM`. Check tensor shapes and how hidden state carries info across a sequence.

There's also `pytorch-basics.ipynb` — tensors, broadcasting, reshape, and a tiny train loop before the bigger notebooks.

## Readings (`learning-materials/`)

PDFs I'm currently treating as fundamentals:

- `ML_fundamentals.pdf`
- `Batch Normalization- Accelerating Deep Network Training b y Reducing Internal Covariate Shift.pdf`
- `Deep Residual Learning for Image Recognition.pdf`
- `Attention_Is_All_You_Need.pdf`

## Tech note: env vars at the top of notebooks

Some notebooks set these **before** `import torch`:

```python
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

- **`KMP_DUPLICATE_LIB_OK`**: On Apple Silicon (Mac M-series), OpenMP can collide and crash the Jupyter kernel. This flag stops that crash.
- **`CUDA_VISIBLE_DEVICES`**: On shared GPU servers, pin which cards I can see. Here it exposes physical GPUs 0 and 1 only, so `cuda:0` in code maps to the first allowed card.

If you're only on CPU/MPS and not sharing a machine, you can ignore the CUDA line.
