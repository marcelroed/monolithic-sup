# Monolithic Sup

Mojo is great for inference, but currently lacks support for training the LLMs it runs so well. Well.. not anymore! Say hello to Monolithic Sup! Not quite Modular Max (but kind of close).

It is a growing collection of fused backward-forward kernels implemented in Mojo (no CUDA) like cross entropy, linear, AdamW.. (and some backward implementations using the Max graph api, like attention).

### Getting started

```bash
magic run python train_loop.py
```