# mlx_usage
trying out mlx framework
* MLX Framework: https://github.com/ml-explore/mlx

## Instruction substitution examples (torch -> mlx)
```python
* view -> reshape
x = x.view(new_x_shape)
->
x = mx.reshape(x, new_x_shape)

* permute(*args) -> transpose ([*args])
x.permute(0, 2, 1, 3)
->
x.transpose([0, 2, 1, 3])

* cat -> concatenate (dim -> axis)
torch.cat([a,b], dim = 2)
->
mx.concatenate([a,b], axis = 2)

* torch.nn.functional.softmax -> mx.softmax (dim -> axis)
torch.nn.functional.softmax(attention_scores, dim = -1)
->
mx.softmax(attention_scores, axis=-1)

## Activation Layers
* GELUActivation -> mlx.nn.gelu
transformers GeLU
# https://github.com/huggingface/transformers/blob/58e7f9bb2faf30622c9bead7adf472ac59f3d301/src/transformers/activations.py#L59C7-L59C21
->
mlx.nn.GELU()

* torch.nn.Tanh() -> mx.tanh
activation = torch.nn.Tanh()
activation(x)
->
activation = mx.tanh
activation(x)
```


## WIP - BERT
* [load_bert_test.ipynb](./load_bert_test.ipynb)
	* BERT code from transformers package
	* a lot of extensions were removed in WIP (ex. chunk feed forward, is_decoder checks, ..)

