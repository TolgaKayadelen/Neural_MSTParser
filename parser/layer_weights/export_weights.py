
import numpy as np
from parser.utils import load_models

def export(*, pretrained_model_name, layer_name, model_type):
  """Exports and saves layer weights from a pretrained model."""
  word_embeddings = load_models.load_word_embeddings()
  prep = load_models.load_preprocessor(word_embeddings=word_embeddings)
  if model_type == "parser":
    pretrained_model = load_models.load_parser(pretrained_model_name, prep)
  elif model_type == "labeler":
    pretrained_model = load_models.load_labeler(pretrained_model_name, prep)
  else:
    raise ValueError(f"Unsupported model type {model_type}")
  layer_weights = None
  for layer in pretrained_model.model.layers:
    if layer.name == layer_name:
      layer_weights = layer.get_weights()
      break
  print("Weights from pretrained model ", layer_weights)
  if layer_weights is None:
    sys.exit(f"Fatal: Layer weights with name {layer_name} not found in {pretrained_model_name}")
  np.save(f"./parser/layer_weights/{layer_name}_weights", layer_weights)
  print(f"{layer_name} weights are saved to dir.")


if __name__ == "__main__":
  export(pretrained_model_name="bilstm_labeler_topk",
         layer_name="pos_embeddings",
         model_type="labeler") # the mdoel type can be labeler or parser.