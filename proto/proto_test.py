from proto import metrics_pb2
from util.nn import nn_utils
from util import reader
import matplotlib.pyplot as plt
import numpy as np
import random

def plot_metrics(*, name: str, metrics: metrics_pb2.Metrics, plot_losses=False):
  """Plots the metrics from a parser run."""
  metrics = reader.ReadMetricsTextProto(metrics)
  fig = plt.figure()
  ax = plt.axes()
  loss_metrics = ["edge_loss_padded", "label_loss_padded"] 
  colors = ["b", "g", "r", "c", "m", "y", "sienna", "orchid", "k"]
  for key in metrics.metric:
    if key in loss_metrics and not plot_losses:
      continue
    if metrics.metric[key].tracked:
      color = random.choice(colors)
      colors.remove(color)
      ax.plot(np.arange(len(metrics.metric[key].value_list.value)),
              metrics.metric[key].value_list.value,
              "-g", label=key, color=color)

  plt.title("Neural MST Parser Performance")
  plt.xlabel("epochs")
  plt.ylabel("accuracy")
  plt.legend()
  plt.savefig(f"test_metrics_plot")


def main():
  plot_metrics(name="test_plot", metrics="./parser/nn/metrics.pbtxt")

if __name__ == "__main__":
  main()


