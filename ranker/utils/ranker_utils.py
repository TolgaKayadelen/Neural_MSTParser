import os
from parser.utils import load_models
from util import reader, writer
from ranker.training import model
from ranker.preprocessing import ranker_preprocessor


_RANKER_DATA_DIR = "./ranker/data"

def sort_by_reward(ranker_data_proto):
  pass

def n_low_top_rank_hypo(ranker_data_proto, lowest_score=0):
  """Returns the number and ratio of datapoints that has a top hypothesis with low score"""
  low_scoring_rank1s = 0
  better_hypo_in_hypos = 0
  for datapoint in ranker_data_proto.datapoint:
    hypotheses = datapoint.hypotheses
    if hypotheses[0].reward == lowest_score:
      low_scoring_rank1s += 1
      all_rewards = [h.reward for h in hypotheses]
      if max(all_rewards) > lowest_score:
        better_hypo_in_hypos += 1

  total_datapoints = len(ranker_data_proto.datapoint)
  ratio = low_scoring_rank1s / total_datapoints
  print(f"Total dataopints: {total_datapoints}")
  print(f"Total low scoring top hypotheses: {low_scoring_rank1s}")
  print(f"Datapoints with better scoring hypotheses: {better_hypo_in_hypos}")
  print(f"Ratio of low scoring top hypothesis: {ratio}")
  return low_scoring_rank1s, ratio

def rerank(ranker_data_proto, model_name, word_embeddings):
  ranker_prep = ranker_preprocessor.RankerPreprocessor(word_embeddings=word_embeddings)
  ranker = model.Ranker(word_embeddings=word_embeddings, from_disk=True, name=model_name)
  for datapoint in ranker_data_proto.datapoint:
    dataset = ranker_prep.make_dataset_from_generator([datapoint], batch_size=20)
    print("datapoint ", datapoint)
    print("dataset  ", dataset.get_single_element())
    input()
    scores = ranker.label_ranker(dataset.get_single_element(), training=False)
    print("scores ", scores)
    for hypo, score in zip(datapoint.hypotheses, scores):
      print("hypo ", hypo, "score ", score)
      hypo.reward = score


if __name__ == "__main__":
  datapoints_name = "tr_boun-ud-dev-k20-only-edges-dp.pbtxt"
  model_name="k20-edge-only-mse-error"
  ranker_data_proto = reader.ReadRankerTextProto(os.path.join(_RANKER_DATA_DIR, datapoints_name))
  # n_low_top_rank_hypo(ranker_data_proto=ranker_data_proto)
  word_embeddings = load_models.load_word_embeddings()
  rerank(ranker_data_proto, model_name, word_embeddings)


