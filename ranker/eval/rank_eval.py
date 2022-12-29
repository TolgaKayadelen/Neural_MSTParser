"""Evaluates ranking accuracy on ranked label hypothesis."""

from util import reader

def rank_accuracy(dataset):
  """Computes the number of cases where the top hypothesis is the one with the highest rank score (reward). """
  datapoints = 0
  correct_top_hypo, headroom = 0, 0
  for datapoint in dataset.datapoint:
    datapoints += 1
    ranks_and_rewards = [(h.rank, h.reward) for h in datapoint.hypotheses]
    # print("ranks and rewards ", ranks_and_rewards)
    highest_reward = max(ranks_and_rewards, key=lambda h: h[1])
    # print("highest reward ", highest_reward)
    # input()
    if highest_reward[0] == 1:
      correct_top_hypo += 1
    else:
      headroom += 1
  rank_acc = correct_top_hypo / datapoints
  headroom = headroom / datapoints
  return rank_acc, headroom, datapoints, correct_top_hypo

if __name__ == "__main__":
  dataset = reader.ReadRankerTextProto(path="./ranker/train_datapoints.pbtxt")
  rank_acc, headroom, total_tokens, correct_top_hypo = rank_accuracy(dataset)
  print("rank accuracy ", rank_acc)
  print("headroom ", headroom)
  print("total tokens ", total_tokens)
  print("tokens where top hypothesis has highest reward ", correct_top_hypo)
