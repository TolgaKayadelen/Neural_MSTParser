from collections import deque
import numpy as np
import tensorflow as tf

class Memory:
  """Sets up the memory replay buffer for a Deep Q Network.

  Args:
      memory_size (int): How many elements to hold in the memory buffer.
      batch_size (int): The number of elements to include in a replay batch.
  """
  def __init__(self, memory_size, batch_size):
    self.buffer = deque(maxlen=memory_size)
    self.memory_size = memory_size
    self.batch_size = batch_size

  def update(self, experiences):
    """Adds a list of experiences into the memory buffer."""
    self.buffer.extend(experiences)

  def __len__(self):
    return len(self.buffer)

  def __str__(self):
    return str(self.buffer)

  def filled(self):
    return len(self.buffer) == self.memory_size

  def random_sample(self):
    """Randomly selects datapoints from the replay memory buffer.

    Uniformally and randomly selects experiences to train the neural
    network on. Transposes the experiences to allow batch math on
    the experience components.

    Returns:
        (list): A list of lists with structure [
            [states], [actions], [rewards], [state_primes], [dones]
        ]
    """
    print("Sampling randomly..")
    # Buffer is the total number of elements we hold in memory.
    buffer_size = len(self.buffer)
    # Sampling is based on batch size. We randomly select n number of elements from
    # buffer for training. Here we get the indices only. Then based on these indices,
    # we pick states, actions, rewards and state_primes in these indices from the batch.
    sample_indexes = np.random.choice(
      np.arange(buffer_size), size=self.batch_size, replace=False)
    # print("sample_indexes ", sample_indexes)
    # input("press to cont.")

    # Columns have different data types, so numpy array would be awkward.
    # print("buffer ", self.buffer)
    # input("press to cont.")
    batch = np.array([self.buffer[i] for i in sample_indexes]).T.tolist()
    print("***********************")
    states = tf.convert_to_tensor([exp.state for exp in batch])
    # print("states ", states)
    # input("***********************")
    actions = tf.convert_to_tensor([exp.action for exp in batch])
    # print("actions ", actions)
    # input("***********************")
    rewards = tf.convert_to_tensor([exp.reward for exp in batch])
    # print("rewards ", rewards)
    # input("***********************")
    actions_hot = tf.convert_to_tensor([exp.action_hot for exp in batch])
    # print("actions one hot ", actions_hot)
    # input("***********************")
    action_names = tf.convert_to_tensor([exp.action_name for exp in batch])
    # print("action names ", action_names)
    # input("***********************")

    action_provenances = tf.convert_to_tensor([exp.action_provenance for exp in batch])

    return states, actions, rewards, actions_hot, action_names, action_provenances

  def weighted_sample(self):
    pass

  def targeted_sample(self):
    batch = [exp for exp in self.buffer if exp.label_correct == False and exp.edge_correct == False]
    states = tf.convert_to_tensor([exp.state for exp in batch])
    actions = tf.convert_to_tensor([exp.action for exp in batch])
    rewards = tf.convert_to_tensor([exp.reward for exp in batch])
    actions_hot = tf.convert_to_tensor([exp.action_hot for exp in batch])
    action_names = tf.convert_to_tensor([exp.action_name for exp in batch])
    action_provenances = tf.convert_to_tensor([exp.action_provenance for exp in batch])
    return states, actions, rewards, actions_hot, action_names, action_provenances

