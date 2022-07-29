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
    """Adds a list of experiences into the memory buffer.

    Args:
        experience: a (state, action_name, reward, action_one_hot) tuple.
    """
    self.buffer.extend(experiences)

  def __len__(self):
    return len(self.buffer)

  def filled(self):
    return len(self.buffer) == self.memory_size

  def random_sample(self):
    """Uniformally selects from the replay memory buffer.

    Uniformally and randomly selects experiences to train the nueral
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
    index = np.random.choice(
      np.arange(buffer_size), size=self.batch_size, replace=False)
    # print("index ", index)
    # input("press to cont.")

    # Columns have different data types, so numpy array would be awkward.
    # print("buffer ", self.buffer)
    # input("press to cont.")
    batch = np.array([self.buffer[i] for i in index]).T.tolist()
    # print("***************")
    # print("batch ", batch)
    # print("length of batch ", len(batch))
    # input("press to cont.")
    states = tf.convert_to_tensor([exp.state for exp in batch])
    # print("states ", states)
    actions = tf.convert_to_tensor([exp.action for exp in batch])
    # print("actions ", actions)
    # input("press to cont")
    # actions = np.array(batch[1], dtype=np.int8)
    # rewards = np.array(batch[2], dtype=np.float32)
    action_qs = tf.convert_to_tensor([exp.action_qs for exp in batch])
    # print("action qs ", action_qs)
    rewards = tf.convert_to_tensor([exp.reward for exp in batch])
    # print("rewards ", rewards)
    # actions_one_hot = tf.one_hot(actions, output_size)
    actions_one_hot = tf.convert_to_tensor([exp.action_hot for exp in batch])
    # print("actions one hot ", actions_one_hot)
    action_names = tf.convert_to_tensor([exp.action_name for exp in batch])
    # input("press to cont.")
    return states, actions, action_qs, rewards, actions_one_hot, action_names

  def weighted_sample(self):
    pass

  def targeted_sample(self):
    pass
