
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, models
from collections import deque


def deep_q_network(state_shape, action_size, learning_rate, hidden_neurons):
  """Creates a Deep Q Network to emulate Q-learning.

  Creates a two hidden-layer Deep Q Network. Similar to a typical neural
  network, the loss function is altered to reduce the difference between
  predicted Q-values and Target Q-values.

  Args:
      state_shape: a tuple of ints representing the observation space.
      action_size (int): the number of possible actions.
      learning_rate (float): the nueral network's learning rate.
      hidden_neurons (int): the number of neurons to use per hidden
          layer.
  """
  state_input = layers.Input(state_shape, name='frames')
  actions_input = layers.Input((action_size,), name='mask')

  hidden_1 = layers.Dense(hidden_neurons, activation='relu')(state_input)
  hidden_2 = layers.Dense(hidden_neurons, activation='relu')(hidden_1)
  q_values = layers.Dense(action_size)(hidden_2)
  print("q values ", q_values)
  masked_q_values = layers.Multiply()([q_values, actions_input])
  print("masked q values ", masked_q_values)

  model = models.Model(
    inputs=[state_input, actions_input], outputs=masked_q_values)
  # optimizer = tf.keras.optimizers.RMSprop(lr=learning_rate)
  optimizer = tf.keras.optimizers.Adam(0.001, beta_1=0.9, beta_2=0.9)
  model.compile(loss='mse', optimizer=optimizer)
  return model


class Agent:
  """Sets up a reinforcement learning agent to play in a game environment."""
  def __init__(self, network, memory, epsilon_decay, action_size):
    """Initializes the agent with DQN and memory sub-classes.

    Args:
        network: A neural network created from deep_q_network().
        memory: A Memory class object.
        epsilon_decay (float): The rate at which to decay random actions.
        action_size (int): The number of possible actions to take.
    """
    self.network = network
    self.action_size = action_size
    self.memory = memory
    self.epsilon = 1  # The chance to take a random action.
    # We decay the change of taking random actions by epsilon_decay.
    self.epsilon_decay = epsilon_decay

  def act(self, state, training=False):
    """Selects an action for the agent to take based on a game state.

    Args:
        state (list of numbers): The state of the environment to act on.
        training (bool): True if the agent is training.

    Returns:
        (int) The index of the action to take.
    """
    if training:
      # Random actions until enough simulations to train the model.
      if len(self.memory.buffer) >= self.memory.batch_size:
        self.epsilon *= self.epsilon_decay

      if self.epsilon > np.random.rand():
        print("Exploration!")
        return random.randint(0, self.action_size-1)

    # If not acting randomly, take action with highest predicted value.
    print("Exploitation!")
    state_batch = np.expand_dims(state, axis=0)
    predict_mask = np.ones((1, self.action_size,))
    action_qs = self.network.predict([state_batch, predict_mask])
    print("action qs ", action_qs)
    input("press to cont.")
    # In exploitation, you take the argmax of the returned action.
    return np.argmax(action_qs[0])



  def learn(self):
    """Trains the Deep Q Network based on stored experiences."""
    batch_size = self.memory.batch_size
    if len(self.memory.buffer) < batch_size:
      return None

    # Obtain random mini-batch from memory.
    state_mb, action_mb, reward_mb, next_state_mb, done_mb = (
      self.memory.sample())

    # Get Q values for next_state.
    predict_mask = np.ones(action_mb.shape + (self.action_size,))
    print("predict mask ", predict_mask)

    # predicting the next q mb
    # TODO: we need to make sure that the values returned by predict
    # are scaled log probas.
    next_q_mb = self.network.predict([next_state_mb, predict_mask])
    print("next q mb ", next_q_mb)

    # here we just take the argmax value for the next state,action pair.
    next_q_mb = tf.math.reduce_max(next_q_mb, axis=1)
    print("next q mb ", next_q_mb)
    input("press to cont. ")

    # Apply the Bellman Equation
    # this part is the equation (reward + max(Q(s+1, a)))
    target_qs = (next_q_mb * self.memory.gamma) + reward_mb
    print("target qs before tf.where", target_qs)
    target_qs = tf.where(done_mb, reward_mb, target_qs)
    print("target qs after tf.where ", target_qs)
    input("press to cont.")

    # Match training batch to network output:
    # target_q where action taken, 0 otherwise.
    action_mb = tf.convert_to_tensor(action_mb, dtype=tf.int32)
    print("action mb ", action_mb)
    action_hot = tf.one_hot(action_mb, self.action_size)
    target_mask = tf.multiply(tf.expand_dims(target_qs, -1), action_hot)
    print("target mask ", target_mask)

    # Below target_mask is passed to the sample_weight parameter of the train_on_batch
    # method. It's an array that applies some weights to the model's loss for
    # each sample. In the case of sequential data, this can be a 2D array of
    # (samples, timesteps) that applies a different weight to the loss for every
    # timestep in every sample.
    # In practice here, this target mask makes sure we are only updating the loss for
    # the particular class/action we are interested in, as we zero out the rest of
    # the probabilities for the other actions.
    # The train on batch method also takes care of the backpropagation as well.
    return self.network.train_on_batch(
      [state_mb, action_hot], target_mask, reset_metrics=False
    )

class Memory:
  """Sets up a memory replay buffer for a Deep Q Network.

  A simple memory buffer for a DQN. This one randomly selects state
  transitions with uniform probability, but research has gone into
  other methods. For instance, a weight could be given to each memory
  depending on how big of a difference there is between predicted Q values
  and target Q values.

  Args:
      memory_size (int): How many elements to hold in the memory buffer.
      batch_size (int): The number of elements to include in a replay batch.
      gamma (float): The "discount rate" used to assess Q values.
  """
  def __init__(self, memory_size, batch_size, gamma):
    self.buffer = deque(maxlen=memory_size)
    self.batch_size = batch_size
    self.gamma = gamma

  def add(self, experience):
    """Adds an experience into the memory buffer.

    Args:
        experience: a (state, action, reward, state_prime, done) tuple.
    """
    self.buffer.append(experience)

  def sample(self):
    """Uniformally selects from the replay memory buffer.

    Uniformally and randomly selects experiences to train the nueral
    network on. Transposes the experiences to allow batch math on
    the experience components.

    Returns:
        (list): A list of lists with structure [
            [states], [actions], [rewards], [state_primes], [dones]
        ]
    """
    print("sampling.....")
    # Buffer is the total number of elements we hold in memory.
    buffer_size = len(self.buffer)
    # Sampling is based on batch size. We randomly select n number of elements from
    # buffer for training. Here we get the indices only. Then based on these indices,
    # we pick states, actions, rewards and state_primes in these indices from the batch.
    index = np.random.choice(
      np.arange(buffer_size), size=self.batch_size, replace=False)

    # Columns have different data types, so numpy array would be awkward.
    print("buffer ", self.buffer)
    input("press to cont.")
    batch = np.array([self.buffer[i] for i in index]).T.tolist()
    print("batch ", batch)
    input("press to cont.")
    states_mb = tf.convert_to_tensor(np.array(batch[0], dtype=np.float32))
    actions_mb = np.array(batch[1], dtype=np.int8)
    rewards_mb = np.array(batch[2], dtype=np.float32)
    states_prime_mb = np.array(batch[3], dtype=np.float32)
    dones_mb = batch[4]
    return states_mb, actions_mb, rewards_mb, states_prime_mb, dones_mb

