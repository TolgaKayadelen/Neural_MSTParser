import tensorflow as tf
from tagset.dep_labels import dep_label_enum_pb2 as dep_label_tags
from util import writer
import collections

def _fine_label_index_to_name(label_index):
  return dep_label_tags.Tag.Name(label_index)

def _fine_label_name_to_index(label_name):
  return dep_label_tags.Tag.Value(label_name)

def _coarse_label_index_to_name(label_index):
  return dep_label_tags.CoarseDepTag.Name(label_index)

def _coarse_label_name_to_index(label_name):
  return dep_label_tags.CoarseDepTag.Value(label_name)

fine_to_coarse_map_by_index = {
  0: 0,
  # adjuncts
  2: 1, 7: 1, 8: 1, 18: 1, 35: 1, 40: 1,
  # arguments
  12: 2, 19: 2, 27: 2, 32: 2, 34: 2, 41: 2,
  # modifiers
  1: 3, 3: 3, 4: 3, 5: 3, 6: 3, 9: 3, 13: 3, 14: 3, 15: 3, 16: 3, 20:3, 21:3,
  22:3, 23:3, 24:3, 25:3, 26:3, 28:3, 29:3, 30:3, 31:3, 33:3, 36:3, 38:3,
  # root
  39: 4,
  # cc
  10: 5, 11: 5,
  # conjunct
  17: 6, 37: 6,
  # TOP token
  42: 7,
}

# fine tag index -> coarse tag name
fine_to_coarse_map_by_value = {
  fine_dep :_coarse_label_index_to_name(fine_to_coarse_map_by_index[fine_dep]) for fine_dep in
                                        fine_to_coarse_map_by_index.keys()}

# fine tag name -> coarse tag name
fine_to_coarse_map_by_name = {
  _fine_label_index_to_name(fine_dep) :
  _coarse_label_index_to_name(fine_to_coarse_map_by_index[fine_dep]) for fine_dep in fine_to_coarse_map_by_index.keys()
}

# given a coarse tag index, return set of fine tags that are mapped to it.
def coarse_to_fine_map_by_index(coarse_index):
  fine_index = []
  for k,v in fine_to_coarse_map_by_index.items():
    if v == coarse_index:
      fine_index.append(k)
  return {coarse_index: fine_index}

# given a coarse tag index, return set of fine tag names that are mapped to it.
def coarse_to_fine_map_by_value(coarse_index):
  fine_value = []
  for k,v in fine_to_coarse_map_by_index.items():
    if v == coarse_index:
      fine_value.append(_fine_label_index_to_name(k))
  return {coarse_index: fine_value}


# given a coarse tag name, return set of fine tag names that are mapped to it.
def coarse_to_fine_map_by_name(coarse_name):
  fine_names = []
  for k,v in fine_to_coarse_map_by_name.items():
    if v == coarse_name:
      fine_names.append(k)
  return {coarse_name: fine_names}

def convert_labels(dep_labels):
  converted_labels = []
  for vector in dep_labels:
    # print("vector ", vector)
    converted = [fine_to_coarse_map_by_index[tf.keras.backend.get_value(label)] for label in vector]
    # print("converted ", converted)
    # print("with names ", [fine_to_coarse_map_by_value[tf.keras.backend.get_value(label)] for label in vector])
    converted_labels.append(tf.expand_dims(converted, 0))
  converted_dep_labels = tf.concat(converted_labels, axis=0)
  return converted_dep_labels

if __name__ == "__main__":
  print(fine_to_coarse_map_by_value)
  print()
  print(fine_to_coarse_map_by_name)
  print()
  print(coarse_to_fine_map_by_index(1))
  print()
  print(coarse_to_fine_map_by_value(1))
  print()
  print(coarse_to_fine_map_by_name("root_coarse"))