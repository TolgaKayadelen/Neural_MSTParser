"""One of script to get all the morphological tags in the training data."""

from util import reader

def from_data():
  data = "data/UDv23/Turkish/training/treebank_train_0_500.pbtxt"
  trb = reader.ReadTreebankTextProto(data)
  morph_tagset = set()
  for sentence in trb.sentence:
    for token in sentence.token:
      for morph in token.morphology:
        morph_tagset.add(morph.name.replace("[psor]", "_psor") + "_" + morph.value.replace(",", "_"))
  morph_tags = dict(enumerate(sorted(list(morph_tagset))))
  return morph_tags

def from_token(*, token):
  return [morph.name.replace("[psor]", "_psor") + "_" + morph.value.replace(",", "_") for morph in token.morphology]


if __name__ == "__main__":
  morph_tags = from_data()
  for key, tag in morph_tags.items():
    print(f"{tag} = {key+1};")
        