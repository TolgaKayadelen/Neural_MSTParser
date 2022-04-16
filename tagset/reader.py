from tagset.fine_pos import fine_tag_enum_pb2 as fine_tags
from tagset.coarse_pos import coarse_tag_enum_pb2 as coarse_tags
from tagset.dep_labels import dep_label_enum_pb2 as dep_labels
from tagset.arg_str import semantic_role_enum_pb2 as srl
from tagset.morphology import morph_tag_enum_pb2 as morph_tags

class LabelReader:
  """The label reader returns a dict of label:index pairs."""
  def __init__(self, tagset, reverse):
    self._tagset = tagset
    self._reverse = reverse
  
  @classmethod
  def get_labels(cls, tagset, reverse=False):
    return cls(tagset, reverse)
  
    
  def itov(self, idx: int):
    """Returns the label value given an integer index."""
    reverse_dict = {k:v for v,k in self.labels.items()}
    return reverse_dict[idx]
      
  def vtoi(self, value: str):
    """Returns index given label value."""
    return self.labels[value]
  
  
  @property
  def labels(self):
    """Returns a dict of labels.
    
    Args:
      tagset: string, the label set to use. 
    Returns:
      label_dict: A dict of [label:index] pairs.
    """
    # read the tagset
    if self._tagset == "pos":
      tags = fine_tags
    elif self._tagset == "category":
      tags = coarse_tags
    elif self._tagset == "dep_labels":
      tags = dep_labels
    elif self._tagset == "srl":
      tags = srl
    elif self._tagset == "morph":
      tags = morph_tags
    else:
      raise ValueError("Invalid tagset requested.")
      
    def _get_bio_tags_from_srl():
      labels_list = ["-pad-"]
      for key in tags.Tag.DESCRIPTOR.values_by_name.keys():
        if key in ["UNKNOWN_SRL", "V"]:
          continue
        if key.startswith(("A_", "AM_", "A4_", "C_", "R_", "notset")):
          key = key.replace("_", "-")
        labels_list.extend(["B-"+key, "I-"+key])
      labels_list.extend(["O", "V"])
      return {v: k for k, v in enumerate(labels_list)}
  
    
    if tags == srl:
      label_dict = _get_bio_tags_from_srl()
    else:
      label_dict = {}
      for key in tags.Tag.DESCRIPTOR.values_by_name.keys():
        if key.startswith("UNKNOWN_"):
          continue
        if key in {"advmod_emph", "aux_q", "compound_lvc",
                   "compound_redup", "nmod_poss", "cc_preconj"}:
          label_dict[key.replace("_", ":")] = tags.Tag.Value(key)
        else:
          label_dict[key] = tags.Tag.Value(key)
    label_dict["-pad-"] = 0
    
    if self._reverse:
      return {k:v for v,k in label_dict.items()}
    # print("label dict ", label_dict)
    # input("press to cont.")
    return label_dict
