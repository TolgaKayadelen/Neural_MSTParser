from tagset.fine_pos import fine_tag_enum_pb2 as fine_tags
from tagset.dep_labels import dep_label_enum_pb2 as dep_labels
from tagset.coarse_pos import coarse_tag_enum_pb2 as coarse_tags


from tagset.dep_labels.imst import imst_dep_label_enum_pb2 as imst_dep_labels
from tagset.fine_pos.imst import imst_fine_tag_enum_pb2 as imst_fine_tags
from tagset.dep_labels.en import dep_label_enum_pb2 as dep_labels_en
from tagset.dep_labels.de import dep_label_enum_pb2 as dep_labels_de
from tagset.dep_labels.fi import dep_label_enum_pb2 as dep_labels_fi
from tagset.dep_labels.zh import dep_label_enum_pb2 as dep_labels_zh
from tagset.dep_labels.ru import dep_label_enum_pb2 as dep_labels_ru
from tagset.dep_labels.ko import dep_label_enum_pb2 as dep_labels_ko


# from tagset.arg_str import semantic_role_enum_pb2 as srl
from tagset.morphology import morph_tag_enum_pb2 as morph_tags
from tagset.morphology.imst import imst_morph_tag_enum_pb2 as imst_morph_tags

_LANGUAGE_TO_TAG = {
  "en": {"dep_labels": dep_labels_en},
  "de": {"dep_labels": dep_labels_de},
  "fi": {"dep_labels": dep_labels_fi},
  "zh": {"dep_labels": dep_labels_zh},
  "ko": {"dep_labels": dep_labels_ko},
  "ru": {"dep_labels": dep_labels_ru},
  #"tr": {"pos": fine_tags,
  #       "category": coarse_tags,
  #      "dep_labels": dep_labels,
        # "srl": srl,
  #      "morph": morph_tags}
   "tr": {"pos": imst_fine_tags, # TODO
         # "category": coarse_tags,
         "dep_labels": imst_dep_labels,
         # "srl": srl,
         "morph": imst_morph_tags}
}

_TAGS_TO_REPLACE = {'SEMICOLON': ":", 'COMMA': ",", 'DOT': ".", 'LRB': "-LRB-", 'RRB': "-RRB-",
                    'PRP_DOLLAR': "PRP$", 'DOUBLE_QUOTE_ITALIC': "``",
                    'DOUBLE_QUOTE': "''", 'WP_DOLLAR': "WP$", 'DOLLAR': "$"}

class LabelReader:
  """The label reader returns a dict of label:index pairs."""
  def __init__(self, tagset, language="tr", reverse=False):
    self._tagset = tagset
    self._language = language
    self._reverse = reverse

  @classmethod
  def get_labels(cls, tagset, language="tr", reverse=False):
    return cls(tagset, language, reverse)
    
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
      tags = _LANGUAGE_TO_TAG[self._language]["pos"]
    elif self._tagset == "category":
      tags = coarse_tags
      tags = _LANGUAGE_TO_TAG[self._language]["category"]
    elif self._tagset == "dep_labels":
      tags = dep_labels
      tags = _LANGUAGE_TO_TAG[self._language]["dep_labels"]
    elif self._tagset == "srl":
      tags = srl
    elif self._tagset == "morph":
      tags = imst_morph_tags # TODO
      #tags = morph_tags
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

    # if tags == srl:
    if False:
      label_dict = _get_bio_tags_from_srl()
    else:
      label_dict = {}
      for key in tags.Tag.DESCRIPTOR.values_by_name.keys():
        # print(key)
        # input()
        if key.startswith("UNKNOWN_"):
          continue
        if key in {"advmod_emph",
                   "aux_q",
                   "aux_pass",
                   'acl_relcl',
                   # "case_loc",
                   "compound_lvc",
                   "compound_redup",
                   "compound_ext",
                   "compound_prt",
                   "compound_nn",
                   "csubj_pass",
                   "csubj_cop",
                   "cc_preconj",
                   "cop_own",
                   'det_predet',
                   "det_poss",
                   "discourse_sp",
                   "expl_pv",
                   'flat_name',
                   'flat_foreign',
                   'mark_rel',
                   'mark_adv',
                   'mark_prt',
                   "nsubj_pass",
                   "nsubj_cop",
                   "nummod_gov",
                   "nummod_entity",
                   "nmod_poss",
                   'nmod_tmod',
                   'nmod_npmod',
                   "nmod_gobj",
                   "nmod_gsubj",
                   "obl_agent",
                   'obl_npmod',
                   'obl_tmod',
                   'obl_arg',
                   'obl_patient',
                   "xcomp_ds"}:
          label_dict[key.replace("_", ":")] = tags.Tag.Value(key)
        elif key in {'SEMICOLON', 'COMMA', 'DOT', 'LRB', 'RRB', 'PRP_DOLLAR', 'DOUBLE_QUOTE_ITALIC',
                     'DOUBLE_QUOTE', 'WP_DOLLAR', 'DOLLAR'}:
          label_dict[_TAGS_TO_REPLACE[key]] = tags.Tag.Value(key)
        else:
          label_dict[key] = tags.Tag.Value(key)
          # print(label_dict)
    label_dict["-pad-"] = 0
    
    if self._reverse:
      return {k:v for v,k in label_dict.items()}
    # print("label dict ", label_dict)
    # input("press to cont.")
    return label_dict
