#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from itertools import product

import re
import json

def postprocessEntities(entities, sentence):

    ents_out = []

    for i in range(len(entities)):
        ent = None
        try:
            
            ent = (
                entities[i] 
                .replace(" ##", "")               
                .replace("##", "")
                .replace("[UNK]", "")
                .replace(" . ", ".")
                .replace("  ", " ")
                .strip()
            )

            if ent.endswith(",") or ent.endswith(".") or ent.endswith(":"):
                ent = ent[0 : len(ent) - 1]
            if ent.startswith(",") or ent.startswith(".") or ent.startswith(":"):
                ent = ent[1 : len(ent)]

            ent = (re.findall(
                "(?i)[^a-zA-Z0-9]*" + re.escape(ent) + "+[^a-zA-Z]",
                sentence,
            )+re.findall(
                "(?i)[^a-zA-Z0-9]*" + re.escape(ent.replace(" ", "")) + "+[^a-zA-Z]",
                sentence,
            ))[0].strip()

            if ent.endswith(",") or ent.endswith(".") or ent.endswith(":"):
                ent = ent[0 : len(ent) - 1]
            if ent.startswith(",") or ent.startswith(".") or ent.startswith(":"):
                ent = ent[1 : len(ent)]

            if (ent is not None):
                ents_out.append(ent)

        except:
            ent = (
                    entities[i]
                    .replace(" ##", "")               
                    .replace("##", "")
                    .replace("[UNK]", "")
                    .replace(" . ", ".")
                    .strip()
                )

            if ent.endswith(",") or ent.endswith(".") or ent.endswith(":"):
                ent = ent[0 : len(ent) - 1]
            if ent.startswith(",") or ent.startswith(".") or ent.startswith(":"):
                ent = ent[1 : len(ent)]
            
    return ents_out


def generateRelationsData(entities, sentence, source):

    rel_dict = {"sentence": [], "source": []}
    
    if "B-BANDGAP" in entities.keys():
        ent_names = ["B-CHEM", "B-BANDGAP"]
    else:
        ent_names = ["B-CHEM", "B-TEMP"]
    
    for p in product(*[entities[key] for key in ent_names]):
        rel_sentence = sentence
        for i, ent in enumerate(p):
            
            ent = (
                ent.replace("$", "")
                .replace("_", "")
                .replace("}", "")
                .replace("{", "")
                .replace("~", "")
                .replace("\\", "")
            )

            rel_sentence = re.sub(
                "[^a-zA-Z0-9]" + re.escape(ent) + "+[^a-zA-Z0-9]",
                " [E"+str(i+1)+"]" + ent+ "[/E"+str(i+1)+"] ",
                rel_sentence,
                1,
            )
    
        rel_dict["sentence"].append(rel_sentence)
        rel_dict["source"].append(source)

    return rel_dict


def fromNer(ner_dict):
    data = {"sentence": [], "source": []}

    for n in range(len(ner_dict)):
        entities = {}
        for key in ner_dict[n].keys():
            if key.startswith("B-"):
                entities[key] = postprocessEntities(ner_dict[n][key], ner_dict[n]["sentence"])

        entry_rel = generateRelationsData(entities, ner_dict[n]["sentence"], ner_dict[n]["source"])

        data["sentence"].extend(entry_rel["sentence"])
        data["source"].extend(entry_rel["source"])

    return data