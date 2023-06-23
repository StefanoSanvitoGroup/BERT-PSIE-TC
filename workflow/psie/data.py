#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module defines the Data class

@author: matteo
"""
import os

from dataclasses import dataclass, field
from typing import List
from monty.serialization import dumpfn, loadfn
from bs4 import BeautifulSoup

from nltk.tokenize import sent_tokenize
import nltk
import re

# nltk.download("punkt", quiet=True)

from transformers import AutoTokenizer

# UNITS = ['C', 'K']


UNITS = [
    "K",
    "Kelvin",
    "h",
    "muB",
    "V",
    "wt",
    "MHz",
    "kHz",
    "GHz",
    "Hz",
    "days",
    "weeks",
    "hours",
    "minutes",
    "seconds",
    "T",
    "MPa",
    "GPa",
    "mol",
    "at",
    "m",
    "s-1",
    "vol.",
    "vol",
    "eV",
    "A",
    "atm",
    "bar",
    "kOe",
    "Oe",
    "mWcm−2",
    "keV",
    "MeV",
    "meV",
    "day",
    "week",
    "hour",
    "minute",
    "month",
    "months",
    "year",
    "cycles",
    "years",
    "fs",
    "ns",
    "ps",
    "rpm",
    "g",
    "mg",
    "mAcm−2",
    "mA",
    "mK",
    "mT",
    "s-1",
    "dB",
    "Ag-1",
    "mAg-1",
    "mAg−1",
    "mAg",
    "mAh",
    "mAhg−1",
    "m-2",
    "mJ",
    "kJ",
    "m2g−1",
    "THz",
    "KHz",
    "kJmol−1",
    "Torr",
    "gL-1",
    "Vcm−1",
    "mVs−1",
    "J",
    "GJ",
    "mTorr",
    "bar",
    "cm2",
    "mbar",
    "kbar",
    "mmol",
    "mol",
    "molL−1",
    "MΩ",
    "Ω",
    "kΩ",
    "mΩ",
    "mgL−1",
    "moldm−3",
    "m2",
    "m3",
    "cm-1",
    "cm",
    "Scm−1",
    "Acm−1",
    "eV−1cm−2",
    "cm-2",
    "sccm",
    "cm−2eV−1",
    "cm−3eV−1",
    "kA",
    "s−1",
    "emu",
    "L",
    "cmHz1",
    "gmol−1",
    "kVcm−1",
    "MPam1",
    "cm2V−1s−1",
    "Acm−2",
    "cm−2s−1",
    "MV",
    "ionscm−2",
    "Jcm−2",
    "ncm−2",
    "Jcm−2",
    "Wcm−2",
    "GWcm−2",
    "Acm−2K−2",
    "gcm−3",
    "cm3g−1",
    "mgl−1",
    "mgml−1",
    "mgcm−2",
    "mΩcm",
    "cm−2s−1",
    "cm−2",
    "ions",
    "moll−1",
    "nmol",
    "psi",
    "mol·L−1",
    "Jkg−1K−1",
    "km",
    "Wm−2",
    "mass",
    "mmHg",
    "mmmin−1",
    "GeV",
    "m−2",
    "m−2s−1",
    "Kmin−1",
    "gL−1",
    "ng",
    "hr",
    "w",
    "mN",
    "kN",
    "Mrad",
    "rad",
    "arcsec",
    "Ag−1",
    "dpa",
    "cdm−2",
    "cd",
    "mcd",
    "mHz",
    "m−3",
    "ppm",
    "phr",
    "mL",
    "ML",
    "mlmin−1",
    "MWm−2",
    "Wm−1K−1",
    "Wm−1K−1",
    "kWh",
    "Wkg−1",
    "Jm−3",
    "m-3",
    "gl−1",
    "A−1",
    "Ks−1",
    "mgdm−3",
    "mms−1",
    "ks",
    "appm",
    "C",
    "HV",
    "kDa",
    "Da",
    "kG",
    "kGy",
    "MGy",
    "Gy",
    "mGy",
    "Gbps",
    "μB",
    "μL",
    "μF",
    "nF",
    "pF",
    "mF",
    "Å",
    "μgL−1",
]


class Dataset:
    def __init__(self, entries=dict(), directory=None, papers_index=None):
        self.entries = entries
        self.directory = directory
        self.papers_index = papers_index

    def __len__(self):
        return len(self.entries.keys())

    def __getitem__(self, key):
        return self.entries[key]

    def __setitem__(self, key, value):
        self.entries[key] = value

    def __add__(self, dataset):
        for key in dataset.keys():
            self[key] = dataset[key]
        return self

    def keys(self):
        return self.entries.keys()

    @classmethod
    def open(cls, filename):
        return loadfn(filename)

    def save(self, filename):
        dumpfn(self, filename, indent=2)

    @property
    def text(self, nolatex=True):
        plain_text = []
        for key in sorted(self.keys()):
            plain_text.append(BeautifulSoup(self[key].text, "html.parser").get_text())

        # if mask_numbers is True:
        #    for i in range(len(plain_text)):
        #        plain_text[i] = re.sub(
        #            r"(\s+-?–?=?[0-9]+\.?[0-9]*\s+)",
        #            " <num> ",
        #            repr(
        #                " ".join(
        #                    _
        #                    for _ in re.split(r"(\s+-?–?=?[0-9]+\.?[0-9]*)", plain_text[i])
        #                    if _
        #                )
        #            ),
        #        )

        return plain_text

    @property
    def relevant(self):
        relevant = []
        for key in sorted(self.keys()):
            relevant.append(self[key].relevant)
        return relevant

    @relevant.setter
    def relevant(self, rel_list):
        assert len(self) == len(rel_list)

        for i, key in enumerate(sorted(self.keys())):
            self[key].relevant = bool(rel_list[i])

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "entries": self.entries,
            "directory": self.directory,
            "papers_index": self.papers_index,
        }

    def get_relevant(self, relevant=True):
        entries = {}
        for key in sorted(self.keys()):
            if self[key].relevant is relevant:
                entries[key] = self[key]
        return Dataset(entries, self.directory)

    def search(self, query, negative=False):
        entries = {}
        for key in sorted(self.keys()):
            if (
                query in BeautifulSoup(self[key].text, "html.parser").get_text()
            ) is not negative:
                entries[key] = self[key]
        return Dataset(entries, self.directory)

    def get_sentences(self):
        sentences = []
        for key in sorted(self.keys()):
            abstract = BeautifulSoup(self[key].text, "html.parser").find_all("body")[0]
            for sentence in sent_tokenize(str(abstract)):
                sentences.append(BeautifulSoup(sentence, "html.parser").get_text())

        return sentences

    def to_bert_classifier(self):
        sentences = []
        labels = []
        for key in sorted(self.keys()):
            abstract = BeautifulSoup(self[key].text, "html.parser").find_all("body")[0]
            for sentence in sent_tokenize(str(abstract)):
                soup = BeautifulSoup(sentence, "html.parser")
                sentences.append(soup.get_text())
                tags = soup.find_all("a")

                label = 0
                for tag in tags:
                    if tag["name"] == "temperature":
                        label = 1

                labels.append(label)

        return {"labels": labels, "text": sentences}

    def get_word_entities(
        self, entities=["chemical-entity", "keyword-entity", "temperature"]
    ):

        IOBs = {"sentences": {}, "tags": {}}

        n = 0
        for key in sorted(self.keys()):
            abstract = BeautifulSoup(self[key].text, "html.parser").find_all("body")[0]
            sentences = sent_tokenize(str(abstract))

            IOB_a = []
            sentences_a = []
            for sentence in sentences:
                ent_dict = get_entities(sentence, entities=entities)

                processed_sentence = (
                    BeautifulSoup(sentence, "html.parser").get_text().lower()
                )
                processed_sentence = (
                    processed_sentence.replace("$", "")
                    .replace("_", "")
                    .replace("}", "")
                    .replace("{", "")
                    .replace("~", "")
                    .replace("\\", "")
                )

                for unit in UNITS:
                    processed_sentence = re.sub(
                        "[^a-zA-Z0-9]*[0-9]+[^a-zA-Z0-9]*[0-9]*[^a-zA-Z0-9]*"
                        + unit
                        + "+[^a-zA-Z0-9]*",
                        " <Num> " + unit + " ",
                        processed_sentence,
                    )

                sentence_t = processed_sentence.split()

                IOB = ["X" for _ in range(len(sentence_t))]
                for ent_type in ent_dict.keys():
                    ent_tag = ent_type[0:4].upper()

                    for entity in ent_dict[ent_type]:
                        token = entity.lower()
                        token = (
                            token.replace("$", "")
                            .replace("_", "")
                            .replace("}", "")
                            .replace("{", "")
                            .replace("~", "")
                            .replace("\\", "")
                        )

                        for unit in UNITS:
                            token = re.sub(
                                "[^a-zA-Z0-9]*[0-9]+[^a-zA-Z0-9]*[0-9]*[^a-zA-Z0-9]*"
                                + unit
                                + "+[^a-zA-Z0-9]*",
                                " <Num> " + unit + " ",
                                token,
                            )

                        token.split()
                        for i in range(len(sentence_t)):
                            if token[0] in sentence_t[i]:
                                if i < len(sentence_t) - len(token):
                                    if "".join(token) in "".join(
                                        sentence_t[i : i + len(token)]
                                    ):
                                        IOB[i : i + len(token)] = ["B-" + ent_tag] + [
                                            "I-" + ent_tag
                                            for _ in range(len(token) - 1)
                                        ]
                if IOB != ["X" for _ in range(len(sentence_t))]:
                    IOB_a.append(IOB)
                    sentences_a.append(" ".join(sentence_t))

            if len(sentences_a) > 0:
                IOBs["sentences"][str(n)] = sentences_a
                IOBs["tags"][str(n)] = IOB_a
                n += 1

        return IOBs

    def get_token_entities(
        self,
        tokenizer,
        entities=["chemical-entity", "keyword-entity", "temperature"],
        padding=False,
        max_len=128,
    ):
        IOBs = {"sentences": [], "labels": []}

        tokenizer = tokenizer

        for key in sorted(self.keys()):
            abstract = BeautifulSoup(self[key].text, "html.parser").find_all("body")[0]
            sentences = sent_tokenize(str(abstract))

            for sentence in sentences:
                ent_dict = get_entities(sentence, entities=entities)
                processed_sentence = preprocess_text(
                    BeautifulSoup(sentence, "html.parser").get_text()
                )
                sentence_t = tokenizer.tokenize(processed_sentence)

                sentence_t.insert(0, "[CLS]")
                sentence_t.append("[SEP]")

                lent = len(sentence_t)

                if padding is True:
                    IOB = (
                        ["[CLS]"]
                        + ["O" for _ in range(len(sentence_t) - 2)]
                        + ["[SEP]"]
                        + ["[PAD]" for _ in range(max_len - len(sentence_t))]
                    )
                    sentence_t += ["[PAD]" for _ in range(max_len - len(sentence_t))]
                else:
                    IOB = (
                        ["[CLS]"]
                        + ["O" for _ in range(len(sentence_t) - 2)]
                        + ["[SEP]"]
                    )

                for ent_type in ent_dict.keys():
                    ent_tag = ent_type[0:4].upper()

                    for entity in ent_dict[ent_type]:
                        entity = " " + entity.strip() + " "
                        token = tokenizer.tokenize(preprocess_text(entity))
                        for i in range(len(sentence_t)):
                            if token[0] == sentence_t[i]:
                                if i < len(sentence_t) - len(token):
                                    if token == sentence_t[i : i + len(token)]:
                                        IOB[i : i + len(token)] = ["B-" + ent_tag] + [
                                            "I-" + ent_tag
                                            for _ in range(len(token) - 1)
                                        ]

                if IOB != ["[CLS]"] + ["O" for _ in range(lent - 2)] + ["[SEP]"] + [
                    "[PAD]" for _ in range(max_len - lent)
                ]:
                    IOBs["sentences"].append(processed_sentence.lower())
                    IOBs["labels"].append(IOB)

        return IOBs

    def get_relations(self):
        relations = {"compound": [], "Tc": []}
        for key in sorted(self.keys()):
            relations["compound"] += self[key].compound
            relations["Tc"] += self[key].Tc

        return relations

    @classmethod
    def from_dict(cls, d):
        entries = dict()

        for key, value in d["entries"].items():
            entries[key] = Entry.from_dict(value)
        return cls(entries, d["directory"], d["papers_index"])

    @classmethod
    def from_dir(cls, *args):
        dataset = cls()

        for arg in args:
            for filename in sorted(os.listdir(arg)):
                dataset[os.path.join(arg, filename)] = Entry(
                    os.path.join(arg, filename)
                )
                with open(os.path.join(arg, filename), "r") as f:
                    dataset[os.path.join(arg, filename)].text = f.read()
        return dataset

    def to_dir(self, dir_path):
        try:
            os.mkdir(dir_path)
        except:
            pass

        for key in self.keys():
            filename = os.path.split(self[key].source)[-1]

            with open(os.path.join(dir_path, filename), "w") as f:
                f.write(self[key].text)


@dataclass
class Entry:
    source: str
    relevant: bool = None
    text: str = None
    compound: List = field(default_factory=list)
    Tc: List = field(default_factory=list)
    type: List = field(default_factory=list)
    additional_info: dict = None

    def as_dict(self):
        return {
            "@module": self.__class__.__module__,
            "@class": self.__class__.__name__,
            "source": self.source,
            "relevant": self.relevant,
            "text": self.text,
            "compound": self.compound,
            "Tc": self.Tc,
            "type": self.type,
            "additional_info": self.additional_info,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            source=d["source"],
            relevant=d["relevant"],
            text=d["text"],
            compound=d["compound"],
            Tc=d["Tc"],
            type=d["type"],
            additional_info=d["additional_info"],
        )


def get_entities(text, entities=["chemical-entity", "keyword-entity", "temperature"]):
    soup = BeautifulSoup(text, "html.parser")
    tags = soup.find_all(["a", "span"])

    ent_dict = {entity: [] for entity in entities}
    for i in range(len(tags)):
        if tags[i].name == "a":
            j = i + 1
            content = []

            flag = True
            while flag == True and j < len(tags):
                if tags[j].name != "span":
                    flag = False
                else:
                    content += tags[j].string
                    j += 1

            if len(content) > 0:
                if tags[i]["name"] in entities:
                    ent_dict[tags[i]["name"]].append("".join(content))

    return ent_dict


def preprocess_text(text):
    processed_text = (
        text.replace("$", "")
        .replace("_", "")
        .replace("}", "")
        .replace("{", "")
        .replace("~", "")
        .replace("\\", "")
    )

    for unit in UNITS:

        split_points = re.findall(
            "[a-zA-Z=][^a-zA-Z0-9]*[0-9]+[^a-zA-Z0-9]*[0-9]*[^a-zA-Z0-9]*"
            + unit
            + "+[^a-zA-Z0-9]*[^a-zA-Z0-9]+",
            processed_text,
        )

        for split_point in split_points:
            processed_text = str(split_point[0] + " " + split_point[1:]).join(
                processed_text.split(split_point)
            )

        # processed_text = re.sub(
        #    " +[^a-zA-Z0-9]*[0-9]+[^a-zA-Z0-9]*[0-9]*[^a-zA-Z0-9]*" + unit + "+[^a-zA-Z0-9]*[^a-zA-Z0-9]+",
        #    " <nUm> " + unit + " ",
        #    processed_text,
        # )

    # processed_text = re.sub(
    #    " [^a-zA-Z0-9]*[0-9]+[^a-zA-Z0-9]*[0-9]*[^a-zA-Z0-9]* +",
    #    " <nUm> ",
    #    processed_text,
    # )

    # processed_text = re.sub(
    #    "=[^a-zA-Z0-9]*[0-9]+[^a-zA-Z0-9]*[0-9]*[^a-zA-Z0-9]*" + unit + "+[^a-zA-Z0-9]* +",
    #    "= <nUm> " + unit + " ",
    #    processed_text,
    # )

    return processed_text.lower()
