import os
import re
from pathlib import Path, PureWindowsPath
from dataclasses import dataclass
from sklearn.model_selection import train_test_split;
from typing import List, Tuple, Dict, Any
import json
from transformers import AutoTokenizer
from copy import copy, deepcopy
import numpy.random as rng
from itertools import chain
from random import sample
from pyleetspeak import LeetSpeaker

def write_document_file(file_path:str, file_list:List[dict]):
    with open(file_path, mode='w', encoding='utf-8') as fd:
        fd.writelines([f"{json.dumps(file_info)}\n" for file_info in file_list])


def load_document_file(file_path:str):
    with open(file_path, mode='r', encoding='utf-8') as fd:
        lines = fd.readlines()
    return [json.loads(line) for line in lines]


def save_file(file_path:str, file: dict):
    with open(file_path, mode='w', encoding='utf-8') as fd:
        json.dump(file, fd)


def load_file(file_path:str):
    with open(file_path, mode='r', encoding='utf-8') as fd:
        file_dict = json.load(fd)
    return file_dict

class AgnosticPath(Path): # From https://stackoverflow.com/questions/60291545/converting-windows-path-to-linux
    """A class that can handle input with Windows (\\) and/or posix (/) separators for paths"""

    def __new__(cls, *args, **kwargs):
        new_path = PureWindowsPath(*args).parts
        if (os.name != "nt") and (len(new_path) > 0) and (new_path[0] in ("/", "\\")):
          new_path = ("/", *new_path[1:])
        return super().__new__(Path, *new_path, **kwargs)

class DatasetLoader:
    """
    You can modify the storage_information attributes updating the following dict (only works on initial load) passed
    on the init argument:
    storage_information = {
        "token_size": 32,
        "tolerance": 2,
        "tokenizer_name": "Alibaba-NLP/gte-base-en-v1.5",
        "files_per_document": 4000,
        "file_format": "processed_files_%d.jsonl",
        "dataset_info": {
            "ai-storage-name": "ai-docs",
            "human-storage-name": "human-docs",
            "document-info": "document_info.json"
        },
        "dataset_types": {"train": 0.6, "validate": 0.2, "test": 0.2}
    }
    """
    def __init__(self,
                 storage_location:str="documents",
                 storage_information_load_name:str="loader_info.json",
                 storage_information: Dict[str, Any] = None):
        self.storage_location = AgnosticPath(".") / storage_location
        # Load storage information
        storage_information_path = self.storage_location / storage_information_load_name
        if self.storage_location.exists():
            st_information = load_file(storage_information_path)
        else:
            st_information = {
                "token_size": 32,
                "tolerance": 2,
                "tokenizer_name": "Alibaba-NLP/gte-base-en-v1.5",
                "files_per_document": 4000,
                "file_format": "processed_files_%d.jsonl",
                "dataset_info": {
                    "ai-storage-name": "ai-docs",
                    "human-storage-name": "human-docs",
                    "document-info-name": "document_info.json"
                },
                "dataset_types": {"train": 0.6, "validate": 0.2, "test": 0.2}
            }
            if storage_information:
                st_information.update(storage_information)
            assert (not any(v < 0. for v in st_information["dataset_types"].values()) and
                    sum(st_information["dataset_types"].values()) == 1.), "Invalid sizes for the dataset types"
            os.mkdir(self.storage_location)
            save_file(storage_information_path, st_information)
        # Loaded parameters
        self.token_size = st_information["token_size"]
        self.tolerance = st_information["tolerance"]
        self.tokenizer = AutoTokenizer.from_pretrained(st_information["tokenizer_name"])
        self.files_per_document = st_information["files_per_document"]
        self.file_format = st_information["file_format"]
        self.dataset_info = st_information["dataset_info"]
        self.dataset_types = st_information["dataset_types"]
        # Generated setup
        for dataset_name in self.dataset_types:
            self.setup(dataset_name)
    
    def setup(self, dataset_name: str):
        memory_path = self.storage_location / dataset_name
        ai_path = memory_path / self.dataset_info["ai-storage-name"]
        human_path = memory_path / self.dataset_info["human-storage-name"]
        document_info_path = memory_path / self.dataset_info["document-info-name"]
        # Create directories
        if all(path.exists() for path in (memory_path, ai_path, human_path)):
            return
        if not memory_path.exists():
            os.mkdir(str(memory_path))
        if not ai_path.exists():
            os.mkdir(str(ai_path))
        if not human_path.exists():
            os.mkdir(str(human_path))
        first_ai_file = str(ai_path / (self.file_format % (0,)))
        first_human_file = str(human_path / (self.file_format % (0,)))
        write_document_file(first_ai_file, [])
        write_document_file(first_human_file, [])
        document_info = {
            "ai-docs": {
                "location": str(ai_path),
                "file_lengths": [],
                "last_file": first_ai_file,
            },
            "human-docs": {
                "location": str(human_path),
                "file_lengths": [],
                "last_file": first_human_file,
            }
        }
        save_file(document_info_path, document_info)
    
    def preprocess_text(self, text: dict):
        """
        Processes the passed text and obtains the token chunks to save with the document.
        """
        variance = int(self.token_size / self.tolerance)
        tk_variance = (self.token_size - variance, self.token_size + variance)
        text_positions = [0]
        last_position = 0
        prev_token = 0
        for phrase in re.split(r"[.!?\n\r]", text["text"]):
            if len(phrase) == 0:
                last_position += 1
                continue
            curr_position = last_position + len(phrase) + 1
            curr_tokens = len(self.tokenizer.tokenize(text["text"][text_positions[-1]:curr_position]))
            if prev_token > tk_variance[0] and curr_tokens > tk_variance[1]:
                text_positions.append(last_position)
                prev_token = curr_tokens - prev_token
            else:
                prev_token = curr_tokens
            last_position += len(phrase) + 1
        if prev_token != 0:
            text_positions.append(last_position)
        text["tk_positions"] = text_positions

    def save_file_info(self, text:dict, files:List[dict], file_info:dict):
        """
        Saves the processed text into files (modifiyng the corresponding file_info) and if files > files_per_document,
        saves the current document and opens a new one.
        """
        files.append(text)
        file_info["file_lengths"].append(len(text["tk_positions"]))
        # If maximum files per document are reached, save the current files and open a new one
        if len(files) % self.files_per_document == 0:
            write_document_file(file_info["last_file"], files)
            files.clear()
            file_info["last_file"] = str(AgnosticPath(file_info["location"]) / (self.file_format % (len(file_info["file_lengths"]),)))

    def preprocess_dataset_files(self, dataset_location:str, texts_to_load: List[dict]):
        """
        Preprocesses dataset files that are meant to be loaded to the preprocessor / document generator. The files to
        load must be dictionaries such that they have a `text` and a `label` (0 human / 1 AI). 
        """
        document_info_path = str(AgnosticPath(dataset_location) / self.dataset_info["document-info-name"])
        document_info = load_file(document_info_path)
        ai_info = document_info["ai-docs"]
        human_info = document_info["human-docs"]
        # Load Last document files
        ai_files = load_document_file(ai_info["last_file"])
        human_files = load_document_file(human_info["last_file"])
        for text in texts_to_load:
            self.preprocess_text(text)
            if text["label"] == 0:
                self.save_file_info(text, human_files, human_info)
            else:
                self.save_file_info(text, ai_files, ai_info)
        write_document_file(human_info["last_file"], human_files)
        write_document_file(ai_info["last_file"], ai_files)
        save_file(document_info_path, document_info)
    
    def preprocess_files(self, texts_to_load:List[dict]):
        """
        Preprocesses files that are meant to be loaded to the preprocessor / document generator. The files to load must be
        dictionaries such that they have a `text` and a `label` (0 human / 1 AI). 
        """
        curr_total = 1.
        curr_texts = texts_to_load
        num_datasets = len(self.dataset_types)
        for idx, dataset_name in enumerate(self.dataset_types):
            dataset_path = str(self.storage_location / dataset_name)
            if idx == num_datasets - 1:
                self.preprocess_dataset_files(dataset_path, curr_texts)
                continue
            train_size = self.dataset_types[dataset_name] / curr_total
            files_to_add, curr_texts = train_test_split(curr_texts, train_size=train_size)
            self.preprocess_dataset_files(dataset_path, files_to_add)
            curr_total -= self.dataset_types[dataset_name]
    
    def create_generators(self,
                          n_test_cases: int,
                          token_size:int=2,
                          rep_noisy_cases:float=1.0,
                          change_prob:float=0.8,
                          change_freq:float=0.8
                          ):
        return {dataset_name: DatasetGenerator(
            document_info_path=str(self.storage_location / dataset_name / self.dataset_info["document-info-name"]),
            files_per_document=self.files_per_document,
            file_format=self.file_format,
            n_test_cases=n_test_cases,
            token_size=token_size,
            rep_noisy_cases=rep_noisy_cases,
            change_prob=change_prob,
            change_freq=change_freq
        ) for dataset_name in self.dataset_types}

@dataclass
class DatasetGenerator:
    document_info_path: str
    files_per_document: int
    file_format: str
    n_test_cases: int
    token_size:int=2,
    rep_noisy_cases:float=1.0,
    change_prob:float=0.8,
    change_freq:float=0.8

    def __post_init__(self):
        self.document_info = load_file(self.document_info_path)
    
    def _rng_text_chunk(self, file_info: List[int], size:int):
        return [
            (text_id, rng.randint(0, file_info[text_id] - self.token_size) if file_info[text_id] > self.token_size else 0)
            for text_id in rng.randint(0, len(file_info), size=size)
        ]
    
    def _read_text_chunks(self, doc_type:str, text_list_id: List[Tuple[int, int]]):
        files_per_document = self.files_per_document
        file_path = str(AgnosticPath(self.document_info[doc_type]["location"]) / self.file_format)
        text_dict = {}
        file_start = 0
        texts = load_document_file(file_path % (file_start,))
        for doc_id, pos in text_list_id:
            while doc_id - file_start >= files_per_document:
                file_start += files_per_document
                texts = load_document_file(file_path % (file_start,))
            text = texts[doc_id - file_start]
            tk_positions = text["tk_positions"]
            if pos + self.token_size >= len(tk_positions):
                text_dict[(doc_id, pos)] = text["text"][tk_positions[pos]:]
            else:
                text_dict[(doc_id, pos)] = text["text"][tk_positions[pos]:tk_positions[pos + self.token_size]]
        return text_dict
    
    def _rng_text(self, file_info: List[int], size:int):
        return rng.randint(0, len(file_info), size=size)
    
    def _read_text(self, doc_type:str, text_ids: List[int]):
        files_per_document = self.files_per_document
        file_path = str(AgnosticPath(self.document_info[doc_type]["location"]) / self.file_format)
        text_dict = {}
        file_start = 0
        texts = load_document_file(file_path % (file_start,))
        for doc_id in text_ids:
            while doc_id - file_start >= files_per_document:
                file_start += files_per_document
                texts = load_document_file(file_path % (file_start,))
            text = texts[doc_id - file_start]
            text_dict[doc_id] = text["text"] # {"id": text["id"], "text": text["text"]}
        return text_dict

    def _generate_test_cases_tuples(self, min_dist:int=0, max_dist:int=1, only_ai_human:bool=False):
        ai_info = self.document_info["ai-docs"]
        human_info = self.document_info["human-docs"]
        if only_ai_human:
            num_cases = int(self.n_test_cases // 2)
            # Text types
            human_ai = (self._rng_text_chunk(human_info["file_lengths"], num_cases), self._rng_text_chunk(ai_info["file_lengths"], num_cases))
            ai_human = (self._rng_text_chunk(ai_info["file_lengths"], num_cases), self._rng_text_chunk(human_info["file_lengths"], num_cases))
            # Texts to read
            human_dict = self._read_text_chunks("human-docs", sorted(chain(human_ai[0], ai_human[1])))
            ai_dict = self._read_text_chunks("ai-docs", sorted(chain(human_ai[1], ai_human[0])))
            return [
                *((human_dict[h], ai_dict[ai], max_dist) for h, ai in zip(*human_ai)),
                *((ai_dict[ai], human_dict[h], max_dist) for ai, h in zip(*ai_human)),
            ]
        num_cases = int(self.n_test_cases // 4)
        # Text types
        human_human = (self._rng_text_chunk(human_info["file_lengths"], num_cases), self._rng_text_chunk(human_info["file_lengths"], num_cases))
        human_ai = (self._rng_text_chunk(human_info["file_lengths"], num_cases), self._rng_text_chunk(ai_info["file_lengths"], num_cases))
        ai_human = (self._rng_text_chunk(ai_info["file_lengths"], num_cases), self._rng_text_chunk(human_info["file_lengths"], num_cases))
        ai_ai = (self._rng_text_chunk(ai_info["file_lengths"], num_cases), self._rng_text_chunk(ai_info["file_lengths"], num_cases))
        # Texts to read
        human_dict = self._read_text_chunks("human-docs", sorted(chain(human_human[0], human_human[1], human_ai[0], ai_human[1])))
        ai_dict = self._read_text_chunks("ai-docs", sorted(chain(ai_ai[0], ai_ai[1], human_ai[1], ai_human[0])))
        return [
            *((human_dict[h1], human_dict[h2], min_dist) for h1, h2 in zip(*human_human)),
            *((human_dict[h], ai_dict[ai], max_dist) for h, ai in zip(*human_ai)),
            *((ai_dict[ai], human_dict[h], max_dist) for ai, h in zip(*ai_human)),
            *((ai_dict[ai1], ai_dict[ai2], min_dist) for ai1, ai2 in zip(*ai_ai))
        ]
    
    def _add_noise(self, text:str):
        leet_speaker = LeetSpeaker(text, change_prb=self.change_prob, change_frq=self.change_freq, mode="basic")
        return leet_speaker.text2leet()

    def generate_tuples(self, min_dist:int=0, max_dist:int=1, only_ai_human:bool=False):
        test_cases = self._generate_test_cases_tuples(min_dist, max_dist, only_ai_human)
        if not self.rep_noisy_cases == 0.:
            noisify_cases = (copy(test_cases) if self.rep_noisy_cases == 1. else
                             sample(test_cases, int(len(test_cases)*self.rep_noisy_cases)))
            noisy_cases = []
            for t1, t2, label in noisify_cases:
                rng_values = rng.randint(0, 2, size = 2)
                t1_noisy = self._add_noise(t1) if 0 in rng_values else t1
                t2_noisy = self._add_noise(t2) if 1 in rng_values else t2
                noisy_cases.append((t1_noisy, t2_noisy, label))
            test_cases.extend(noisy_cases)
        return test_cases
    
    def _generate_test_cases_singles(self, human_label:int=0, ai_label:int=1):
        num_cases = int(self.n_test_cases // 2)
        human_l = self._rng_text_chunk(self.document_info["human-docs"]["file_lengths"], num_cases)
        ai_l = self._rng_text_chunk(self.document_info["ai-docs"]["file_lengths"], num_cases)
        human_dict = self._read_text_chunks("human-docs", sorted(human_l))
        ai_dict = self._read_text_chunks("ai-docs", sorted(ai_l))
        return [
            *((human_dict[h], human_label) for h in human_l),
            *((ai_dict[ai], ai_label) for ai in ai_l),
        ]

    def generate_singles(self, human_label:int=0, ai_label:int=1):
        test_cases = self._generate_test_cases_singles(human_label, ai_label)
        if not self.rep_noisy_cases == 0.:
            noisify_cases = (copy(test_cases) if self.rep_noisy_cases == 1. else
                             sample(test_cases, int(len(test_cases)*self.rep_noisy_cases)))
            noisy_cases = ((self._add_noise(t), v) for t, v in noisify_cases)
            test_cases.extend(noisy_cases)
        return test_cases
    
    def _generate_test_cases_triplets(self, same_positive_anchor:bool=False):
        anchor = self._rng_text_chunk(self.document_info["human-docs"]["file_lengths"], self.n_test_cases)
        positive = (anchor if same_positive_anchor else
                    self._rng_text_chunk(self.document_info["human-docs"]["file_lengths"], self.n_test_cases))
        negative = self._rng_text_chunk(self.document_info["ai-docs"]["file_lengths"], self.n_test_cases)
        human_dict = self._read_text_chunks("human-docs", sorted(chain(anchor, positive)))
        ai_dict = self._read_text_chunks("ai-docs", sorted(negative))
        return {
            "anchor": [human_dict[v] for v in anchor],
            "positive": [human_dict[v] for v in positive],
            "negative": [ai_dict[v] for v in negative]
        }

    def generate_triplets(self, same_positive_anchor:bool=False):
        triplet_cases = self._generate_test_cases_triplets(same_positive_anchor)
        if not self.rep_noisy_cases == 0.:
            for test_cases in triplet_cases.values():
                noisify_cases = (copy(test_cases) if self.rep_noisy_cases == 1. else
                                sample(test_cases, int(len(test_cases)*self.rep_noisy_cases)))
                test_cases.extend(map(self._add_noise, noisify_cases))
        return triplet_cases

    def _generate_random_pairings(self, is_human:float=0.):
        ai_info = self.document_info["ai-docs"]
        human_info = self.document_info["human-docs"]
        num_cases = int(self.n_test_cases // 2)
        human_ai = (self._rng_text(human_info["file_lengths"], num_cases), self._rng_text(ai_info["file_lengths"], num_cases))
        ai_human = (self._rng_text(ai_info["file_lengths"], num_cases), self._rng_text(human_info["file_lengths"], num_cases))
        human_dict = self._read_text("human-docs", sorted(chain(human_ai[0], ai_human[1])))
        ai_dict = self._read_text("ai-docs", sorted(chain(human_ai[1], ai_human[0])))
        return [
            *({"text1": human_dict[h], "text2": ai_dict[ai], "is_human": is_human} for h, ai in zip(*human_ai)),
            *({"text1": ai_dict[ai], "text2": human_dict[h], "is_human": 1. - is_human} for ai, h in zip(*ai_human)),
        ]

    def generate_random_pairings(self, is_human:float=0.):
        random_pairings = self._generate_random_pairings(is_human)
        if not self.rep_noisy_cases == 0.:
            noisify_cases = (deepcopy(random_pairings) if self.rep_noisy_cases == 1. else
                             sample(random_pairings, int(len(random_pairings)*self.rep_noisy_cases)))
            noisy_cases = []
            for texts in noisify_cases:
                rng_values = rng.randint(0, 2, size = 2)
                noisy_texts = deepcopy(texts)
                if 0 in rng_values:
                    noisy_texts["text1"] = self._add_noise(noisy_texts["text1"])
                if 1 in rng_values:
                    noisy_texts["text2"] = self._add_noise(noisy_texts["text2"])
                noisy_cases.append(noisy_texts)
            random_pairings.extend(noisy_cases)
        return random_pairings

'''
num_cases = int(self.n_test_cases // 4)
human_ai = (self._rng_text_chunk(human_info["file_lengths"], num_cases), self._rng_text_chunk(ai_info["file_lengths"], num_cases))
human_human = (self._rng_text_chunk(human_info["file_lengths"], num_cases), self._rng_text_chunk(human_info["file_lengths"], num_cases))
ai_human = (self._rng_text_chunk(ai_info["file_lengths"], num_cases), self._rng_text_chunk(human_info["file_lengths"], num_cases))
ai_ai = (self._rng_text_chunk(ai_info["file_lengths"], num_cases), self._rng_text_chunk(ai_info["file_lengths"], num_cases))
human_dict = self._read_text("human-docs", sorted(chain(human_ai[0], ai_human[1], human_human[0], human_human[1])))
ai_dict = self._read_text("ai-docs", sorted(chain(human_ai[1], ai_human[0], ai_ai[0], ai_ai[1])))
return [
    *((human_dict[h1], ai_dict[h2], (human_label, human_label)) for h1, h2 in zip(*human_human)),
    *((human_dict[h], ai_dict[ai], (human_label, ai_label)) for h, ai in zip(*human_ai)),
    *((ai_dict[ai], human_dict[h], (ai_label, human_label)) for ai, h in zip(*ai_human)),
    *((human_dict[ai1], ai_dict[ai2], (ai_label, ai_label)) for ai1, ai2 in zip(*ai_ai))
]
'''