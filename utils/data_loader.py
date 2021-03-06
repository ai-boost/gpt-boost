import os
import pickle
import random
import time

import torch
from torch.utils.data.dataset import Dataset

from filelock import FileLock

from transformers import PreTrainedTokenizer
from transformers import logging

logger = logging.get_logger(__name__)


class TextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_lm_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )

            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = []
                with open(file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    print("lines number: ", len(lines))

                tokenized_text = []
                temp_lines = []
                counter = 0
                for line in lines:
                    counter += 1
                    temp_lines.append(line)
                    if counter % 50000 == 0:
                        text = "".join(temp_lines)
                        print(counter)
                        line_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
                        tokenized_text.extend(line_ids)
                        temp_lines = []

                for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                    self.examples.append(
                        tokenizer.build_inputs_with_special_tokens(tokenized_text[i: i + block_size])
                    )
                # Note that we are losing the last truncated example here for the sake of simplicity (no padding)
                # If your dataset is small, first you should loook for a bigger one :-) and second you
                # can change this behavior by adding (model specific) padding.

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class TextDatasetRandom(Dataset):
    """
    random text dataset, no cache, dynamic tokenize and sample
    will be fast on loading datasets and slow on training
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            data_num=1000000,
            pad_id=1,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        self.data_num = data_num
        self.tokenizer = tokenizer
        self.block_size = block_size - tokenizer.num_special_tokens_to_add(pair=False)
        self.paragraph_size = 5 * block_size
        self.pad_id = pad_id

        with open(file_path, encoding="utf-8") as f:
            self.text = f.read()
            self.tokens_num = len(self.text)
            print("tokens number: ", self.tokens_num)
        self.right_boundary = self.tokens_num - self.paragraph_size

    def __len__(self):
        return self.data_num

    def __getitem__(self, i) -> torch.Tensor:
        pos = random.randint(0, self.right_boundary)
        ids = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.text[pos:pos + self.paragraph_size]))

        while len(ids) < self.block_size:
            ids.append(self.pad_id)
        ids = self.tokenizer.build_inputs_with_special_tokens(ids[:self.block_size])
        return torch.tensor(ids, dtype=torch.long)


class LineByLineTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i) -> torch.Tensor:
        return torch.tensor(self.examples[i], dtype=torch.long)


class TextDatasetForNextSentencePrediction(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            file_path: str,
            block_size: int,
            overwrite_cache=False,
    ):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"

        block_size = block_size - tokenizer.num_special_tokens_to_add(pair=True)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory,
            "cached_nsp_{}_{}_{}".format(
                tokenizer.__class__.__name__,
                str(block_size),
                filename,
            ),
        )

        self.tokenizer = tokenizer
        self.examples = []

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"

        # Input file format:
        # (1) One sentence per line. These should ideally be actual sentences, not
        # entire paragraphs or arbitrary spans of text. (Because we use the
        # sentence boundaries for the "next sentence prediction" task).
        # (2) Blank lines between documents. Document boundaries are needed so
        # that the "next sentence prediction" task doesn't span between documents.
        #
        # Example:
        # I am very happy.
        # Here is the second sentence.
        #
        # A new document.

        with FileLock(lock_path):
            if os.path.exists(cached_features_file) and not overwrite_cache:
                start = time.time()
                with open(cached_features_file, "rb") as handle:
                    self.examples = pickle.load(handle)
                logger.info(
                    f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start
                )
            else:
                logger.info(f"Creating features from dataset file at {directory}")

                self.examples = [[]]
                with open(file_path, encoding="utf-8") as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = line.strip()

                        # Empty lines are used as document delimiters
                        if not line and len(self.examples[-1]) != 0:
                            self.examples.append([])
                        tokens = tokenizer.tokenize(line)
                        tokens = tokenizer.convert_tokens_to_ids(tokens)
                        if tokens:
                            self.examples[-1].append(tokens)

                start = time.time()
                with open(cached_features_file, "wb") as handle:
                    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)
                logger.info(
                    "Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]
