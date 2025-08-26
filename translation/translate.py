# This file is based on Easy-Translate (https://github.com/ikergarcia1996/Easy-Translate)
# Original code licensed under the Apache License, Version 2.0.
#
# Modified by Paloma Piot on 2025:
#    - Added support for CSV/TSV input files with specified columns for original and translated text.
#    - Added periodic saving of translations to avoid data loss during long runs.


import argparse
import glob
import math
import os
import tempfile
from typing import List, Optional

import torch
from accelerate import Accelerator, DistributedType, find_executable_batch_size
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorWithPadding,
    PreTrainedTokenizerBase,
)

import pandas as pd

from dataset import DatasetReader, count_lines
from model import load_model_for_inference


def encode_string(text):
    return text.replace("\r", r"\r").replace("\n", r"\n").replace("\t", r"\t")


def get_dataloader(
        accelerator: Accelerator,
        filename: str,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int,
        max_length: int,
        prompt: str,
) -> DataLoader:
    dataset = DatasetReader(
        filename=filename,
        tokenizer=tokenizer,
        max_length=max_length,
        prompt=prompt,
    )
    if accelerator.distributed_type == DistributedType.XLA:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            padding="max_length",
            max_length=max_length,
            # label_pad_token_id=tokenizer.pad_token_id,
            return_tensors="pt",
        )
    else:
        data_collator = DataCollatorWithPadding(
            tokenizer,
            padding=True,
            # label_pad_token_id=tokenizer.pad_token_id,
            # max_length=max_length, No need to set max_length here, we already truncate in the preprocess function
            # pad_to_multiple_of=8,
            return_tensors="pt",
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=0,  # Disable multiprocessing
    )


def main(
        sentences_path: Optional[str],
        sentences_dir: Optional[str],
        files_extension: str,
        output_path: str,
        source_lang: Optional[str],
        target_lang: Optional[str],
        starting_batch_size: int,
        model_name: str = "facebook/m2m100_1.2B",
        lora_weights_name_or_path: str = None,
        force_auto_device_map: bool = False,
        precision: str = None,
        max_length: int = 256,
        num_beams: int = 4,
        num_return_sequences: int = 1,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        keep_special_tokens: bool = False,
        keep_tokenization_spaces: bool = False,
        repetition_penalty: float = None,
        prompt: str = None,
        trust_remote_code: bool = False,
        original_column: Optional[str] = None,
        translated_column: Optional[str] = None,
):
    accelerator = Accelerator()

    if force_auto_device_map and starting_batch_size >= 64:
        print(
            f"WARNING: You are using a very large batch size ({starting_batch_size}) and the auto_device_map  flag. "
            f"auto_device_map will offload model parameters to the CPU when they don't fit on the GPU VRAM. "
            f"If you use a very large batch size, it will offload a lot of parameters to the CPU and slow down the "
            f"inference. You should consider using a smaller batch size, i.e '--starting_batch_size 8'"
        )

    if sentences_path is None and sentences_dir is None:
        raise ValueError(
            "You must specify either --sentences_path or --sentences_dir. Use --help for more details."
        )

    if sentences_path is not None and sentences_dir is not None:
        raise ValueError(
            "You must specify either --sentences_path or --sentences_dir, not both. Use --help for more details."
        )

    if precision is None:
        quantization = None
        dtype = None
    elif precision == "8" or precision == "4":
        quantization = int(precision)
        dtype = None
    elif precision == "fp16":
        quantization = None
        dtype = "float16"
    elif precision == "bf16":
        quantization = None
        dtype = "bfloat16"
    elif precision == "32":
        quantization = None
        dtype = "float32"
    else:
        raise ValueError(
            f"Precision {precision} not supported. Please choose between 8, 4, fp16, bf16, 32 or None."
        )

    model, tokenizer = load_model_for_inference(
        weights_path=model_name,
        quantization=quantization,
        lora_weights_name_or_path=lora_weights_name_or_path,
        torch_dtype=dtype,
        force_auto_device_map=force_auto_device_map,
        trust_remote_code=trust_remote_code,
    )

    is_translation_model = hasattr(tokenizer, "lang_code_to_id")
    lang_code_to_idx = None

    if (
            is_translation_model
            and (source_lang is None or target_lang is None)
            and "small100" not in model_name
    ):
        raise ValueError(
            f"The model you are using requires a source and target language. "
            f"Please specify them with --source-lang and --target-lang. "
            f"The supported languages are: {tokenizer.lang_code_to_id.keys()}"
        )
    if not is_translation_model and (
            source_lang is not None or target_lang is not None
    ):
        if prompt is None:
            print(
                "WARNING: You are using a model that does not support source and target languages parameters "
                "but you specified them. You probably want to use m2m100/nllb200 for translation or "
                "set --prompt to define the task for you model. "
            )
        else:
            print(
                "WARNING: You are using a model that does not support source and target languages parameters "
                "but you specified them."
            )

    if prompt is not None and "%%SENTENCE%%" not in prompt:
        raise ValueError(
            f"The prompt must contain the %%SENTENCE%% token to indicate where the sentence should be inserted. "
            f"Your prompt: {prompt}"
        )

    if is_translation_model:
        try:
            _ = tokenizer.lang_code_to_id[source_lang]
        except KeyError:
            raise KeyError(
                f"Language {source_lang} not found in tokenizer. Available languages: {tokenizer.lang_code_to_id.keys()}"
            )
        tokenizer.src_lang = source_lang

        try:
            lang_code_to_idx = tokenizer.lang_code_to_id[target_lang]
        except KeyError:
            raise KeyError(
                f"Language {target_lang} not found in tokenizer. Available languages: {tokenizer.lang_code_to_id.keys()}"
            )
        if "small100" in model_name:
            tokenizer.tgt_lang = target_lang
            # We don't need to force the BOS token, so we set is_translation_model to False
            is_translation_model = False

    if model.config.model_type == "seamless_m4t":
        # Loading a seamless_m4t model, we need to set a few things to ensure compatibility

        supported_langs = tokenizer.additional_special_tokens
        supported_langs = [lang.replace("__", "") for lang in supported_langs]

        if source_lang is None or target_lang is None:
            raise ValueError(
                f"The model you are using requires a source and target language. "
                f"Please specify them with --source-lang and --target-lang. "
                f"The supported languages are: {supported_langs}"
            )

        if source_lang not in supported_langs:
            raise ValueError(
                f"Language {source_lang} not found in tokenizer. Available languages: {supported_langs}"
            )
        if target_lang not in supported_langs:
            raise ValueError(
                f"Language {target_lang} not found in tokenizer. Available languages: {supported_langs}"
            )

        tokenizer.src_lang = source_lang

    gen_kwargs = {
        "max_new_tokens": max_length,
        "num_beams": num_beams,
        "num_return_sequences": num_return_sequences,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_k": top_k,
        "top_p": top_p,
    }

    if repetition_penalty is not None:
        gen_kwargs["repetition_penalty"] = repetition_penalty

    if is_translation_model:
        gen_kwargs["forced_bos_token_id"] = lang_code_to_idx

    if model.config.model_type == "seamless_m4t":
        gen_kwargs["tgt_lang"] = target_lang

    if accelerator.is_main_process:
        print(
            f"** Translation **\n"
            f"Input file: {sentences_path}\n"
            f"Sentences dir: {sentences_dir}\n"
            f"Output file: {output_path}\n"
            f"Source language: {source_lang}\n"
            f"Target language: {target_lang}\n"
            f"Force target lang as BOS token: {is_translation_model}\n"
            f"Prompt: {prompt}\n"
            f"Starting batch size: {starting_batch_size}\n"
            f"Device: {str(accelerator.device).split(':')[0]}\n"
            f"Num. Devices: {accelerator.num_processes}\n"
            f"Distributed_type: {accelerator.distributed_type}\n"
            f"Max length: {max_length}\n"
            f"Quantization: {quantization}\n"
            f"Precision: {dtype}\n"
            f"Model: {model_name}\n"
            f"LoRA weights: {lora_weights_name_or_path}\n"
            f"Force auto device map: {force_auto_device_map}\n"
            f"Keep special tokens: {keep_special_tokens}\n"
            f"Keep tokenization spaces: {keep_tokenization_spaces}\n"
        )
        print("** Generation parameters **")
        print("\n".join(f"{k}: {v}" for k, v in gen_kwargs.items()))
        print("\n")

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inference(batch_size, sentences_path, output_path):
        nonlocal \
            model, \
            tokenizer, \
            max_length, \
            gen_kwargs, \
            precision, \
            prompt, \
            is_translation_model

        print(f"Translating {sentences_path} with batch size {batch_size}")

        total_lines: int = count_lines(sentences_path)

        data_loader = get_dataloader(
            accelerator=accelerator,
            filename=sentences_path,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            prompt=prompt,
        )

        model, data_loader = accelerator.prepare(model, data_loader)

        samples_seen: int = 0

        with tqdm(
                total=total_lines,
                desc="Dataset translation",
                leave=True,
                ascii=True,
                disable=(not accelerator.is_main_process),
        ) as pbar, open(output_path, "w", encoding="utf-8") as output_file:
            with torch.no_grad():
                for step, batch in enumerate(data_loader):
                    batch["input_ids"] = batch["input_ids"]
                    batch["attention_mask"] = batch["attention_mask"]

                    generated_tokens = accelerator.unwrap_model(model).generate(
                        **batch,
                        **gen_kwargs,
                    )

                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
                    )

                    generated_tokens = (
                        accelerator.gather(generated_tokens).cpu().numpy()
                    )

                    tgt_text = tokenizer.batch_decode(
                        generated_tokens,
                        skip_special_tokens=not keep_special_tokens,
                        clean_up_tokenization_spaces=not keep_tokenization_spaces,
                    )
                    if accelerator.is_main_process:
                        if (
                                step
                                == math.ceil(
                            math.ceil(total_lines / batch_size)
                            / accelerator.num_processes
                        )
                                - 1
                        ):
                            tgt_text = tgt_text[
                                       : (total_lines * num_return_sequences) - samples_seen
                                       ]
                        else:
                            samples_seen += len(tgt_text)

                        print(
                            "\n".join(
                                [encode_string(sentence) for sentence in tgt_text]
                            ),
                            file=output_file,
                        )

                    pbar.update(len(tgt_text) // gen_kwargs["num_return_sequences"])

        print(f"Translation done. Output written to {output_path}\n")

    @find_executable_batch_size(starting_batch_size=starting_batch_size)
    def inference_df(batch_size, sentences_path, df, original_column, translated_column, output_path, delimiter):
        nonlocal model, tokenizer, max_length, gen_kwargs, prompt, accelerator

        print(f"Translating {sentences_path} with batch size {batch_size}")

        # Do not drop any rows – use fillna to replace NaN with empty string
        # Preserve the original index
        df[original_column] = df[original_column].fillna("")

        # Custom dataset that iterates over the DataFrame rows and preserves indices.
        class DFPromptDataset(Dataset):
            def __init__(self, df, original_column, prompt, tokenizer, max_length):
                self.df = df
                self.indices = df.index.tolist()  # preserve original indices
                self.sentences = df[original_column].tolist()
                self.prompt = prompt
                self.tokenizer = tokenizer
                self.max_length = max_length

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                orig_idx = self.indices[idx]
                sentence = self.sentences[idx]
                # Build prompt string using only the sentence text.
                prompt_str = f"Translate Spanish to Portuguese (pt_PT): {sentence}"
                tokenized = self.tokenizer(
                    prompt_str,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                # Remove extra batch dimension for easier collation.
                tokenized = {k: v.squeeze(0) for k, v in tokenized.items()}
                return orig_idx, tokenized

        # Create dataset
        dataset = DFPromptDataset(df, original_column, prompt, tokenizer, max_length)

        # Create a data collator that will batch the tokenized dicts.
        # We define a custom collate_fn that also collects the row indices.
        if accelerator.distributed_type == DistributedType.XLA:
            data_collator = DataCollatorWithPadding(
                tokenizer,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
        else:
            data_collator = DataCollatorWithPadding(
                tokenizer,
                padding=True,
                return_tensors="pt",
            )

        def collate_fn(batch):
            # Batch is a list of tuples: (orig_idx, tokenized dict)
            indices, tokenized_list = zip(*batch)
            collated_inputs = data_collator(list(tokenized_list))
            return list(indices), collated_inputs

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=0,  # Disable multiprocessing for simplicity
        )

        # Prepare the model and dataloader with accelerator.
        model, data_loader = accelerator.prepare(model, data_loader)

        total_sentences = len(dataset)
        processed_count = 0  # counter for flush
        print(f"[DEBUG] Total sentences to translate: {total_sentences}")

        with tqdm(total=total_sentences, desc="Translating", unit="sentences") as pbar, torch.no_grad():
            for step, (indices, batch_inputs) in enumerate(data_loader):
                print(f"[DEBUG] Processing batch {step + 1} with {len(indices)} items...")
                generated_tokens = accelerator.unwrap_model(model).generate(
                    **batch_inputs, **gen_kwargs
                )
                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                print(f"[DEBUG] Raw model output (first 5): {translations[:5]}")

                # Clean extra content if needed.
                cleaned_translations = [t.split("ONLY INCLUDE THE TRANSLATION")[0].strip() for t in translations]
                print(f"[DEBUG] Cleaned translations (first 5): {cleaned_translations[:5]}")

                # Update the DataFrame at the correct indices.
                for orig_idx, translation in zip(indices, cleaned_translations):
                    df.at[orig_idx, translated_column] = translation

                processed_count += len(cleaned_translations)
                pbar.update(len(cleaned_translations))

                # Save progress every 50 translations.
                if processed_count >= 50:
                    df.to_csv(output_path, index=False, sep=delimiter)
                    print(f"[DEBUG] Saved {processed_count} translations to disk.")
                    processed_count = 0

            # Final save after processing all batches.
            df.to_csv(output_path, index=False, sep=delimiter)
            print(f"[DEBUG] Final save: output written to {output_path}")

        print(f"Translation completed. Output written to {output_path}\n")

    if sentences_path is not None:
        ext = os.path.splitext(sentences_path)[1].lower()
        if ext in ['.csv', '.tsv']:
            if original_column is None:
                raise ValueError("When using a CSV/TSV file, please specify --original_column")
            if translated_column is None:
                translated_column = "translation"  # Default new column name

            delimiter = ',' if ext == '.csv' else '\t'
            df = pd.read_csv(sentences_path, delimiter=delimiter)
            if original_column not in df.columns:
                raise ValueError(f"Column '{original_column}' not found in input file.")

            if translated_column not in df.columns:
                df[translated_column] = pd.NA

            inference_df(
                sentences_path=sentences_path,
                df=df,
                original_column=original_column,
                translated_column=translated_column,
                output_path=output_path,
                delimiter=delimiter
            )
            
            print(f"CSV/TSV translation done. Output saved to {output_path}\n")
        else:
            # Fall back to processing as a text file
            os.makedirs(os.path.abspath(os.path.dirname(output_path)), exist_ok=True)
            inference(sentences_path=sentences_path, output_path=output_path)

    if sentences_dir is not None:
        print(
            f"Translating all files in {sentences_dir}, with extension {files_extension}"
        )
        os.makedirs(os.path.abspath(output_path), exist_ok=True)
        for filename in glob.glob(
                os.path.join(
                    sentences_dir, f"*.{files_extension}" if files_extension else "*"
                )
        ):
            output_filename = os.path.join(output_path, os.path.basename(filename))
            inference(sentences_path=filename, output_path=output_filename)

    print("Translation done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the translation experiments")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--sentences_path",
        default=None,
        type=str,
        help="Path to a txt file containing the sentences to translate. One sentence per line.",
    )

    input_group.add_argument(
        "--sentences_dir",
        type=str,
        default=None,
        help="Path to a directory containing the sentences to translate. "
             "Sentences must be in  .txt files containing containing one sentence per line.",
    )

    parser.add_argument(
        "--files_extension",
        type=str,
        default="txt",
        help="If sentences_dir is specified, extension of the files to translate. Defaults to txt. "
             "If set to an empty string, we will translate all files in the directory.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to a txt file where the translated sentences will be written. If the input is a directory, "
             "the output will be a directory with the same structure.",
    )

    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        required=False,
        help="Source language id. See: supported_languages.md. Required for m2m100 and nllb200",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        required=False,
        help="Source language id. See: supported_languages.md. Required for m2m100 and nllb200",
    )

    parser.add_argument(
        "--starting_batch_size",
        type=int,
        default=128,
        help="Starting batch size, we will automatically reduce it if we find an OOM error."
             "If you use multiple devices, we will divide this number by the number of devices.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="facebook/m2m100_1.2B",
        help="Path to the model to use. See: https://huggingface.co/models",
    )

    parser.add_argument(
        "--lora_weights_name_or_path",
        type=str,
        default=None,
        help="If the model uses LoRA weights, path to those weights. See: https://github.com/huggingface/peft",
    )

    parser.add_argument(
        "--force_auto_device_map",
        action="store_true",
        help=" Whether to force the use of the auto device map. If set to True, "
             "the model will be split across GPUs and CPU to fit the model in memory. "
             "If set to False, a full copy of the model will be loaded  into each GPU. Defaults to False.",
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum number of tokens in the source sentence and generated sentence. "
             "Increase this value to translate longer sentences, at the cost of increasing memory usage.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=5,
        help="Number of beams for beam search, m2m10 author recommends 5, but it might use too much memory",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of possible translation to return for each sentence (num_return_sequences<=num_beams).",
    )

    parser.add_argument(
        "--precision",
        type=str,
        default=None,
        choices=["bf16", "fp16", "32", "4", "8"],
        help="Precision of the model. bf16, fp16 or 32, 8 , 4 "
             "(4bits/8bits quantification, requires bitsandbytes library: https://github.com/TimDettmers/bitsandbytes). "
             "If None, we will use the torch.dtype of the model weights.",
    )

    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Use sampling instead of beam search.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling, value used only if do_sample is True.",
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=100,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--top_p",
        type=float,
        default=0.75,
        help="If do_sample is True, will sample from the top k most likely tokens.",
    )

    parser.add_argument(
        "--keep_special_tokens",
        action="store_true",
        help="Keep special tokens in the decoded text.",
    )

    parser.add_argument(
        "--keep_tokenization_spaces",
        action="store_true",
        help="Do not clean spaces in the decoded text.",
    )

    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=None,
        help="Repetition penalty.",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt to use for generation. "
             "It must include the special token %%SENTENCE%% which will be replaced by the sentence to translate.",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="If set we will trust remote code in HuggingFace models. This is required for some models.",
    )
    parser.add_argument(
        "--original_column",
        type=str,
        default=None,
        help="(For CSV/TSV input) Name of the column containing text to translate."
    )
    parser.add_argument(
        "--translated_column",
        type=str,
        default=None,
        help="(For CSV/TSV input) Name of the column where the translated text should be stored."
    )

    args = parser.parse_args()

    main(
        sentences_path=args.sentences_path,
        sentences_dir=args.sentences_dir,
        files_extension=args.files_extension,
        output_path=args.output_path,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        starting_batch_size=args.starting_batch_size,
        model_name=args.model_name,
        max_length=args.max_length,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return_sequences,
        precision=args.precision,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        keep_special_tokens=args.keep_special_tokens,
        keep_tokenization_spaces=args.keep_tokenization_spaces,
        repetition_penalty=args.repetition_penalty,
        prompt=args.prompt,
        trust_remote_code=args.trust_remote_code,
        original_column=args.original_column,
        translated_column=args.translated_column,
    )
