import os
import glob
import pandas as pd
import sys
import time
from tqdm import tqdm
from PIL import Image

from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset

from kobert_tokenizer import KoBERTTokenizer
from transformers import ViTFeatureExtractor, TrOCRProcessor, VisionEncoderDecoderModel, AdamW

import evaluate
from datasets import load_metric

import argparse

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text
        file_name = self.df['file_name'][idx] # 컬럼 file_name, text로 구성되어 있는 dataframe을 input으로 넣어야 함
        text = self.df['text'][idx]

        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_name).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values # processor : TrOCRProcessor.from_pretrained

        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text,
                                          padding="max_length",
                                          max_length=self.max_target_length).input_ids

        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def compute_cer(pred_ids, label_ids):
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return cer

if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='FineTuning')
    parser.add_argument('--simple_test', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test', type=int, default=0)
    args = parser.parse_args()

    # base_path
    base_path = os.getcwd().split('/src')[0]

    # load feature_extractor, tokenizer, processor
    encode = 'google/vit-base-patch16-224-in21k'
    decode = 'skt/kobert-base-v1'

    feature_extractor = ViTFeatureExtractor.from_pretrained(encode)
    tokenizer = KoBERTTokenizer.from_pretrained(decode)
    processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    # load train data list(csv)
    df = pd.read_csv(base_path + '/data/train.csv')
    df.rename({'img_path' : 'file_name', 'label' : 'text'}, axis=1, inplace=True)
    df['file_name'] = df.file_name.map(lambda x : x[1:])
    """
    file_list = glob.glob(base_path + '/data/train/*.png')
    file_list = pd.DataFrame(list(map(lambda x: os.path.basename(x).replace('.png',''), file_list)), columns=['id'])
    df = pd.merge(df, file_list, on='id', how='right')
    df.reset_index(drop=True, inplace=True)
    df = df.iloc[0:100,:] # for simple test
    """
    if args.simple_test:
        df = df.iloc[0:100, :]  # for simple test

    # train test split
    train_df, test_df = train_test_split(df, test_size=0.2)
    train_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # prepare dataset
    train_dataset = IAMDataset(root_dir=base_path + '/data',
                               df=train_df,
                               processor=processor)
    eval_dataset = IAMDataset(root_dir=base_path + '/data',
                               df=test_df,
                               processor=processor)
    print("Number of training examples:", len(train_dataset))
    print("Number of validation examples:", len(eval_dataset))

    # encoding test
    """
    encoding = train_dataset[0]
    for k,v in encoding.items():
      print(k, v.shape)
    
    image = Image.open(train_dataset.root_dir + train_df['file_name'][0]).convert("RGB")
    image
    
    labels = encoding['labels']
    labels[labels == -100] = processor.tokenizer.pad_token_id
    label_str = processor.decode(labels, skip_special_tokens=True)
    print(label_str)
    """

    # prepare dataloader
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4)

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model & setting
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode)
    model.to(device)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 20 # 최대 길이 : 여기서는 20자 미만으로 해도 될 듯
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    # load loss & optimizer
    cer_metric = evaluate.load('cer') # load_metric("cer")
    optimizer = AdamW(model.parameters(), lr=5e-5)

    print("=" * 20)
    print("=" * 20)
    print("\n\n\n\n\n>>> Start Fine Tunning")
    print(f">>> Using {device}")
    start_time = time.time()

    # fine-tunning
    for epoch in tqdm(range(args.epochs)):  # loop over the dataset multiple times
        # train
        model.train()
        train_loss = 0.0
        for batch in train_dataloader:
            # get the inputs
            for k, v in batch.items():
                batch[k] = v.to(device)

            # forward + backward + optimize
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()

        print(f"Loss after epoch {epoch}:", train_loss / len(train_dataloader))

        # evaluate
        model.eval()
        valid_cer = 0.0
        with torch.no_grad():
            for batch in eval_dataloader:
                # run batch generation
                outputs = model.generate(batch["pixel_values"].to(device))
                # compute metrics
                try:
                    cer = compute_cer(pred_ids=outputs, label_ids=batch["labels"])
                except:
                    cer = 1
                valid_cer += cer

        print("Validation CER:", valid_cer / len(eval_dataloader))

    model.save_pretrained(base_path + "/model/tr_ocr.pt")
    print(f">>> End. Total Using Time : {time.time() - start_time}")
    print("=" * 20)
    print("=" * 20)

# test
if args.test:
    image_path = glob.glob(base_path + '/data/train/*.png')[0]
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pred_ids = model.generate(pixel_values.to(device))
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    print(f"Sample image : {image_path}")
    print(f"Predict : {pred_str}")
