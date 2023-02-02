import os
import glob
import re
from PIL import Image
import torch
from kobert_tokenizer import KoBERTTokenizer
from transformers import ViTFeatureExtractor, TrOCRProcessor, VisionEncoderDecoderModel, AdamW

import argparse

if __name__ == '__main__':

    # argparse
    parser = argparse.ArgumentParser(description='Predict')
    parser.add_argument('--image_index', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=os.getcwd().split('/src')[0] + '/model')
    args = parser.parse_args()

    # base_path
    base_path = os.getcwd().split('/src')[0]

    # load feature_extractor, tokenizer, processor
    encode = 'google/vit-base-patch16-224-in21k'
    decode = 'skt/kobert-base-v1'

    feature_extractor = ViTFeatureExtractor.from_pretrained(encode)
    tokenizer = KoBERTTokenizer.from_pretrained(decode)
    processor = TrOCRProcessor(image_processor=feature_extractor, tokenizer=tokenizer)

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model & setting
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(encode, decode)
    pretrained_list = glob.glob(args.save_path + '/*.pt')
    start_epoch = 0
    if len(pretrained_list) > 0:
        pretrained_path = pretrained_list[-1]
        start_epoch = int(re.findall('[0-9]+.pt', pretrained_path)[0].replace('.pt', '')) + 1
        model = VisionEncoderDecoderModel.from_pretrained(pretrained_path)
        print("=" * 20, flush=True)
        print("=" * 20, flush=True)
        print(f">>> Existence of model in progress for learning\nEpoch : {start_epoch}", flush=True)
    model.to(device)

    # set special tokens used for creating the decoder_input_ids from the labels
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = 50  # 최대 길이 : 여기서는 50자 미만으로 해도 될 듯
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4

    print("=" * 20, flush=True)
    print("=" * 20, flush=True)
    print(">>> Start Predict", flush=True)
    print(f">>> Using {device}", flush=True)

    image_path = glob.glob(base_path + '/data/train/*.png')[args.image_index]
    image = Image.open(image_path).convert("RGB")
    image.show()
    image.save('test_image.jpg', "JPEG")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pred_ids = model.generate(pixel_values.to(device))
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    print(f"Sample image : {image_path}", flush=True)
    print(f"Predict : {pred_str}", flush=True)