import argparse
import json
import os
import re
import random
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_vila import eval_model
from llava.model.builder import load_pretrained_model
from tqdm import tqdm

VILA_PATH = "/gscratch/sciencehub/zanqil/VILA"
LLAVA_PATH = "/gscratch/sciencehub/zanqil/LLaVA"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=os.path.join(VILA_PATH, "checkpoints/v1_5-3b-s2-ft-shelf-arm"))
    parser.add_argument('--input-file', type=str, default=os.path.join(LLAVA_PATH, "data/stow/test/dataset.json"))
    return parser.parse_args()

def load_prompts():
    prompts = {}
    prompt_path = os.path.join(VILA_PATH, "prompts")
    files = os.listdir(prompt_path)
    for file in files:
        with open(os.path.join(prompt_path, file), "r") as f:
            name = file.split(".")[0]
            prompts[name] = f.read()
    return prompts

if __name__ == "__main__":
    args = get_args()
    model_path = args.model_path
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_name=get_model_name_from_path(model_path)
    )

    prompts = load_prompts()
    prompt = prompts['object_detection_xyxy']
    image_folder = os.path.join(LLAVA_PATH, "data/stow")
    dataset_json = json.load(open(args.input_file, "r"))
    image_files = [os.path.join(image_folder, x["image"]) for x in dataset_json]

    res = []

    for x in tqdm(dataset_json):
        image_file = os.path.join(image_folder, x["image"])
        eval_args = type('Args', (), {
            "model_path": model_path,
            "model_base": None,
            "query": prompt,
            "conv_mode": None,
            "image_file": image_file,
            "video_file": None,
            "sep": ",",
            "temperature": 0.2,
            "top_p": None,
            "num_beams": 1,
            "max_new_tokens": 512
        })()

        output = eval_model(eval_args, model, tokenizer, image_processor)
        res.append({
            "id": x["id"],
            "text": output
        })

    predictions_path = os.path.join(VILA_PATH, "predictions")
    output_file = os.path.join(predictions_path, f"{args.model_path.split('/')[-1]}.json")
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    with open(output_file, 'w') as f:
        json.dump(res, f)
    

    
