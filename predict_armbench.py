import argparse
import json
import os
import re
import random
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_vila import eval_model
from llava.model.builder import load_pretrained_model
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-file', type=str, default="../LLaVA/armbench/train/dataset100.json")
    parser.add_argument('--output-file', type=str, default="predictions/vila-3b-s2-10e.json")
    parser.add_argument('--split', type=str, default='test')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model_path = "checkpoints"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_name=get_model_name_from_path(model_path)
    )

    prompt = "Perform object detection on the given image. First, resize the image to 1x1, then answer: what are the bounding boxes of the  objects in the image? Output should be in the format of a list of lists, where each list represents a bounding box: [[x_min, y_min, width, height], [x_min, y_min, width, height], ...] \n\nx_min and y_min are placeholder for the coordinate of the top left corner of an object bounding box, width and height are placeholder for the width and height of the bounding box. \n\nThe x_min, y_min, width and height of the bounding box should be positive decimal numbers between 0 and 1. \n\n \n\nFor example, if there are two objects in the image, the bounding boxes of the objects could be [[0.13, 0.24, 0.37, 0.4], [0.5, 0.55, 0.23, 0.25]]."
    image_folder = "../LLaVA/armbench/images"
    dataset_json = json.load(open(args.input_file, "r"))
    image_files = [os.path.join(image_folder, x["image"]) for x in dataset_json]

    res = []

    for image_file in tqdm(image_files):
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
            "image_id": image_file,
            "text": output
        })
    
    if not os.path.exists(os.path.dirname(args.output_file)):
        os.makedirs(os.path.dirname(args.output_file))

    with open(args.output_file, 'w') as f:
        json.dump(res, f)
    

    
