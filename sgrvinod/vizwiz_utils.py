import pandas as pd
import json
import os
import spacy
import random

spacy_eng = spacy.load('en_core_web_sm')


def vizwiz_to_karpathy_split(vizwiz_folder, output_file='dataset/dataset_vizwiz.json'):
    train_list = get_vizwiz_caption_karpathy_split(vizwiz_folder, "train")
    val_list = get_vizwiz_caption_karpathy_split(vizwiz_folder, "val")
    val_list, test_list = split_list(val_list, 75)
    print(f'val list len {len(val_list)}')
    print(f'test list len {len(test_list)}')

    with open(output_file, 'w') as f:
        dataset = train_list + val_list + test_list
        dataset = {
            'images': dataset
        }
        json.dump(dataset, f)
    
    
def split_list(a_list, percent):
    random.shuffle(a_list)
    # 100 - len
    # percent - x

    # x = percent * len // 100
    part = (percent * len(a_list)) // 100
    return a_list[:part], a_list[part:]


def get_vizwiz_caption_karpathy_split(captions_folder, split):
    captions_file = os.path.join(captions_folder, split + '.json')

    dataset = json.load(open(captions_file, 'r'))
    images = pd.json_normalize(dataset['images'])
    captions = pd.json_normalize(dataset['annotations'])
    merged = pd.merge(
        images,
        captions,
        left_on='id',
        right_on='image_id'
    ) \
        .query('is_rejected == False & is_precanned == False') \
        .rename(columns={"file_name": "image"}) \
        .filter(items=['image', 'caption']) \
        .groupby('image')\
        .agg({'caption': lambda x: list(x)}) \
        .reset_index()

    print(merged.head())
    print(merged.shape)

    karpathy_split = []
    for _, row in merged.iterrows():
        sentences = [{'tokens': tokenizer_eng(text), 'raw': text} for text in row['caption']]

        image = {
            'sentences': sentences,
            'split': split,
            'filename': row['image']
        }
        karpathy_split.append(image)

    return karpathy_split


def tokenizer_eng(text):
    return [tok.text.lower() for tok in spacy_eng.tokenizer(text.rstrip('.'))]


if __name__ == "__main__":
    vizwiz_to_karpathy_split(
        "C:/Users/Irina/Desktop/ML/dataset/annotations",
        "C:/Users/Irina/Desktop/ML/dataset/dataset_vizwiz.json"
    )
    # data = get_vizwiz_caption_karpathy_split(
    #     "C:/Users/Irina/Desktop/ML/dataset/annotations", "train"
    # )
    # print(json.dumps(data[0:5], indent=2))

