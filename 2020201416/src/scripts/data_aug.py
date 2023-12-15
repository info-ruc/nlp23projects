import nlpaug.augmenter.word as naw
import pandas as pd
import tqdm
from data import read_data

def DataAugmentation(file_raw):
    back_translation_aug = naw.BackTranslationAug(
        from_model_name='facebook/wmt19-en-de', 
        to_model_name='facebook/wmt19-de-en'
    )

    text_raw, label, _ = read_data(file_raw)
    text_aug = []
    for text in tqdm.tqdm(text_raw):
        text_aug.append(back_translation_aug.augment(text))

    data_aug = pd.DataFrame(columns=['text', 'label'])
    data_aug['text'] = text_aug
    data_aug['label'] = label
    return data_aug