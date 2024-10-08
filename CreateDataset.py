import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import talib
import shutil
import os

caption_labels = {
    'Bullish CDL2CROWS': 'bullish two crows',
    'Bullish CDL3BLACKCROWS': 'bullish three black crows',
    'Bullish CDL3INSIDE': 'bullish three inside up',
    'Bullish CDL3LINESTRIKE': 'bullish three-line strike',
    'Bullish CDL3OUTSIDE': 'bullish three outside up',
    'Bullish CDL3STARSINSOUTH': 'bullish three stars in the south',
    'Bullish CDL3WHITESOLDIERS': 'bullish three white soldiers',
    'Bullish CDLABANDONEDBABY': 'bullish abandoned baby',
    'Bullish CDLADVANCEBLOCK': 'bullish advance block',
    'Bullish CDLBELTHOLD': 'bullish belt-hold',
    'Bullish CDLBREAKAWAY': 'bullish breakaway',
    'Bullish CDLCLOSINGMARUBOZU': 'bullish closing marubozu',
    'Bullish CDLCONCEALBABYSWALL': 'bullish conceal baby swallow',
    'Bullish CDLCOUNTERATTACK': 'bullish counterattack',
    'Bullish CDLDARKCLOUDCOVER': 'bullish dark cloud cover',
    'Bullish CDLDOJI': 'bullish doji',
    'Bullish CDLDOJISTAR': 'bullish doji star',
    'Bullish CDLDRAGONFLYDOJI': 'bullish dragonfly doji',
    'Bullish CDLENGULFING': 'bullish engulfing',
    'Bullish CDLEVENINGDOJISTAR': 'bullish evening doji star',
    'Bullish CDLEVENINGSTAR': 'bullish evening star',
    'Bullish CDLGAPSIDESIDEWHITE': 'bullish up-gap side-by-side white lines',
    'Bullish CDLGRAVESTONEDOJI': 'bullish gravestone doji',
    'Bullish CDLHAMMER': 'bullish hammer',
    'Bullish CDLHANGINGMAN': 'bullish hanging man',
    'Bullish CDLHARAMI': 'bullish harami',
    'Bullish CDLHARAMICROSS': 'bullish harami cross',
    'Bullish CDLHIGHWAVE': 'bullish high-wave candle',
    'Bullish CDLHIKKAKE': 'bullish hikkake',
    'Bullish CDLHIKKAKEMOD': 'bullish modified hikkake',
    'Bullish CDLHOMINGPIGEON': 'bullish homing pigeon',
    'Bullish CDLIDENTICAL3CROWS': 'bullish identical three crows',
    'Bullish CDLINNECK': 'bullish in-neck',
    'Bullish CDLINVERTEDHAMMER': 'bullish inverted hammer',
    'Bullish CDLKICKING': 'bullish kicking',
    'Bullish CDLKICKINGBYLENGTH': 'bullish kicking',
    'Bullish CDLLADDERBOTTOM': 'bullish ladder bottom',
    'Bullish CDLLONGLEGGEDDOJI': 'bullish long-legged doji',
    'Bullish CDLLONGLINE': 'bullish long line candle',
    'Bullish CDLMARUBOZU': 'bullish marubozu',
    'Bullish CDLMATCHINGLOW': 'bullish matching low',
    'Bullish CDLMATHOLD': 'bullish mat hold',
    'Bullish CDLMORNINGDOJISTAR': 'bullish morning doji star',
    'Bullish CDLMORNINGSTAR': 'bullish morning star',
    'Bullish CDLONNECK': 'bullish on-neck',
    'Bullish CDLPIERCING': 'bullish piercing',
    'Bullish CDLRICKSHAWMAN': 'bullish rickshaw man',
    'Bullish CDLRISEFALL3METHODS': 'bullish rising three methods',
    'Bullish CDLSEPARATINGLINES': 'bullish separating lines',
    'Bullish CDLSHOOTINGSTAR': 'bullish shooting star',
    'Bullish CDLSHORTLINE': 'bullish short line candle',
    'Bullish CDLSPINNINGTOP': 'bullish spinning top',
    'Bullish CDLSTALLEDPATTERN': 'bullish stalled',
    'Bullish CDLSTICKSANDWICH': 'bullish stick sandwich',
    'Bullish CDLTAKURI': 'bullish takuri',
    'Bullish CDLTASUKIGAP': 'bullish tasuki gap',
    'Bullish CDLTHRUSTING': 'bullish thrusting',
    'Bullish CDLTRISTAR': 'bullish tristar',
    'Bullish CDLUNIQUE3RIVER': 'bullish unique three river',
    'Bullish CDLUPSIDEGAP2CROWS': 'bullish upside gap two crows',
    'Bullish CDLXSIDEGAP3METHODS': 'bullish upside gap three methods',
    'Bearish CDL2CROWS': 'bearish two crows',
    'Bearish CDL3BLACKCROWS': 'bearish three black crows',
    'Bearish CDL3INSIDE': 'bearish three inside down',
    'Bearish CDL3LINESTRIKE': 'bearish three-line strike',
    'Bearish CDL3OUTSIDE': 'bearish three outside down',
    'Bearish CDL3STARSINSOUTH': 'bearish three stars in the south',
    'Bearish CDL3WHITESOLDIERS': 'bearish three white soldiers',
    'Bearish CDLABANDONEDBABY': 'bearish abandoned baby',
    'Bearish CDLADVANCEBLOCK': 'bearish advance block',
    'Bearish CDLBELTHOLD': 'bearish belt-hold',
    'Bearish CDLBREAKAWAY': 'bearish breakaway',
    'Bearish CDLCLOSINGMARUBOZU': 'bearish closing marubozu',
    'Bearish CDLCONCEALBABYSWALL': 'bearish conceal baby swallow',
    'Bearish CDLCOUNTERATTACK': 'bearish counterattack',
    'Bearish CDLDARKCLOUDCOVER': 'bearish dark cloud cover',
    'Bearish CDLDOJI': 'bearish doji',
    'Bearish CDLDOJISTAR': 'bearish doji star',
    'Bearish CDLDRAGONFLYDOJI': 'bearish dragonfly doji',
    'Bearish CDLENGULFING': 'bearish engulfing',
    'Bearish CDLEVENINGDOJISTAR': 'bearish evening doji star',
    'Bearish CDLEVENINGSTAR': 'bearish evening star',
    'Bearish CDLGAPSIDESIDEWHITE': 'bearish down-gap side-by-side white lines',
    'Bearish CDLGRAVESTONEDOJI': 'bearish gravestone doji',
    'Bearish CDLHAMMER': 'bearish hammer',
    'Bearish CDLHANGINGMAN': 'bearish hanging man',
    'Bearish CDLHARAMI': 'bearish harami',
    'Bearish CDLHARAMICROSS': 'bearish harami cross',
    'Bearish CDLHIGHWAVE': 'bearish high-wave candle',
    'Bearish CDLHIKKAKE': 'bearish hikkake',
    'Bearish CDLHIKKAKEMOD': 'bearish modified hikkake',
    'Bearish CDLHOMINGPIGEON': 'bearish homing pigeon',
    'Bearish CDLIDENTICAL3CROWS': 'bearish identical three crows',
    'Bearish CDLINNECK': 'bearish in-neck',
    'Bearish CDLINVERTEDHAMMER': 'bearish inverted hammer',
    'Bearish CDLKICKING': 'bearish kicking',
    'Bearish CDLKICKINGBYLENGTH': 'bearish kicking',
    'Bearish CDLLADDERBOTTOM': 'bearish ladder bottom',
    'Bearish CDLLONGLEGGEDDOJI': 'bearish long-legged doji',
    'Bearish CDLLONGLINE': 'bearish long line candle',
    'Bearish CDLMARUBOZU': 'bearish marubozu',
    'Bearish CDLMATCHINGLOW': 'bearish matching low',
    'Bearish CDLMATHOLD': 'bearish mat hold',
    'Bearish CDLMORNINGDOJISTAR': 'bearish morning doji star',
    'Bearish CDLMORNINGSTAR': 'bearish morning star',
    'Bearish CDLONNECK': 'bearish on-neck',
    'Bearish CDLPIERCING': 'bearish piercing',
    'Bearish CDLRICKSHAWMAN': 'bear rickshaw man',
    'Bearish CDLRISEFALL3METHODS': 'bearish falling three methods',
    'Bearish CDLSEPARATINGLINES': 'bearish separating lines',
    'Bearish CDLSHOOTINGSTAR': 'bearish shooting star',
    'Bearish CDLSHORTLINE': 'bearish short line candle',
    'Bearish CDLSPINNINGTOP': 'bearish spinning top',
    'Bearish CDLSTALLEDPATTERN': 'bearish stalled',
    'Bearish CDLSTICKSANDWICH': 'bearish stick sandwich',
    'Bearish CDLTAKURI': 'bearish takuri',
    'Bearish CDLTASUKIGAP': 'bearish tasuki gap',
    'Bearish CDLTHRUSTING': 'bearish thrusting',
    'Bearish CDLTRISTAR': 'bearish tristar',
    'Bearish CDLUNIQUE3RIVER': 'bearish unique three river',
    'Bearish CDLUPSIDEGAP2CROWS': 'bearish upside gap two crows',
    'Bearish CDLXSIDEGAP3METHODS': 'bearish downside gap three methods'
}

def get_all_labels():
    labels = {}
    pattern_names = talib.get_function_groups()['Pattern Recognition']
    signals = ['Bullish', 'Bearish']
    for signal in signals:
        for pattern_name in pattern_names:
            labels[len(labels)] = 'A ' + caption_labels[f'{signal} {pattern_name}'] + ' pattern'
    return labels

def remove_data_folders():
    if os.path.exists(f"data/train"):
        shutil.rmtree(f"data/train")
    os.makedirs(f"data/train")
    if os.path.exists(f"data/val"):
        shutil.rmtree(f"data/val")
    if os.path.exists(f"data/test"):
        shutil.rmtree(f"data/test")
    os.makedirs(f"data/test")

def create_data_folders(label):
    os.mkdir(f'data/train/{label}')
    os.mkdir(f'data/test/{label}')

def sample_files(label, N):
    files = os.listdir(f"all_data/{label}")
    np.random.shuffle(files)
    files = files[:N]
    train_files = files[:int(len(files)*0.95)]
    test_files = files[int(len(files)*0.95):]
    return train_files, test_files

def copy_files(files, label, idx, folder):
    for file in files:
        shutil.copy(f"all_data/{label}/{file}", f"data/{folder}/{idx}/{file}")

def create_dataset(N):
    keep_labels = os.listdir("all_data")
    print('There are', len(keep_labels), 'labels with at least', N, 'samples')
    remove_data_folders()
    for idx, label in enumerate(keep_labels):
        create_data_folders(idx)
        train_files, test_files = sample_files(label, N)
        copy_files(train_files, label, idx, "train")
        copy_files(test_files, label, idx, "test")

def main():
    create_dataset(3000)

if __name__ == "__main__":
    main()