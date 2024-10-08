import matplotlib.pyplot as plt
import mplfinance as mpf
import yfinance as yf
from PIL import Image
import pandas as pd
import numpy as np
import talib
import io
import os
import gc

def get_stock_data(ticker, start_date):
    data = yf.download(ticker, start=start_date)
    data['High'] = (1 + (data['High'] - data['Close']) / data['Close']) * data['Adj Close']
    data['Low'] = (1 + (data['Low'] - data['Close']) / data['Close']) * data['Adj Close']
    data['Open'] = (1 + (data['Open'] - data['Close']) / data['Close']) * data['Adj Close']
    data['Close'] = data['Adj Close']
    return data

def create_talib_functions():
    pattern_names = talib.get_function_groups()['Pattern Recognition']
    talib_functions = {}
    for pattern_name in pattern_names:
        talib_functions[pattern_name] = lambda prices, func=pattern_name: getattr(talib, func)(
            prices['Open'], prices['High'], prices['Low'], prices['Close']
        )
    return talib_functions

def test_single_pattern(pattern_data):
    return pattern_data.abs().sum(axis=1).values[-1] == 100

def identify_pattern_data(data, talib_functions):
    pattern_data = pd.DataFrame()
    for pattern_name, pattern_func in talib_functions.items():
        pattern_data[pattern_name] = pattern_func(data)
    return pattern_data

def combine_direction_and_pattern_data(pattern_data):
    direction = "Bullish" if pattern_data.sum(axis=1).values[-1] > 0 else 'Bearish'
    pattern = pattern_data.abs().idxmax(axis=1).values[-1]
    return direction, pattern

custom_style = mpf.make_mpf_style(
    base_mpf_style='classic',  # Using 'classic' style as a base
    marketcolors=mpf.make_marketcolors(
        up='white',      # Color for upward candles/bars
        down='black',    # Color for downward candles/bars
        wick={'up': 'black', 'down': 'black'},  # Wicks are black for both up and down
        edge={'up': 'black', 'down': 'black'},  # Edges are black for both up and down
        volume='black',  # Black volume bars
    ),
    facecolor='white',   # Background color (white)
    gridcolor='gray',     # Grid lines in gray
    gridstyle=''
)

def create_image(subset, fname, label):

    buffer = io.BytesIO()

    mpf.plot(subset, type='candle', style=custom_style, volume=False, show_nontrading=False, 
             axisoff=True, update_width_config=dict(candle_width=0.70 , candle_linewidth=0.5), tight_layout=True,
             figsize=(1.25,3) ,savefig=dict(fname=buffer, bbox_inches='tight', pad_inches=0.05, dpi=100, transparent=True))

    plt.savefig(buffer, format='png', bbox_inches='tight', pad_inches=0.05, transparent=True)
    plt.close()  

    buffer.seek(0)

    image = Image.open(buffer)

    image = image.convert('L')

    try:
        image.save(fname)  
    except:  
        os.mkdir(f'all_data/{label}')
        image.save(fname)

    buffer.close()

    del image

def get_labels():
    labels = {}
    pattern_names = talib.get_function_groups()['Pattern Recognition']
    signals = ['Bullish', 'Bearish']
    for signal in signals:
        for pattern_name in pattern_names:
            labels[f'{signal} {pattern_name}'] = len(labels)
    return labels

def get_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)
    sp500_df = sp500_table[0]
    tickers = sp500_df['Symbol'].tolist()
    return tickers

def prep_tickers(tickers):
    tickers.remove('AMTM')
    tickers.remove('BRK.B')
    tickers.insert(0, 'BRK-B')
    tickers.remove('BF.B')
    tickers.insert(0, 'BF-B')
    tickers.remove('GEV')
    tickers.remove('SW')
    tickers.remove('SOLV')
    tickers.remove('VLTO')
    return tickers

def create_pattern_data(ticker, start_date):
    data = get_stock_data(ticker, start_date)
    talib_functions = create_talib_functions()
    labels = get_labels()
    for start_idx in range(0, data.shape[0] - 10):
        subset = data.iloc[start_idx:start_idx+10]
        pattern_data = identify_pattern_data(subset, talib_functions)
        single_pattern = test_single_pattern(pattern_data)
        if single_pattern:
            direction, pattern = combine_direction_and_pattern_data(pattern_data)
            label = labels[f"{direction} {pattern}"]
            date = subset.index[-1].strftime('%Y_%m_%d')
            fname = f'all_data/{label}/{ticker}_{direction}_{pattern}_{date}.png'
            create_image(subset, fname, label)
    return

def main():
    tickers = get_tickers()
    tickers = prep_tickers(tickers)
    start_date = '2010-01-01'
    for ticker in tickers:
        try:
            create_pattern_data(ticker, start_date)
            print(f"Created data for {ticker}")
        except:
            print(f"Failed to create data for {ticker}")

if __name__ == '__main__':
    main()

