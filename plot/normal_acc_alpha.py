import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('plot/result.csv')


# alphaが存在する行のみを抽出
df = df[~df['alpha'].isnull()]

# data_file ごとに処理
for data_file in df['data_file'].unique():
    df_one = df[df['data_file'] == data_file]
    # 縦軸をacc_test_mean(error barを2*acc_test_std)、横軸をalphaとしたグラフを描画
    plt.errorbar(df_one['alpha'], df_one['acc_test_mean'], yerr=2 * df_one['acc_test_std'], label=data_file)
