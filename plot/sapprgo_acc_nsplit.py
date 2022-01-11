import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('plot/result.csv')


# nsplit_alphaが存在する行のみを抽出
df = df[~df['nsplit_alpha'].isnull()]

# data_file ごとに処理
for data_file in df['data_file'].unique():
    df_one = df[df['data_file'] == data_file]
    data_file_name = data_file.split('.')[0].split('/')[-1]
    # 縦軸をacc_test_mean(error barを2*acc_test_std)、横軸をnsplit_alphaとしたグラフを描画。横軸はlogspace

    # logspace がTrueならlogspace, Falseならlinspace で場合分け
    df_logspace = df_one[df_one['logspace']]
    df_linspace = df_one[df_one['logspace'] == False]
    plt.errorbar(df_logspace['nsplit_alpha'], df_logspace['acc_test_mean'], yerr=2 * df_logspace['acc_test_std'], fmt='o', label="logspace")
    plt.errorbar(df_linspace['nsplit_alpha'], df_linspace['acc_test_mean'], yerr=2 * df_linspace['acc_test_std'], fmt='o', label="linspace")
    plt.xlabel('Number of alphas')
    plt.ylabel('Test Accuracy (%)')
    plt.legend()
    # 保存
    plt.savefig('plot/sapprgo_acc_nsplit_' + data_file_name + '.png')
    plt.clf()
