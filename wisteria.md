# Wisteria

# ジョブ操作コマンド

![image](https://user-images.githubusercontent.com/35361533/140845688-b1165d3d-d727-4f09-84aa-d8b20196016e.png)


![image](https://user-images.githubusercontent.com/35361533/140845722-290c2c79-2c70-4e05-9cdb-68ed14d23e7a.png)


![image](https://user-images.githubusercontent.com/35361533/140845732-8c0be6c7-0bf5-41e2-9ded-16e997ecfffa.png)


# 実行
## インタラクティブ
```
pjsub --interact -L rscgrp=interactive-a,node=1 -g gs54

module load gcc/8.3.1
module load python/3.8.12
module load cuda/11.1

python3 demo.py

```

## batch
```
pjsub run.sh
```


# 環境

Environment Modulesを使用してシステムにインストール済の環境を使用する場合、システム環境には追加パッケージをインストールすることはできません。
そのため、共有ファイルシステム領域(/work/groupname/username)にご自身でpython環境を構築し、必要なパッケージをインストールしてください。

'''
cd /work/gs54/s54002
pjsub --interact -L rscgrp=interactive-a,node=1 -g gs54

module load gcc/8.3.1
module load python/3.8.12
module load cuda/11.1

python3 -mvenv  venv


source venv/bin/activate

pip3 install --upgrade pip setuptools wheel
pip3 install torch==1.8.1
pip3 install -e .
'''
