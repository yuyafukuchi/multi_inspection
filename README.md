<<<<<<< HEAD
<<<<<<< HEAD
# Readme
ニューマインド用撮影検査の行い方
- python == 3.7.0
- 画像ファイルはすべて.jpgにする

# Set Up
newmindフォルダ直下で行う.

```
$ pip install -r requirements/base.txt -U
```
以下のものが入る(はず). 作成者と同じ環境がこれなので, 適宜自分の環境に合わせて入れること.

- imageio==2.6.1
- joblib==0.14.1
- Keras==2.3.1
- Keras-Applications==1.0.8
- Keras-Preprocessing==1.1.0
- Markdown==3.1.1
- matplotlib==3.1.2
- numba==0.47.0
- numpy==1.18.1
- opencv-python==4.1.2.30
- Pillow==7.0.0
- pyspin==1.1.1
- scikit-image==0.16.2
- scikit-learn==0.21.3
- scipy==1.3.0
- seaborn==0.9.0
- sklearn==0.0
- spinnaker-python==1.27.0.48
- statsmodels==0.10.2
- tensorboard==2.1.0
- tensorflow==2.1.0
- tensorflow-estimator==2.1.0

また, 追加でSpinnakerおよびSpinViewなどの入ったSDKのパッケージをインストールする.
やり方は「カメラの撮影について.pdf」を参照.

# Usage

## project.py
新しく空のプロジェクトフォルダ(sample_project)を作成する.
```
$ python project.py
Enter a path for new project:sample_project
New project has been created.
```
プロジェクトディレクトリのパス名を聞かれるので入力すると作成される. 

```
sample_project
+-- dataset
|   +-- original_plain
|   +-- original_printed
|   +-- preprocessed
+-- distributions
|   +-- distances
|   +-- histograms
+-- inspection_targets
+-- models
```

## acquisition.py
SDKカメラを用いた撮影を行う. 途中の段階で「q」と押すと終了できる.
(撮影前にSDKカメラが接続されているかどうか確認)

### 撮影モードについて
以下の撮影モードを選べる
 - PLAIN : 印刷前の対象の画像を連続で撮影する. 保存先は(dataset_path)/original_plain/(category)
 - PRINTED : 印刷後の対象の画像を連続で撮影する. 保存先は(dataset_path)/original_printed/(category)
 - CONTI : 印刷前と印刷後の対象の画像を交互に連続で撮影する. 保存先は上記の2つに振り分けられる.
 - SINGLE : 1種類の画像のみ撮影する. 検品画像撮影のみ.


### 訓練用画像(または性能評価用画像)を撮影する場合
```
$ python acquisition.py
Enter training or inspection (t or i):t     # 訓練用画像を撮影する場合は't'と入力
Enter dataset path (q:quit):sample_project/dataset     # データセットを作るディレクトリを指定('q'で終了)
Enter category path (train/OK or test/OK or test/NG, q:quit):train/OK     # カテゴリ(保存パス)指定('q'で終了)
Enter print type (PLAIN or PRINTED or CONTI):CONTI     # 撮影モード指定
Enter start number:1     # 撮影ID(連番)の開始番号を指定
Press Enter to aquire PLAIN image (q:quit):     # Enterで撮影('q'で終了)
...
```
画像は'(id)_0.jpg'という名前で保存される.

### 検品用画像を撮影する場合
#### CONTIモード
```
$ python acquisition.py
Enter training or inspection (t or i):i     # 検品用画像を撮影する場合は'i'と入力
Enter inspection path (q:quit):sample_project/inspection_targets     # ディレクトリ指定('q'で終了)
Enter acquisition type (SINGLE or CONTI):CONTI     # 連続撮影する場合はCONTIを選択
Enter start number:1     # 撮影ID(連番)の開始番号を指定
Press Enter to aquire PLAIN image (q:quit):     # Enterで撮影('q'で終了)
Done! Next...
Press Enter to aquire PRINTED image (q:quit):     # Enterで撮影('q'で終了)
print('This pair has Done!')
print('Get aligned diff.')     # 前処理画像を取得
...
```
1セットの撮影ごとに印刷前/印刷後/差分/印刷前正規化/印刷後正規化の5つの画像が指定ディレクトリ直下の5つのフォルダ(pl,pr,d,pln,prn)の中に保存される.
名前は'(id)_pl_0.jpg', '(id)_pr_0.jpg', '(id)_d_0.jpg', '(id)_pln_0.jpg', '(id)_prn_0.jpg'となる

#### SINGLEモード
```
$ python acquisition.py
Enter training or inspection (t or i):i     # 検品用画像を撮影する場合は'i'と入力
Enter inspection path (q:quit):sample_project/inspection_targets     # ディレクトリ指定('q'で終了)
Enter acquisition type (SINGLE or CONTI):SINGLE     # 1種類の撮影をする場合はSINGLEを選択
Enter start number:1     # 撮影ID(連番)の開始番号を指定
Press Enter to aquire image (q:quit):     # Enterで撮影('q'で終了)
Done!
...
```
画像は(inspection_path)/(id)に'(id)_0.jpg'として保存される.


## preprocess.py
撮影した2枚の画像に画像処理(差分or正規化)を行う. 基本的に訓練用画像(または性能評価用画像)のディレクトリに対して行う.
撮影した画像のサイズが2248x2048である前提で処理を行っているので, サイズが異なる場合は適切にスクリプトを変えること.
位置合わせした出力画像のサイズは224x224にリサイズされている.

### モード
- DIRモード: データセットのoriginal_plainとoriginal_printedをフォルダごとまとめて処理を行う
- FILEモード: 画像2枚を直接指定して差分をとる

### 処理方法
- align_diff: 画像2枚を位置合わせして差分をとる
- normalize: 画像1枚(印刷後)を正規化(コントラスト調整)する

### フォルダごとまとめて訓練用画像(または性能評価用画像)に処理を行う場合
```
$ python preprocess.py
Enter process mode (FILE or DIR):DIR     # モード選択
align_diff or normalize? (a or n):n     # 処理方法選択
Enter dataset path (q:quit):     # ディレクトリ指定('q'で終了)
Enter category path (train/OK (default) or test/OK or test/NG, q:quit):     #カテゴリ指定(特に指定しなければtrain/OK, 'q'で終了)
Default path "train/OK" is selected.
```
画像は(dataset_path)/preprocessed/(category)に保存される.

### 画像2枚を直接指定して処理を行う場合
```
$ python preprocess.py
Enter process mode (FILE or DIR):FILE     # モード選択
align_diff or normalize? (a or n):a     # 処理方法選択
Enter PLAIN image path:plain.jpg     # 印刷前の画像パス
Enter PRINTED image path:printed_jpg     # 印刷後の画像パス
Enter save path:save_dir         # 保存先ディレクトリのパス
```
画像は(save_dir)/diff.jpgとして保存される.



## rotate.py
画像を回転させて水増しする.
角度は-1,1(度)で, 2枚(元画像も含めると3枚)となる.
```
$ python rotate.py
Enter dataset path (q:quit):     # ディレクトリ指定('q'で終了)
Enter category path (train/OK (default) or test/OK or test/NG, q:quit):     #カテゴリ指定(特に指定しなければtrain/OK, 'q'で終了)
```
(dataset_path)/preprocessed/(category)以下の画像のみが水増しされる. original_plainとoriginal_printed以下は増えないので注意.


## translate.py
画像を平行移動させて水増しする.
左右上下に1ピクセルずつずらして4枚(元画像も含めると5枚)となる.

```
$ python translate.py
Enter dataset path (q:quit):     # ディレクトリ指定('q'で終了)
Enter category path (train/OK (default) or test/OK or test/NG, q:quit):     #カテゴリ指定(特に指定しなければtrain/OK, 'q'で終了)
```
(dataset_path)/preprocessed/(category)以下の画像のみが水増しされる. original_plainとoriginal_printed以下は増えないので注意.


## multi_inspection.py
(複数領域の)訓練/性能評価および検品を行う.
```
$ python multi_inspection.py
Enter the project root path (q:quit):     # データセットのディレクトリ指定('q'で終了)
Do you train and test? (y or n):y     # 訓練を行う場合は'y'
...
Do you train and test? (y or n):y     # 性能評価を行う場合は'y'
...
Inspection starts.
alpha value (default: 0.0013):     # 検品の判定基準となるα値の指定(defaultは0.0013, 3σ)
capture id (q:quit):     # 検品する画像のIDを指定して実行('q'で終了)
```
acquisition.pyで撮影した検品対象画像を指定する場合, captured idは'(id)_prn', '(id)_d'などのようになるので注意.


=======
# multi_inspection
>>>>>>> origin/master
=======
# multi_inspection
>>>>>>> ca72e4785f5cc7dd86db798b8eee724a99ce347d
