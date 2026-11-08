# 獣医X線撮影条件・画質評価シミュレーター

Python / Streamlitによる教育用の計算・可視化ツールです。**アプリv0.1.3、数式モデルv0.1.0**。kVp・mAs・SID・体厚を操作し、検出器入射空気カーマ、仮定した校正関係によるEI、DI、理論ノイズ・SNRと合成画像を比較します。

この配布物は公開準備版です。**GitHubへのアップロードとStreamlit Community Cloudへのデプロイは未実施**です。公開は担当教員による内容・公開範囲の確認後に行います。装置固有の妥当性や学術的な正式承認は未確認です。

学生の操作は仮想撮影に限り、実際のX線照射は行いません。実測データは使用していません。EIは患者被ばく線量ではなく、臨床条件の推奨、診断可否や再撮影の判定、自動採点は行いません。

## ローカルで起動する

Python 3.13を使用します。確認した開発環境はPython 3.13.0です。以下のフォルダー名は、展開先または取得したリポジトリ名に置き換えてください。

macOS / Linux:

```sh
cd streamlit_release_v0.1.3
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Windows PowerShell:

```powershell
cd streamlit_release_v0.1.3
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

ブラウザーで[ローカルアプリ](http://localhost:8501)を開きます。既定のポートは8501です。別のポートを使う場合は起動コマンド末尾に`--server.port 8503`などを追加します。終了は実行中のターミナルで`Ctrl+C`です。

## 実装している機能

| 画面 | 内容 |
| --- | --- |
| サイドバー | kVp、mAs、SID、体厚、仮定した物理係数、目標EI、表示設定 |
| 模擬画像 | 合成パターン選択、同条件の標準・強調表示、固定基準・現在条件の比較、犬の撮影配置模式図 |
| 応答曲線 | ほかの条件を固定したK・EI・DI・相対量子ノイズの曲線 |
| 条件を比較 | 最大8記録、同時に最大4画像、選択記録の削除、CSV保存 |
| 計算モデル・仕様 | 数式と仮定、規格の定義との区別、根拠資料 |

- 初期パターンは512×512画素。左上に従来の256×256幾何学パターン、右上にポメラニアン風の細部、下部に幅・間隔8/6/4/3/2/1画素の線対6群を配置します。従来の256×256も選択できます。いずれも合成信号で、犬のX線像や解剖モデルではありません。
- `C ~ Poisson(NS)`、`I=C/N`で生成し、表示だけを`J=S+g(I−S)`、g=1/2/3/5で強調します。強調によってK・EI・DI・理論ノイズ・SNRは変わりません。実測ノイズや低線量での再撮影を再現する処理ではありません。
- 表示窓は0–1.2で固定し、1/2/4倍で全体または中央領域を最近傍法で表示します。クリップ率は表示領域内の値です。拡大は物理的なぼけや鮮鋭度の変更ではなく、線対の目視はMTF測定ではありません。
- 乱数は固定seedまたは物理条件変更時に別標本を生成するモードを選べ、手動再生成もできます。目標EI・強調・拡大の変更ではseedを変えません。パターン切り替えはseedを保ちますが、信号と標本配置を変えます。
- すべての比較画像に現在の選択パターン・表示設定を共通適用します。固定基準は条件とseedを保持します。CSVでは保存時の情報と表示時の`view_`情報を分け、版情報も記録します。

## 記録とデータの扱い

比較記録は各ブラウザー接続のサーバー側セッションメモリーに保持します。本アプリには入力値・記録名・比較記録をサーバーのファイルやデータベースへ永続保存する処理、入力内容を独自ログへ記録する処理はありません。必要な記録は終了前にCSVで手元へ保存してください。再読み込み、接続終了、再起動等で記録は失われ得ます。

ホスト側のアクセス記録・Cloudログ・利用統計は、本アプリのセッション保存とは別です。Cloudでの挙動と公開後の運用は[公開手順](docs/deployment.md)を参照してください。記録名に氏名、学籍番号、動物・飼い主の識別情報を入力しないでください。

## 未実装・対象外

- DICOMや実測画像の読み込み、画像アップロード、アプリからのPNG/NPY出力。
- 画像ROIの平均・SD・SNR/CNR測定、ステップウェッジ解析、Slanted-edge MTF測定。
- 物理的なぼけ・動き、解剖画像、kVpによる実組織コントラストの再現。
- CSV再読み込み、保存条件の入力欄への復元、患者線量・臨床適否の自動判定。

## 配布構成と依存関係

`app.py`をリポジトリ直下に置き、`simulator/`、`assets/`、`docs/`、`.streamlit/config.toml`を同じ構成で保持します。ローカル環境専用の起動スクリプトや仮想環境は含めません。

実行用[requirements.txt](requirements.txt)はStreamlit 1.67.0、NumPy 2.5.4、Pillow 13.0.0、pandas 2.3.3、Altair 6.3.0の5パッケージを固定しています。試験用のpytest 9.2.0は[requirements-dev.txt](requirements-dev.txt)へ分離しています。これらは直接依存の固定であり、間接依存を含む全環境の固定ではありません。

試験を実行する場合は、仮想環境のPythonで次を実行します。

```sh
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m pip check
```

実施した試験と未確認事項は[試験記録](docs/test_results.md)を参照してください。ローカル試験はCloud上での起動や授業人数分の同時利用を保証しません。

## 資料

- [計算式・物理モデルと一次資料](docs/physics_model.md)
- [今後の開発順序](docs/roadmap.md)
- [承認後のGitHub登録・Cloud公開手順](docs/deployment.md)

Streamlit Community Cloudでは、リポジトリ、ブランチ、エントリーポイント`app.py`を指定し、Advanced settingsでPython 3.13を選びます。本アプリにAPIキーやSecretsは不要です。手順の根拠は[Streamlit公式デプロイ案内](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)です。
