# GitHub・Streamlit Community Cloud公開手順

確認日：2026年11月8日。対象：公開準備版v0.1.3。

**この資料は承認後に行う手順です。GitHubへのアップロードとCloudへのデプロイは、今回実施していません。** 担当教員が配布内容・公開範囲を確認し、公開の最終承認を記録してから進めます。公開承認は、装置校正や学術的妥当性の検証とは別です。

## 1. GitHubに登録する

1. GitHubアカウントがなければ[登録ページ](https://github.com/signup)で作成し、メール確認を完了します。[GitHub公式の登録説明](https://docs.github.com/en/account-and-profile/how-tos/account-management/creating-an-account-on-github)
2. GitHubで「New repository」を選び、所有者と名前を指定します。例は`vet-xray-simulator`です。承認された公開範囲を選びます。この手順では公開リポジトリを想定します。既存のREADME等を使うため、作成画面でのファイル自動追加は選びません。[リポジトリ作成](https://docs.github.com/en/repositories/creating-and-managing-repositories/creating-a-new-repository)
3. 「Add file」→「Upload files」から、この配布フォルダーの**中身**を追加します。ZIPそのものや外側のフォルダーを一段追加せず、`app.py`が直下になるようにします。`.streamlit/config.toml`と`.gitignore`などの隠しファイルも確認します。[ファイル追加](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
4. ブラウザーから追加する場合は、配布ZIPを新しいフォルダーに展開した内容を選び、ローカル検証で生成したファイルを混ぜないでください。公開対象は本配布物に限定します。`.venv/`、キャッシュ、個人のCSV、秘密情報、開発用の旧版保存物は含めません。内容を確認して変更を確定し、リポジトリとブランチ名を控えます。

## 2. Community Cloudで公開する

1. [Streamlit Community Cloud](https://share.streamlit.io/)でアカウントを作成します。GitHubによるログインを利用できます。[Cloudアカウント作成](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/create-your-account)
2. Workspaceの「Connect GitHub account」から対象リポジトリへのアクセスを接続します。GitHubでログインしただけでは、この接続は完了しません。公開担当者にはリポジトリの管理権限が必要です。[GitHub接続](https://docs.streamlit.io/deploy/streamlit-community-cloud/get-started/connect-your-github-account)
3. 「Create app」→「Yup, I have an app」を選び、次の項目を設定します。

| 項目 | 設定 |
| --- | --- |
| Repository | 承認後に作成したリポジトリ |
| Branch | 公開するブランチ。例：`main` |
| Main file path | `app.py` |
| Advanced settings → Python version | **3.13** |
| Secrets | 空欄。本アプリは不要 |

Pythonの既定値に任せず3.13を選びます。ローカル確認は3.13.0ですが、Cloudのパッチ版まで同一になるとは限りません。設定を保存し、「Deploy」を実行します。[公式デプロイ手順](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)

実行用依存関係は直下の`requirements.txt`から導入します。`requirements-dev.txt`は開発・試験用です。本配布物ではローカル専用の`server.address`や`server.port`を設定しません。Cloudに合わせて8501/8503等を手入力する必要はありません。[依存関係](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)、[Cloud設定](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)

## 3. 公開後に確認する

Cloud上での起動確認は未実施です。公開後、依存関係の導入ログを確認し、4タブ、パターン切り替え、標準・強調、固定基準・現在、記録追加・削除、CSV保存を確認します。別のブラウザー接続でも、記録が独立していることを確認します。

複数学生の同時利用性能は未検証です。授業前に実際の端末・ネットワーク・想定人数で応答とメモリー使用を確認し、遅延・停止が生じる場合は運用人数や配置を見直します。ローカル試験結果だけで授業全体の収容能力を保証しません。

GitHubの公開ブランチを更新するとアプリにも反映され、依存関係の変更では再導入が発生します。授業中の更新を避け、必要な比較CSVは更新・再起動前に保存します。Cloudログは「Manage app」から確認できます。[アプリ管理](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)

## 4. セッションとホストの制限

本アプリの比較記録は接続ごとのサーバー側セッションメモリーに置き、入力内容をファイル・データベース・独自ログへ保存しません。利用者がダウンロードしたCSVは手元に残ります。再読み込み、切断、再起動、休眠等をまたぐ記録の保持は保証しません。

この説明はCloud運営側の記録がないことを意味しません。Cloudログと閲覧統計はホスト側の機能です。公式仕様では`browser.gatherUsageStats = true`がアプリ設定を上書きし、アプリは米国でホストされます。本アプリの設定だけでCloudのログ・分析・ホスト所在地を制御できるとは扱いません。[Cloudの制約](https://docs.streamlit.io/deploy/streamlit-community-cloud/status)、[閲覧統計](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app/app-analytics)

公式管理資料が示す資源量は**2024年2月時点の概算**で、CPU 0.078–2コア、メモリー690 MB–2.7 GB、ストレージ最大50 GBです。無予告で変更され得る数値であり、現在の確約容量や同時利用人数の根拠にはしません。公開時にはWorkspaceのSettings → Limitsも確認します。[資源制限](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)、[Workspace設定](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-account/workspace-settings)

公式資料では12時間アクセスのないアプリは休眠します。休眠画面の「Yes, get this app back up!」で起動できるため、授業開始前にアクセスして起動状態を確認します。[休眠の説明](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
