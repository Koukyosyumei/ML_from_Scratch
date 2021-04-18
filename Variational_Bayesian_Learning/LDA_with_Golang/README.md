# GoLda

LDA (トピックモデル) の変分ベイズバージョンをgolangで実装  
多次元配列は、"gorgonia.org/tensor" を使った  

フォルダ構成は以下の通り

        ---- app
         |    |
         |    |-- main.py
         |
         |
         |-- internal
         | 
         |-- pkg
              |
              |- golda.py      LDAのモデル
              |- gonlp.py　　　自然言語処理全般の関数
              |- stopwords.txt

実行  
go run main.go  

コマンドライン引数一覧  

    -k:  トピック数  
    -w:  各ドキュメントの何文字目まで使うか  
    -m:  MAXITR (訓練回数)  
    -p:  データのパス  

