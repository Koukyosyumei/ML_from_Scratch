package main

import (
	"flag"
	golda "github.com/workspace/GoLda/pkg"
)

// -----------------------------------------------------

func main() {
	fk := flag.Int("k", 10, "num of topic")
	fmaxitr := flag.Int("m", 20, "max itr")
	fw := flag.Int("w", -1, "num of word in each document to train")
	fp := flag.String("p", "input/lda.csv", "path to inputfile (csv)")
	flag.Parse()

	// ---------- ハイパーパラメータの設定

	K := *fk
	MAXITR := *fmaxitr
	WORD := *fw
	INPUTPATH := *fp

	golda.LDAvb(K, MAXITR, WORD, INPUTPATH)
}
