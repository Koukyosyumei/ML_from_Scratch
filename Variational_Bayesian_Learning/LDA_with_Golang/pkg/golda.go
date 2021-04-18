package golda

import (
	"fmt"
	"io"
	"math"
	"os"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/stat/distmv"
	"gorgonia.org/tensor"
)

// ------------------- 型の定義 -----------------------

// ss is a single slice, representing this: [start:start+1:0]
type ss int

func (s ss) Start() int { return int(s) }
func (s ss) End() int   { return int(s) + 1 }
func (s ss) Step() int  { return 0 }

// ------------------ 関数の定義------------------------

func dirichletrand(alpha float64) float64 {
	var a []float64
	a = append(a, alpha)
	s := rand.NewSource(20200428)
	r := rand.New(s)
	distribution := distmv.NewDirichlet(a, r)
	return distribution.Rand(nil)[0]
}

// -----------------------------------------------------

// LDAvb is the model of LDA with Variational Bayes
func LDAvb(K int, MAXITR int, WORD int, INPUTPATH string) {
	/*

		K:         the number of topics
		MAXITR:    the maxium number of iteration to trainig
		WORD:      the num of words appearing in all documents
		INPUTPATH: the path to documents

	*/

	// --------- データの読み込み、コーパスその他の処理
	fmt.Println("load the data")

	docs := UserCsvNewReader(INPUTPATH)
	docstokenized := Tokenizer(docs[1:])
	fmt.Println("num of documtents: ", len(docstokenized))

	fmt.Println("pre-process the data")

	vocab := Unique(docstokenized)
	fmt.Println("num of vocablary: ", len(vocab))

	//word2id, id2word := Vocab2Dict(vocab)
	word2id, id2word := Vocab2Dict(vocab)
	courpus := GetCourpus(docstokenized, word2id)

	Writeid2word("output/id2word.txt", id2word)
	SaveCourpus("output/courpus.csv", courpus)

	// パラメータの設定
	// ほんとは、train data の長さでいく
	M := len(courpus)
	V := len(vocab)

	//  ---------------------------------------------- //
	fmt.Println("initialize the parameters")

	// 行列の初期化
	gammainit := make([]float64, M*K)
	lambdainit := make([]float64, V*K)
	qinit := make([]float64, M*V*K)

	alphainit := make([]float64, M*K)
	etainit := make([]float64, V*K)

	for i := range gammainit {
		gammainit[i] = rand.NormFloat64()
	}
	gamma := tensor.New(tensor.WithBacking(gammainit), tensor.WithShape(M, K))
	alpha := tensor.New(tensor.WithBacking(alphainit), tensor.WithShape(M, K))

	for i := range lambdainit {
		lambdainit[i] = rand.NormFloat64()
	}
	lambda := tensor.New(tensor.WithBacking(lambdainit), tensor.WithShape(V, K))
	eta := tensor.New(tensor.WithBacking(etainit), tensor.WithShape(V, K))

	for i := range qinit {
		qinit[i] = rand.NormFloat64()
	}
	q := tensor.New(tensor.WithBacking(qinit), tensor.WithShape(M, V, K))

	// ----------- パラメータの表示 ----------------- //

	fmt.Println("M: ", M)
	fmt.Println("V: ", V)
	fmt.Println("K: ", K)
	fmt.Println("MAXITR: ", MAXITR)
	fmt.Println("WORD: ", WORD)

	fmt.Println("shape of gamma: ", gamma.Shape())
	fmt.Println("shape of lambda: ", lambda.Shape())
	fmt.Println("shape of q: ", q.Shape())

	fmt.Println("shape of alpha: ", alpha.Shape())
	fmt.Println("shape of eta: ", eta.Shape())

	// -------- 内部の重み? の設定 ------------------- //

	var doc []int
	var Ndoc int

	gammasum, _ := tensor.Sum(gamma, 1)
	lambdasumview, _ := tensor.Sum(lambda, 0)
	lambdasum := lambdasumview.(*tensor.Dense)

	var w int

	gammadview, _ := gamma.Slice(ss(0))
	gammad := gammadview.(*tensor.Dense)

	lambdawview, _ := lambda.Slice(ss(0))
	lambdaw := lambdawview.(*tensor.Dense)

	var gammasumd interface{}
	var qsumaxis1d interface{}
	var qsumaxis0d interface{}
	qtotalsum, _ := tensor.Sum(q, 0, 1, 2)
	qsumaxis0, _ := q.Sum(0)
	qsumaxis1, _ := q.Sum(1)

	value20, _ := gammad.SubScalar(0, true)
	var value20kintf interface{}
	var value20k float64

	// --------- outputファイルの初期値設定 -----------
	var alphafile io.Writer
	var etafile io.Writer

	// -------------------------------------------- //

	for itr := 0; itr < MAXITR; itr++ {
		fmt.Println("itr: ", itr)

		for d := 0; d < M; d++ {

			// doc = courpus[d]
			doc = courpus[d][:WORD]
			Ndoc = len(doc)

			for n := 0; n < Ndoc; n++ {

				gammasum, _ = tensor.Sum(gamma, 1)
				lambdasumview, _ = tensor.Sum(lambda, 0)
				lambdasum = lambdasumview.(*tensor.Dense)

				w = int(doc[n])

				gammadview, _ = gamma.Slice(ss(d))
				gammad = gammadview.(*tensor.Dense)
				lambdawview, _ = lambda.Slice(ss(w))
				lambdaw = lambdawview.(*tensor.Dense)

				gammad.Apply(Digamma)
				gammasum.Apply(Digamma)
				lambdaw.Apply(Digamma)
				lambdasum.Apply(Digamma)

				gammasumd, _ = gammasum.At(d)

				value20, _ = gammad.SubScalar(gammasumd, true)
				value20, _ = value20.Add(lambdaw)
				value20, _ = value20.Sub(lambdasum)
				value20.Apply(math.Exp)

				for k := 0; k < K; k++ {
					value20kintf, _ = value20.At(k)
					value20k = value20kintf.(float64)
					q.SetAt(value20k, d, w, k)
				}

				qtotalsum, _ = tensor.Sum(q, 0, 1, 2)
				q, _ = q.DivScalar(qtotalsum, true)

				qsumaxis0, _ = q.Sum(0)
				qsumaxis1, _ = q.Sum(1)

				for k := 0; k < K; k++ {
					qsumaxis1d, _ = qsumaxis1.At(d, k)
					qsumaxis0d, _ = qsumaxis0.At(w, k)
					gamma.SetAt(qsumaxis1d, d, k)
					lambda.SetAt(qsumaxis0d, w, k)
				}

			}
		}
		alpha, _ = gamma.Sub(qsumaxis1)
		eta, _ = lambda.Sub(qsumaxis0)

		alphafile, _ = os.OpenFile(fmt.Sprintf("output/alpha.%d.csv", itr), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)
		etafile, _ = os.OpenFile(fmt.Sprintf("output/eta.%d.csv", itr), os.O_RDWR|os.O_CREATE|os.O_TRUNC, 0644)

		alpha.WriteCSV(alphafile)
		eta.WriteCSV(etafile)
	}
}
