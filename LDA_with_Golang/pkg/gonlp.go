package golda

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// ReadLine is the function which read txt file
func ReadLine(filename string) []string {
	file, err := os.Open(filename)
	var lines []string

	if err != nil {
		fmt.Println("error")
		fmt.Println(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		lines = append(lines, line)
	}
	if err := scanner.Err(); err != nil {
		fmt.Println("error")
		fmt.Println(err)
	}
	return lines
}

// Writeword2id writes down the wird2id to txt file
func Writeword2id(filename string, lines map[string]int) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for key, line := range lines {
		_, err := file.WriteString(key + " , " + string(line))
		// fmt.Fprint()の場合
		// _, err := fmt.Fprint(file, line)
		if err != nil {
			return err
		}
	}
	return nil
}

// Writeid2word writes down the id2word to txt file
func Writeid2word(filename string, lines map[int]string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	for key, line := range lines {
		_, err := file.WriteString(string(key) + " , " + line)
		// fmt.Fprint()の場合
		// _, err := fmt.Fprint(file, line)
		if err != nil {
			return err
		}
	}
	return nil
}

// StringInSlice is the function corresponed to the function "if not" in python
func StringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

// Tokenizer returns tokenized tokenized sentence
func Tokenizer(docs []string) [][]string {
	//var docstokenized [][]string
	docstokenized := make([][]string, len(docs))
	for i, doc := range docs {

		docreplaced := strings.Replace(doc, ".", "", -1)
		docreplaced = strings.Replace(docreplaced, ":", "", -1)
		docreplaced = strings.Replace(docreplaced, "-", "", -1)
		docreplaced = strings.Replace(docreplaced, "  ", " ", -1)

		words := strings.Split(docreplaced, " ")

		stopwords := ReadLine("stopwords.txt")
		stopwords = append(stopwords, " ")
		var wordsnonstop []string

		for _, w := range words {
			if StringInSlice(w, stopwords) == false {
				wordsnonstop = append(wordsnonstop, w)
			}
		}

		docstokenized[i] = wordsnonstop
	}
	return docstokenized
}

// Digamma returns the logorithmic derivative of the gamma function at x.
//  ψ(x) = d/dx (Ln (Γ(x)).
// Note that if x is a negative integer in [-7, 0] this function will return
// negative Inf.
func Digamma(x float64) float64 {
	// This is adapted from
	// http://web.science.mq.edu.au/~mjohnson/code/digamma.c
	var result float64
	for ; x < 7.0; x++ {
		result -= 1 / x
	}
	x -= 1.0 / 2.0
	xx := 1.0 / x
	xx2 := xx * xx
	xx4 := xx2 * xx2
	result += math.Log(x) + (1./24.)*xx2 - (7.0/960.0)*xx4 + (31.0/8064.0)*xx4*xx2 - (127.0/30720.0)*xx4*xx4
	return result
}

// UserCsvNewReader retun list of string, input is filename
//  ψ(x) = d/dx (Ln (Γ(x)).
// Note that if x is a negative integer in [-7, 0] this function will return
// negative Inf.
func UserCsvNewReader(fileName string) []string {
	fp, err := os.Open(fileName)
	if err != nil {
		panic(err)
	}
	defer fp.Close()

	var results []string

	reader := csv.NewReader(fp)
	//reader.Comma = ','
	reader.LazyQuotes = true
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic(err)
		}
		results = append(results, record[1])
	}
	return results
}

// Unique is the function which allows us to creat the array (set type in Python)
func Unique(ss [][]string) []string {

	m := make(map[string]struct{}) // 空のstructを使う

	for _, s := range ss {
		for _, w := range s {
			m[w] = struct{}{}
		}
	}

	uniq := []string{}
	for i := range m {
		uniq = append(uniq, i)
	}
	return uniq
}

// Vocab2Dict returns word2id and id2word
func Vocab2Dict(vocab []string) (map[string]int, map[int]string) {
	word2id := make(map[string]int)
	id2word := make(map[int]string)

	for k, v := range vocab {
		word2id[v] = k
		id2word[k] = v
	}

	return word2id, id2word
}

// GetCourpus returns courpus based on the docs and word2id
func GetCourpus(docs [][]string, word2id map[string]int) [][]int {
	var courpus [][]int
	for _, d := range docs {

		var words []int

		for _, w := range d {
			words = append(words, word2id[w])
		}
		courpus = append(courpus, words)
	}
	return courpus
}

// SaveCourpus saves courpus as csv
func SaveCourpus(filename string, courpus [][]int) {
	file, _ := os.OpenFile(filename, os.O_WRONLY|os.O_CREATE, 0600)
	//failOnError(err)
	defer file.Close()

	file.Truncate(0) // ファイルを空っぽにする(実行2回目以降用)
	//failOnError(err)

	writer := csv.NewWriter(file)

	for _, d := range courpus {
		writer.Write(strings.Split(strings.TrimRight(fmt.Sprint(d)[1:], "]"), " "))
	}
	writer.Flush()
}
