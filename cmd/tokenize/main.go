package main

import (
	"errors"
	"flag"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

func main() {
	clioptions.Main(run)
}

func run() error {
	addSpecial := flag.Bool("add-special", false, "apply the GGUF BOS/EOS policy")
	parseSpecial := flag.Bool("parse-special", false, "recognize control and unknown token text")
	flag.Parse()
	if flag.NArg() != 2 {
		return errors.New("usage: tokenize [options] <model.gguf> <text>")
	}
	file, err := gguf.Open(flag.Arg(0))
	if err != nil {
		return err
	}
	defer file.Close()
	vocab, err := tokenizer.Load(file)
	if err != nil {
		return err
	}
	ids, err := vocab.Encode(flag.Arg(1), tokenizer.EncodeOptions{
		AddSpecial:   *addSpecial,
		ParseSpecial: *parseSpecial,
	})
	if err != nil {
		return err
	}
	result := struct {
		Model  string              `json:"model"`
		Pre    string              `json:"pre"`
		IDs    []tokenizer.TokenID `json:"ids"`
		Pieces []string            `json:"pieces"`
	}{
		Model:  vocab.Model,
		Pre:    vocab.Pre,
		IDs:    ids,
		Pieces: make([]string, len(ids)),
	}
	for i, id := range ids {
		token, _ := vocab.Token(id)
		result.Pieces[i] = token.Text
	}
	return clioptions.WriteJSON(os.Stdout, result)
}
