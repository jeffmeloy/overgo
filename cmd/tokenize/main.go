package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/tokenizer"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
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
	return json.NewEncoder(os.Stdout).Encode(result)
}
