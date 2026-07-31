package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"llamacpp2go/internal/gguf"
	"llamacpp2go/internal/model"
)

func run(arguments []string) error {
	if len(arguments) != 1 {
		return errors.New("usage: model-info <model.gguf>")
	}
	file, err := gguf.Open(arguments[0])
	if err != nil {
		return err
	}
	defer file.Close()
	spec, err := model.ReadSpec(file)
	if err != nil {
		return err
	}
	if _, err := model.ReadWeights(file, spec); err != nil {
		return err
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	return encoder.Encode(spec)
}

func main() {
	if err := run(os.Args[1:]); err != nil {
		fmt.Fprintln(os.Stderr, "model-info:", err)
		os.Exit(1)
	}
}
