package main

import (
	"fmt"
	"io"
	"os"

	"llamacpp2go/internal/sampling"
)

func main() {
	input, err := readInput(os.Args[1:])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	grammar, err := sampling.JSONSchemaToGrammar(input)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Println(grammar)
}

func readInput(arguments []string) ([]byte, error) {
	switch len(arguments) {
	case 0:
		return io.ReadAll(os.Stdin)
	case 1:
		return os.ReadFile(arguments[0])
	default:
		return nil, fmt.Errorf("usage: json-schema-grammar [schema.json]")
	}
}
