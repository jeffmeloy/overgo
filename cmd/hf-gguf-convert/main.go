package main

import (
	"flag"
	"fmt"
	"os"

	"overgo/internal/hfconvert"
)

func main() {
	source := flag.String("source", "", "HF checkpoint directory")
	output := flag.String("output", "", "GGUF output path")
	name := flag.String("name", "", "GGUF model name; defaults to the source basename")
	flag.Parse()
	if *source == "" || *output == "" || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: hf-gguf-convert -source checkpoint-directory -output model.gguf [-name name]")
		os.Exit(2)
	}
	report, err := hfconvert.Convert(hfconvert.Options{
		Directory: *source, OutputPath: *output, Name: *name,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d tensors, vocab %d, %d bytes\n", report.Tensors, report.VocabSize, report.OutputBytes)
}
