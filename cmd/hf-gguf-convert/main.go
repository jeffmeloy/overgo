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
	projector := flag.String("mmproj", "", "multimodal-projector GGUF output")
	name := flag.String("name", "", "GGUF model name; defaults to the source basename")
	flag.Parse()
	if *source == "" || (*output == "" && *projector == "") || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: hf-gguf-convert -source checkpoint-directory [-output model.gguf] [-mmproj mmproj.gguf] [-name name]")
		os.Exit(2)
	}
	report, err := hfconvert.Convert(hfconvert.Options{
		Directory: *source, OutputPath: *output, ProjectorPath: *projector, Name: *name,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d model tensors, %d projector tensors, vocab %d, %d bytes\n",
		report.Tensors, report.ProjectorTensors, report.VocabSize, report.OutputBytes)
}
