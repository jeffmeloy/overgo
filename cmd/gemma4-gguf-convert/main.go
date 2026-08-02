package main

import (
	"flag"
	"fmt"
	"os"

	"llamacpp2go/internal/gemma4convert"
)

func main() {
	model := flag.String("model", "", "language-model GGUF output")
	mmproj := flag.String("mmproj", "", "multimodal-projector GGUF output")
	name := flag.String("name", "", "GGUF model name")
	flag.Parse()
	if flag.NArg() != 1 || (*model == "" && *mmproj == "") {
		fmt.Fprintln(os.Stderr, "usage: gemma4-gguf-convert [-model model.gguf] [-mmproj mmproj.gguf] [-name name] checkpoint-directory")
		os.Exit(2)
	}
	report, err := gemma4convert.Convert(gemma4convert.Options{
		Directory: flag.Arg(0), ModelPath: *model, MMProjPath: *mmproj, Name: *name,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %d model tensors and %d projector tensors\n", report.ModelTensors, report.MMProjTensors)
}
