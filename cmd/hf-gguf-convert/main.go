package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"overgo/internal/hfconvert"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

func main() {
	source := flag.String("source", "", "HF checkpoint directory")
	output := flag.String("output", "", "GGUF output path")
	projector := flag.String("mmproj", "", "multimodal-projector GGUF output")
	name := flag.String("name", "", "GGUF model name; defaults to the source basename")
	record := flag.String("record", "", "OvergoDB root: commit the extracted model-config declaration bound to the converted artifact")
	flag.Parse()
	if *source == "" || (*output == "" && *projector == "") || flag.NArg() != 0 {
		fmt.Fprintln(os.Stderr, "usage: hf-gguf-convert -source checkpoint-directory [-output model.gguf] [-mmproj mmproj.gguf] [-name name] [-record overgodb]")
		os.Exit(2)
	}
	if *record != "" && *output == "" {
		fmt.Fprintln(os.Stderr, "hf-gguf-convert: -record requires -output (the declaration binds to the converted model artifact)")
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
	if *record != "" {
		if err := recordModelConfig(*source, *output, *record); err != nil {
			fmt.Fprintln(os.Stderr, "hf-gguf-convert:", err)
			os.Exit(1)
		}
	}
}

// recordModelConfig commits the checkpoint directory's inference- and
// training-relevant declarations as a typed store artifact bound to the
// converted model identity. A directory declaring nothing commits nothing;
// that fact is reported, not padded.
func recordModelConfig(source, output, recordStore string) error {
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	model, err := modelartifact.IdentifyConvertedModel(output)
	if err != nil {
		return err
	}
	id, recorded, err := modelartifact.RecordModelConfig(context.Background(), store, source, model)
	if err != nil {
		return err
	}
	if !recorded {
		fmt.Println("model config: source directory declares no extractable components; nothing committed")
		return nil
	}
	fmt.Printf("model config committed: %s\n", id)
	fmt.Println("audit: components derive from digested source files; absent declarations stay absent")
	return nil
}
