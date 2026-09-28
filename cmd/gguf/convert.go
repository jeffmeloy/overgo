package main

import (
	"context"
	"flag"
	"fmt"

	"overgo/internal/hfconvert"
	"overgo/internal/modelartifact"
	"overgo/internal/overgodb"
)

// convertCommand is gguf convert: it converts a Hugging Face checkpoint
// directory to GGUF and, with -record, commits the model-config declaration
// that ties the conversion back to the directory.
func convertCommand(flags *flag.FlagSet, args []string) error {
	source := flags.String("source", "", "HF checkpoint directory")
	output := flags.String("output", "", "GGUF output path")
	projector := flags.String("mmproj", "", "multimodal-projector GGUF output")
	name := flags.String("name", "", "GGUF model name; defaults to the source basename")
	record := flags.String("record", "", "OvergoDB root: commit the extracted model-config declaration bound to the converted artifact")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *source == "" || (*output == "" && *projector == "") || flags.NArg() != 0 {
		return fmt.Errorf("usage: %s -source checkpoint-directory [-output model.gguf] [-mmproj mmproj.gguf] [-name name] [-record overgodb]", flags.Name())
	}
	if *record != "" && *output == "" {
		return fmt.Errorf("%s: -record requires -output (the declaration binds to the converted model artifact)", flags.Name())
	}
	report, err := hfconvert.Convert(hfconvert.Options{
		Directory: *source, OutputPath: *output, ProjectorPath: *projector, Name: *name,
	})
	if err != nil {
		return err
	}
	fmt.Printf("wrote %d model tensors, %d projector tensors, vocab %d, %d bytes\n",
		report.Tensors, report.ProjectorTensors, report.VocabSize, report.OutputBytes)
	if *record != "" {
		if err := recordModelConfig(*source, *output, *record); err != nil {
			return fmt.Errorf("%s: %w", flags.Name(), err)
		}
	}
	return nil
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
