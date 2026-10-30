package main

import (
	"context"
	"flag"
	"fmt"
	"io"

	"overgo/internal/artifact"
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
	view := flags.Bool("view", false, "write nothing: identify the model the conversion would write, its bytes read from the checkpoint's own files (Qwen 3.5); with -record, bind the declaration to that identity")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *view {
		if *source == "" || *output != "" || *projector != "" || flags.NArg() != 0 {
			return fmt.Errorf("usage: %s -view -source checkpoint-directory [-name name] [-record overgodb]", flags.Name())
		}
		return viewCommand(*source, *name, *record)
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
		model, err := modelartifact.IdentifyConvertedModel(*output)
		if err != nil {
			return err
		}
		if err := recordModelConfig(*source, model, *record); err != nil {
			return fmt.Errorf("%s: %w", flags.Name(), err)
		}
	}
	return nil
}

// viewCommand identifies, without writing it, the model a conversion of
// source would write: the image hfconvert.View serves from the checkpoint's
// own files, identified as a written conversion is. With a store it records
// the checkpoint's declarations bound to that identity, as -record does for a
// written one.
func viewCommand(source, name, recordStore string) error {
	file, err := hfconvert.View(source, name)
	if err != nil {
		return err
	}
	defer file.Close()
	shard := file.Shards()[0]
	model, _, err := artifact.Identify(artifact.KindModel, io.NewSectionReader(shard.Source, 0, int64(shard.Size)))
	if err != nil {
		return err
	}
	fmt.Printf("viewed %d model tensors, %d metadata, %d bytes, nothing written: %s\n", len(file.Tensors), len(file.Metadata), shard.Size, model)
	if recordStore == "" {
		return nil
	}
	return recordModelConfig(source, model, recordStore)
}

// recordModelConfig commits the checkpoint directory's inference- and
// training-relevant declarations as a typed store artifact bound to the
// model identity. A directory declaring nothing commits nothing; that fact is
// reported, not padded.
func recordModelConfig(source string, model artifact.ID, recordStore string) error {
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
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
