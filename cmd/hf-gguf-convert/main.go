package main

import (
	"context"
	"flag"
	"fmt"
	"os"

	"overgo/internal/artifact"
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
		fmt.Fprintln(os.Stderr, "usage: hf-gguf-convert -source checkpoint-directory [-output model.gguf] [-mmproj mmproj.gguf] [-name name] [-record repodb]")
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
	sequence, generation, sources, err := modelartifact.ReadModelConfigComponents(source)
	if err != nil {
		return err
	}
	if sequence == nil && generation == nil {
		fmt.Println("model config: source directory declares no extractable components; nothing committed")
		return nil
	}
	converted, err := os.Open(output)
	if err != nil {
		return err
	}
	defer converted.Close()
	model, _, err := artifact.Identify(artifact.KindModel, converted)
	if err != nil {
		return err
	}
	document, err := modelartifact.NewModelConfigDocument(model, sequence, generation, sources)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	batch, err := document.Batch("model-config/" + document.ID.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(context.Background(), batch); err != nil {
		return err
	}
	fmt.Printf("model config committed: %s model=%s sources=%d\n", document.ID, model, len(document.Sources))
	if document.Sequence != nil {
		fmt.Printf("sequence extension: k=%d range=[%d,%d) specials=%d auto_tags=%t\n",
			document.Sequence.K, document.Sequence.StartID,
			document.Sequence.StartID+document.Sequence.Vocabulary,
			len(document.Sequence.SpecialTokens), document.Sequence.AutoTags)
	}
	if document.Generation != nil {
		fmt.Printf("generation: bos=%v eos=%v context=%d\n",
			document.Generation.BOSTokens, document.Generation.EOSTokens, document.Generation.ContextLength)
	}
	fmt.Println("honesty: components derive from digested source files; absent declarations stay absent")
	return nil
}
