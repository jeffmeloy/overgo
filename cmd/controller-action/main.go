// Command controller-action compiles one allowlisted controller action into
// its content-addressed artifacts and commits them. This is the single
// execution path for controller emissions: actions outside the allowlist are
// unrepresentable, and the executor is deterministic Go -- no code, no shell,
// no runtime orchestration.
//
//	go run ./cmd/controller-action -action <action.json> -record <overgodb>
//	go run ./cmd/controller-action -action <action.json> -record <overgodb> -source <model-id>=<directory> -output <directory>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/controlleraction"
	"overgo/internal/modelmerge"
	"overgo/internal/overgodb"
)

func main() {
	clioptions.MainNamed("controller-action", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("controller-action", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	actionPath := flags.String("action", "", "path to the typed action document (JSON)")
	recordStore := flags.String("record", "", "OvergoDB root: commit the compiled artifacts")
	outputPath := flags.String("output", "", "offline artifact output directory")
	var sources streamingSourceFlags
	flags.Var(&sources, "source", "offline source as MODEL_ID=SAFETENSORS_DIRECTORY; repeat for every input")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 || *actionPath == "" || *recordStore == "" || (*outputPath == "") != (len(sources) == 0) {
		return errors.New("usage: controller-action -action <action.json> -record <overgodb> [-source MODEL_ID=DIRECTORY ... -output DIRECTORY]")
	}
	data, err := os.ReadFile(*actionPath)
	if err != nil {
		return err
	}
	action, err := controlleraction.ParseAction(data)
	if err != nil {
		return err
	}
	store, err := overgodb.Open(*recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	batch, err := controlleraction.CompileTransaction(ctx, store, action)
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return err
	}
	for _, content := range batch.Contents {
		fmt.Fprintf(output, "compiled artifact: %s\n", content.Descriptor.ID)
	}
	if *outputPath != "" {
		result, err := controlleraction.ExecuteOfflineArtifact(ctx, store, action, sources, *outputPath)
		if err != nil {
			return err
		}
		fmt.Fprintf(output, "offline output: %s tensors=%d peak_resident_bytes=%d\n",
			result.Destination, result.TensorCount, result.PeakResidentBytes)
	}
	fmt.Fprintf(output, "action %s compiled %d artifact(s); honesty: deterministic Go executor, allowlist-only, never orchestrates\n",
		action.Kind, len(batch.Contents))
	return nil
}

type streamingSourceFlags []modelmerge.StreamingSource

// String renders repeatable source flags for the flag package.
func (values *streamingSourceFlags) String() string {
	parts := make([]string, len(*values))
	for index, value := range *values {
		parts[index] = value.Model.String() + "=" + value.Directory
	}
	return strings.Join(parts, ",")
}

// Set parses and appends one exact model-to-directory binding.
func (values *streamingSourceFlags) Set(value string) error {
	modelText, directory, found := strings.Cut(value, "=")
	if !found || strings.TrimSpace(directory) == "" {
		return errors.New("source must be MODEL_ID=SAFETENSORS_DIRECTORY")
	}
	model, err := artifact.ParseID(modelText)
	if err != nil || model.Kind() != artifact.KindModel {
		return errors.Join(errors.New("source model identity is invalid"), err)
	}
	*values = append(*values, modelmerge.StreamingSource{Model: model, Directory: directory})
	return nil
}
