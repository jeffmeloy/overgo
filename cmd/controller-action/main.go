// Command controller-action compiles one allowlisted controller action into
// its content-addressed artifacts and commits them. This is the single
// execution path for controller emissions: actions outside the allowlist are
// unrepresentable, and the executor is deterministic Go -- no code, no shell,
// no runtime orchestration.
//
//	go run ./cmd/controller-action -action <action.json> -record <overgodb>
//	go run ./cmd/controller-action -action <action.json> -record <overgodb> -source <model-id>=<directory> -output <directory>
//	go run ./cmd/controller-action -materialize-decision <decision-id> -record <overgodb> -output <directory>
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/controlleraction"
	"overgo/internal/modelmerge"
	"overgo/internal/overgodb"
)

var materializeCandidateTrial = controlleraction.MaterializeCandidateTrial

func main() {
	clioptions.MainNamed("controller-action", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("controller-action", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	actionPath := flags.String("action", "", "path to the typed action document (JSON)")
	materializeDecision := flags.String("materialize-decision", "", "persisted realize-decision evidence identity")
	recordStore := flags.String("record", "", "OvergoDB root: commit the compiled artifacts")
	outputPath := flags.String("output", "", "offline artifact output or candidate materialization root")
	var sources streamingSourceFlags
	flags.Var(&sources, "source", "offline source as MODEL_ID=SAFETENSORS_DIRECTORY; repeat for every input")
	if err := flags.Parse(args); err != nil {
		return err
	}
	materializing := *materializeDecision != ""
	invalidMaterialization := materializing && (*actionPath != "" || *outputPath == "" || len(sources) != 0)
	invalidAction := !materializing && (*actionPath == "" || (*outputPath == "") != (len(sources) == 0))
	if flags.NArg() != 0 || *recordStore == "" || invalidMaterialization || invalidAction {
		return errors.New("usage: controller-action -action <action.json> -record <overgodb> [-source MODEL_ID=DIRECTORY ... -output DIRECTORY] | -materialize-decision <decision-id> -record <overgodb> -output <directory>")
	}
	var action controlleraction.Action
	var decision artifact.ID
	var err error
	if materializing {
		decision, err = artifact.ParseID(*materializeDecision)
		if err != nil || decision.Kind() != artifact.KindEvidence {
			return errors.Join(errors.New("materialize-decision must be an evidence identity"), err)
		}
	} else {
		data, readErr := os.ReadFile(*actionPath)
		if readErr != nil {
			return readErr
		}
		action, err = controlleraction.ParseAction(data)
		if err != nil {
			return err
		}
	}
	store, err := overgodb.Open(*recordStore)
	if err != nil {
		return err
	}
	defer func() { _ = store.Close() }()
	ctx := context.Background()
	if materializing {
		result, err := materializeCandidateTrial(ctx, store, decision, *outputPath)
		if err != nil {
			return err
		}
		realizations := make([]artifact.ID, 0, len(result.Directories))
		for realization := range result.Directories {
			realizations = append(realizations, realization)
		}
		slices.SortFunc(realizations, artifact.CompareID)
		if result.Recovered {
			fmt.Fprintf(output, "candidate materialization: %s recovered arms=%d\n",
				result.Materialization.ID, len(realizations))
		} else {
			fmt.Fprintf(output, "candidate materialization: %s commit=%s arms=%d\n",
				result.Materialization.ID, result.Commit, len(realizations))
		}
		for _, realization := range realizations {
			fmt.Fprintf(output, "materialized arm: %s directory=%s\n", realization, result.Directories[realization])
		}
		fmt.Fprintln(output, "audit: persisted realize decision executed through direct Go owners; sources were store-derived; no activation alias moved")
		return nil
	}
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
	fmt.Fprintf(output, "action %s compiled %d artifact(s); audit: deterministic Go executor, allowlist-only, never orchestrates\n",
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
