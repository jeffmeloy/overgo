// dataset-catalog publishes the legacy dataset inventory into OvergoDB.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/dataset"
	"overgo/internal/overgodb"
	"overgo/internal/strictjson"
)

func main() {
	clioptions.MainNamed("dataset-catalog", func() error { return run(os.Args[1:], os.Stdout) })
}

func run(args []string, output io.Writer) error {
	flags := flag.NewFlagSet("dataset-catalog", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repositoryPath := flags.String("repo", "", "OvergoDB root")
	legacyPath := flags.String("legacy", "", "legacy OvergoDB root")
	contentPath := flags.String("root", "", "dataset content root")
	register := flags.String("register", "", "register one directory dataset under this name (with -root)")
	modality := flags.String("modality", "", "explicit primary modality for -register; empty derives it from the corpus")
	speechSpec := flags.String("speech-spec", "", "materialize row-addressed speech references from an existing inventory version and this JSON specification")
	memoryBytes := flags.Uint64("memory", 0, "required source workspace and per-shard reference-metadata byte bound with -speech-spec")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("usage: dataset-catalog [-repo <path>] -legacy <path> -root <path> | -register <name> -root <dir>")
	}
	repositoryRoot := strings.TrimSpace(*repositoryPath)
	if repositoryRoot == "" {
		roots, err := dataroot.ResolveCurrent()
		if err != nil {
			return err
		}
		repositoryRoot = roots.Store
	}
	contentRoot := filepath.Clean(strings.TrimSpace(*contentPath))
	if strings.TrimSpace(*speechSpec) != "" {
		if contentRoot == "." || *memoryBytes == 0 || strings.TrimSpace(*register) != "" || strings.TrimSpace(*legacyPath) != "" || strings.TrimSpace(*modality) != "" {
			return errors.New("dataset-catalog: -speech-spec requires -root and -memory, without -register, -legacy or -modality")
		}
		data, err := artifact.ReadContentFile(*speechSpec)
		if err != nil {
			return err
		}
		var spec dataset.SpeechMaterializationSpec
		if err := strictjson.DecodeBytes(data, &spec); err != nil {
			return err
		}
		store, err := overgodb.Open(repositoryRoot)
		if err != nil {
			return err
		}
		defer store.Close()
		result, err := dataset.MaterializeSpeechDataset(context.Background(), store, contentRoot, spec, *memoryBytes)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "speech dataset=%s profile=%s shards=%d physical_rows=%d reference_records=%d rejected=%d; source hashes verified; audio decode, model inference and training did not run\n", result.Dataset, result.Profile, result.Shards, result.Rows, result.Records, result.Rejected)
		return err
	}
	if *memoryBytes != 0 {
		return errors.New("dataset-catalog: -memory requires -speech-spec")
	}
	if name := strings.TrimSpace(*register); name != "" {
		if contentRoot == "." {
			return errors.New("dataset-catalog: -register requires -root")
		}
		store, err := overgodb.Open(repositoryRoot)
		if err != nil {
			return err
		}
		defer store.Close()
		registered, err := dataset.RegisterDirectoryDatasetAs(context.Background(), store, name, contentRoot, *modality)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "registered %s files=%d bytes=%d changed=%t dataset=%s commit=%s\n",
			name, registered.Files, registered.Bytes, registered.Changed, registered.Dataset, registered.Commit)
		return err
	}
	legacyRoot := filepath.Clean(strings.TrimSpace(*legacyPath))
	if legacyRoot == "." || contentRoot == "." {
		return errors.New("dataset-catalog: legacy and content roots are required")
	}
	store, err := overgodb.Open(repositoryRoot)
	if err != nil {
		return err
	}
	defer store.Close()
	publication, err := dataset.PublishLegacyCatalog(context.Background(), store, legacyRoot, contentRoot)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(output, "datasets registered=%d published=%d available=%d changed=%t commit=%s\n",
		publication.Coverage.Registered, publication.Coverage.Published, publication.Coverage.Available,
		publication.Changed, publication.Commit)
	return err
}
