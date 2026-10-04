// dataset-catalog publishes the legacy dataset inventory into OvergoDB,
// registers a directory dataset, or publishes a weighted mixture of
// registered datasets.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
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
	mixture := flags.String("mixture", "", "publish a weighted mixture of registered datasets under this name (with -members)")
	members := flags.String("members", "", "the mixture's registered datasets and their weights: <name>=<weight>,...")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 0 {
		return errors.New("usage: dataset-catalog [-repo <path>] -legacy <path> -root <path> | -register <name> -root <dir> | -mixture <name> -members <name>=<weight>,...")
	}
	repositoryRoot, err := dataroot.StoreRoot(*repositoryPath)
	if err != nil {
		return err
	}
	if name := strings.TrimSpace(*mixture); name != "" || *members != "" {
		if name == "" || *members == "" || *speechSpec != "" || *register != "" || *legacyPath != "" || *contentPath != "" || *modality != "" || *memoryBytes != 0 {
			return errors.New("dataset-catalog: -mixture requires -members, without -register, -legacy, -root, -modality, -speech-spec or -memory")
		}
		store, err := overgodb.Open(repositoryRoot)
		if err != nil {
			return err
		}
		defer store.Close()
		published, err := publishMixture(context.Background(), store, name, *members)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(output, "mixture %s dataset=%s members=%d\n", name, published.ID, len(published.Members))
		return err
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

// publishMixture publishes a weighted mixture of registered datasets, named
// by weights as <name>=<weight>,..., and registers it under name, so an
// objective trains on the mix as on any registered dataset.
func publishMixture(ctx context.Context, store artifact.Repository, name, weights string) (dataset.Document, error) {
	var members []dataset.Member
	for field := range strings.SplitSeq(weights, ",") {
		member, text, found := strings.Cut(strings.TrimSpace(field), "=")
		weight, err := strconv.Atoi(text)
		if !found || err != nil || weight <= 0 {
			return dataset.Document{}, fmt.Errorf("dataset-catalog: -members %q: want <name>=<weight>,... with positive weights", weights)
		}
		id, registered, err := artifact.ResolveAlias(ctx, store, dataset.RegisteredAliasPrefix+member)
		if err != nil {
			return dataset.Document{}, err
		}
		if !registered {
			return dataset.Document{}, fmt.Errorf("dataset-catalog: mixture member %q is not a registered dataset", member)
		}
		members = append(members, dataset.Member{Dataset: id, Weight: uint64(weight)})
	}
	mixture, err := dataset.NewMixture(members)
	if err != nil {
		return dataset.Document{}, err
	}
	// A mixture already registered under name is published; a rerun
	// changes nothing.
	alias, moved, err := artifact.MoveAlias(ctx, store, dataset.RegisteredAliasPrefix+name, mixture.ID)
	if err != nil || !moved {
		return mixture, err
	}
	batch, err := dataset.PublicationBatch("dataset/mixture/"+name+"/"+mixture.ID.String(), []dataset.Document{mixture}, []artifact.AliasBinding{alias})
	if err != nil {
		return dataset.Document{}, err
	}
	_, err = artifact.CommitBatch(ctx, store, batch)
	return mixture, err
}
