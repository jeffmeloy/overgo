// Command audio-oracle verifies and publishes one immutable external audio-oracle election.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"overgo/internal/artifact"
	"overgo/internal/audioparity"
	"overgo/internal/clioptions"
	"overgo/internal/overgodb"
)

func main() {
	clioptions.MainNamed("audio-oracle", run)
}

func run() error {
	flags := flag.NewFlagSet("audio-oracle", flag.ContinueOnError)
	flags.SetOutput(os.Stderr)
	repository := flags.String("repo", "", "OvergoDB root")
	recordPath := flags.String("election", "", "canonical election JSON")
	modelRoot := flags.String("model-root", "", "local model Git worktree")
	datasetRoot := flags.String("dataset-root", "", "local dataset Git worktree")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if flags.NArg() != 0 || *repository == "" || *recordPath == "" || *modelRoot == "" || *datasetRoot == "" {
		return errors.New("usage: audio-oracle -repo <store> -election <json> -model-root <directory> -dataset-root <directory>")
	}
	encoded, err := os.ReadFile(*recordPath)
	if err != nil {
		return err
	}
	if len(encoded) == 0 || len(encoded) > artifact.MaxContentBytes {
		return errors.New("audio-oracle: election document size is invalid")
	}
	election, err := audioparity.NormalizeElection(encoded)
	if err != nil {
		return err
	}
	ctx := context.Background()
	if err := election.VerifyInstalled(ctx, *modelRoot, *datasetRoot); err != nil {
		return err
	}
	store, err := overgodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	current, found, err := store.ResolveAlias(ctx, audioparity.FirstOfflineASRAlias)
	if err != nil {
		return err
	}
	if found && current == election.ID {
		return writeSummary(election, true)
	}
	batch, err := election.Batch("audio/oracle/election/" + election.ID.DigestHex())
	if err != nil {
		return err
	}
	if found {
		batch.Aliases[0].Previous = &current
	}
	locations, err := electionLocations(election, *modelRoot, *datasetRoot)
	if err != nil {
		return err
	}
	batch.Locations = locations
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return err
	}
	return writeSummary(election, false)
}

func electionLocations(election audioparity.Election, modelRoot, datasetRoot string) ([]artifact.LocationEvent, error) {
	locations := make([]artifact.LocationEvent, 0, len(election.Files)+2)
	appendLocation := func(id artifact.ID, kind artifact.LocationKind, value string) error {
		location, err := artifact.CanonicalLocalLocation(id, kind, value)
		if err != nil {
			return err
		}
		locations = append(locations, artifact.LocationEvent{Location: location, Action: artifact.LocationAdd})
		return nil
	}
	if err := appendLocation(election.Model.ID, artifact.LocationDirectory, modelRoot); err != nil {
		return nil, err
	}
	for _, file := range election.Files {
		if err := appendLocation(file.Artifact, artifact.LocationFile, filepath.Join(modelRoot, filepath.FromSlash(file.Path))); err != nil {
			return nil, err
		}
	}
	if err := appendLocation(election.Dataset.Artifact, artifact.LocationFile, filepath.Join(datasetRoot, filepath.FromSlash(election.Dataset.Path))); err != nil {
		return nil, err
	}
	return locations, nil
}

func writeSummary(election audioparity.Election, existing bool) error {
	summary := election.Summary()
	fmt.Printf(
		"audio oracle election=%s model=%s files=%d fixtures=%d standard=%d equivalent=%d cpu_wall_ns=%d existing=%t; audit: every tracked model byte, source commit, corpus shard, transcript, runtime, license, and reopen condition is bound; embedded fixture audio identities were produced by the recorded reference execution\n",
		election.ID, election.Model.ID, len(election.Files), summary.Fixtures,
		summary.StandardMatches, summary.EquivalentMatches, summary.WallNS, existing,
	)
	return nil
}
