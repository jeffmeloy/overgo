// Command audio-oracle verifies and publishes one immutable external audio-oracle election.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/audioparity"
	"overgo/internal/clioptions"
	"overgo/internal/modelartifact"
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
	capturePath := flags.String("capture", "", "optional source-bound encoder capture JSON")
	captureSource := flags.String("capture-source", "", "capture script required with -capture")
	captureTensors := flags.String("capture-tensors", "", "captured Safetensors file required with -capture")
	transducer := flags.Bool("transducer-capture", false, "publish native transducer traces for an already registered model")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if *transducer {
		if flags.NArg() != 0 || *repository == "" || *capturePath == "" || *captureSource == "" || *captureTensors == "" || *recordPath != "" || *modelRoot != "" || *datasetRoot != "" {
			return errors.New("audio-oracle: transducer capture requires only repo, capture, capture-source and capture-tensors")
		}
		store, err := overgodb.Open(*repository)
		if err != nil {
			return err
		}
		defer store.Close()
		id, err := audioparity.PublishTransducerCapture(context.Background(), store, *capturePath, *captureSource, *captureTensors)
		if err != nil {
			return err
		}
		fmt.Printf("transducer capture=%s; source/model/tensor identities verified; Go parity, quality evaluation, training and GPU execution did not run\n", id)
		return nil
	}
	if flags.NArg() != 0 || *repository == "" || *recordPath == "" || *modelRoot == "" || *datasetRoot == "" {
		return errors.New("usage: audio-oracle -repo <store> -election <json> -model-root <directory> -dataset-root <directory>")
	}
	if (*capturePath == "") != (*captureSource == "") || (*capturePath == "") != (*captureTensors == "") {
		return errors.New("audio-oracle: capture requires report, source and tensor paths together")
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
	inventory, err := modelartifact.FromHFPath(*modelRoot)
	if err != nil {
		return err
	}
	qualification, err := audioparity.QualifyAudioArtifact(ctx, election, inventory, *modelRoot)
	if err != nil {
		return err
	}
	if !qualification.Loadable {
		return fmt.Errorf("audio-oracle: artifact is not loadable: %s", strings.Join(qualification.Refusals, "; "))
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
	qualified, qualificationFound, err := store.ResolveAlias(ctx, qualification.Alias())
	if err != nil {
		return err
	}
	if found && current == election.ID && qualificationFound && qualified == qualification.ID {
		if err := publishCapture(ctx, store, election, *capturePath, *captureSource, *captureTensors); err != nil {
			return err
		}
		return writeSummary(election, qualification, true)
	}
	batch, err := election.Batch("audio/oracle/qualification/" + qualification.ID.DigestHex())
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
	var previousQualification *artifact.ID
	if qualificationFound {
		previousQualification = &qualified
	}
	if err := qualification.AugmentBatch(&batch, inventory, previousQualification); err != nil {
		return err
	}
	if err := modelartifact.PreserveRawTextDescriptors(ctx, store, &batch); err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return err
	}
	if err := publishCapture(ctx, store, election, *capturePath, *captureSource, *captureTensors); err != nil {
		return err
	}
	return writeSummary(election, qualification, false)
}

func publishCapture(ctx context.Context, repository artifact.Repository, election audioparity.Election, report, source, tensors string) error {
	if report == "" {
		return nil
	}
	id, err := audioparity.PublishEncoderCapture(ctx, repository, election, report, source, tensors)
	if err != nil {
		return err
	}
	fmt.Printf("encoder capture=%s; verified source and tensor identities; Go numerical parity, training and device execution did not run\n", id)
	return nil
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

func writeSummary(election audioparity.Election, qualification audioparity.AudioArtifactQualification, existing bool) error {
	summary := election.Summary()
	fmt.Printf(
		"audio oracle election=%s qualification=%s model=%s files=%d fixtures=%d standard=%d equivalent=%d cpu_wall_ns=%d existing=%t; audit: every tracked model byte, tensor name and shape, architecture recipe, tokenizer, preprocessing asset, source commit, corpus shard, transcript, runtime, license, and reopen condition is bound\n",
		election.ID, qualification.ID, election.Model.ID, len(election.Files), summary.Fixtures,
		summary.StandardMatches, summary.EquivalentMatches, summary.WallNS, existing,
	)
	return nil
}
