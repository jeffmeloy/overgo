// Command offline-artifact-validate records real-load and repeated-generation
// promotion evidence for one produced offline model artifact.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/composition"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/repodb"
)

type validationRequest struct {
	Trials []composition.OfflineArtifactGenerationTrial `json:"trials"`
	Policy validationPolicy                             `json:"policy"`
}

type validationPolicy struct {
	MinimumDistinctSeeds uint32 `json:"minimum_distinct_seeds"`
	RunsPerSeed          uint32 `json:"runs_per_seed"`
	MaximumLatencyNS     uint64 `json:"maximum_latency_ns"`
	MaximumPeakHostBytes uint64 `json:"maximum_peak_host_bytes"`
	Rationale            string `json:"rationale"`
	ReopenTrigger        string `json:"reopen_trigger"`
}

func main() {
	clioptions.MainNamed("offline-artifact-validate", run)
}

func run() error {
	flags := flag.NewFlagSet("offline-artifact-validate", flag.ContinueOnError)
	flags.SetOutput(io.Discard)
	repository := flags.String("record", "", "RepoDB root")
	planText := flags.String("plan", "", "offline tensor execution plan ID")
	modelText := flags.String("model", "", "produced model ID")
	directory := flags.String("directory", "", "produced Safetensors directory")
	requestPath := flags.String("request", "", "generation trials and explicit promotion policy JSON")
	if err := flags.Parse(os.Args[1:]); err != nil {
		return err
	}
	if flags.NArg() != 0 || *repository == "" || *planText == "" || *modelText == "" || *directory == "" || *requestPath == "" {
		return errors.New("usage: offline-artifact-validate -record STORE -plan ID -model ID -directory PATH -request JSON")
	}
	planID, err := artifact.ParseID(*planText)
	if err != nil || planID.Kind() != artifact.KindProfile {
		return errors.Join(errors.New("offline artifact validation plan ID is invalid"), err)
	}
	modelID, err := artifact.ParseID(*modelText)
	if err != nil || modelID.Kind() != artifact.KindModel {
		return errors.Join(errors.New("offline artifact validation model ID is invalid"), err)
	}
	var request validationRequest
	if err := jsonfile.Decode(*requestPath, &request); err != nil {
		return err
	}
	ctx := context.Background()
	store, err := repodb.Open(*repository)
	if err != nil {
		return err
	}
	defer store.Close()
	plan, err := composition.LoadOfflineTensorExecutionPlan(ctx, store, planID)
	if err != nil {
		return err
	}
	generation, err := composition.ValidateOfflineArtifactGeneration(plan, *directory, modelID, request.Trials)
	if err != nil {
		return err
	}
	policy, err := evaluation.NewOfflineArtifactGenerationPolicy(
		request.Policy.MinimumDistinctSeeds, request.Policy.RunsPerSeed,
		request.Policy.MaximumLatencyNS, request.Policy.MaximumPeakHostBytes,
		request.Policy.Rationale, request.Policy.ReopenTrigger,
	)
	if err != nil {
		return err
	}
	promotion, err := evaluation.PromoteOfflineArtifactGeneration(generation, policy)
	if err != nil {
		return err
	}
	generationContent, err := generation.Content()
	if err != nil {
		return err
	}
	policyContent, err := policy.Content()
	if err != nil {
		return err
	}
	promotionContent, err := promotion.Content()
	if err != nil {
		return err
	}
	lineage := append(generation.Lineage(), promotion.Lineage()...)
	batch, err := artifact.NewDocumentBatch(
		"offline-artifact-validation/"+promotion.ID.String(),
		[]artifact.Content{generationContent, policyContent, promotionContent}, lineage, nil,
	)
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return err
	}
	fmt.Printf("OFFLINE ARTIFACT GENERATION GREEN model=%s plan=%s evidence=%s promotion=%s\n",
		modelID, planID, generation.ID, promotion.ID)
	return nil
}
