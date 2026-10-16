package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/trainingworkflow"
)

// trainSpeech runs the registered dataset objective and publishes an inactive speech candidate.
func trainSpeech(ctx context.Context, flags *flag.FlagSet, args []string) error {
	recipeText := flags.String("recipe", "", "exact training recipe with registered dataset membership and latent processing profile")
	objectiveAlias := flags.String("objective", "", "registered dataset objective to bind to the manifest's exact speech base; exclusive with recipe")
	manifest := flags.String("manifest", "", "speech training manifest with base_recipe, profile, memory_bytes, seed and shuffle")
	steps := clioptions.IntOverride(flags, "steps", "additional Muon updates; omission reads one dataset pass")
	maxWall := clioptions.DurationOverride(flags, "max-wall", "optional projected-wall bound")
	storePath := flags.String("store", "overgodb-store", "OvergoDB holding the dataset and training recipe")
	out := flags.String("out", "", "new checkpoint directory")
	resume := flags.String("resume", "", "checkpoint whose exact next example and native optimizer state continue")
	host := flags.Bool("host", false, "use the CPU numerical reference instead of the platform Muon backend")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if *manifest == "" || *out == "" || *steps < 0 || flags.NArg() != 0 || (*recipeText == "") == (*objectiveAlias == "") {
		return fmt.Errorf("%s: exactly one recipe or objective, manifest, new out and nonnegative steps required", flags.Name())
	}
	var spec trainingworkflow.SpeechTrainingSpec
	if err := jsonfile.Decode(*manifest, &spec); err != nil {
		return err
	}
	store, err := overgodb.OpenContext(ctx, *storePath)
	if err != nil {
		return err
	}
	defer store.Close()
	var id artifact.ID
	if *recipeText != "" {
		id, err = artifact.ParseID(*recipeText)
	} else {
		base, baseErr := recipe.RequireDefinition(ctx, store, spec.BaseRecipe)
		if baseErr != nil {
			return baseErr
		}
		if base.Task != recipe.TaskSpeech {
			return fmt.Errorf("speech training: base task %q is not speech", base.Task)
		}
		objective, objectiveErr := trainingworkflow.LoadObjectiveAlias(ctx, store, *objectiveAlias)
		if objectiveErr != nil {
			return objectiveErr
		}
		id, err = trainingworkflow.BootstrapObjectiveForModel(ctx, store, base.Model, objective)
	}
	if err != nil {
		return err
	}
	result, err := trainingworkflow.Execute(ctx, trainingworkflow.Request{Repository: store, Recipe: id, Speech: &spec,
		OutputDirectory: *out, ResumeDirectory: *resume, Steps: *steps, Host: *host, MaxProjectedWall: *maxWall, Observations: store, Progress: os.Stdout})
	if err != nil {
		return err
	}
	fmt.Printf("checkpoint=%s candidate=%s inactive stream_position=%d\n", result.Checkpoint.ID(), result.Candidate, result.StreamPosition)
	return nil
}
