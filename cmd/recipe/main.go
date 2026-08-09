// recipe: drives the model-recipe lifecycle for real artifacts — the command
// the sealed authority was waiting for (the lifecycle API was complete but
// uninvoked; the wave-1 probe recorded the refusal that proved it).
//
//	recipe activate -reason "..." <model>   publish facts + definition,
//	                                        candidate -> validated -> active
//	recipe status <model>                   show the active recipe and tier
//
// Model references resolve through the data-root contract; the store is the
// data-root store unless -repo overrides. Activation records its reason and
// decider commit in the decision event, per the store's decision discipline.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/hfrepo"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/tabularicl"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "recipe: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	if len(os.Args) < 2 {
		return errors.New("usage: recipe <activate|status> [-repo <dir>] [-reason <text>] <model>")
	}
	verb := os.Args[1]
	flags := flag.NewFlagSet("recipe "+verb, flag.ContinueOnError)
	repoFlag := flags.String("repo", "", "RepoDB store; empty resolves via the data-root contract")
	reason := flags.String("reason", "", "activation reason recorded in the decision event (activate)")
	task := flags.String("task", string(recipe.TaskInference), "recipe task (inference|forecast|tabular|seq2seq)")
	if err := flags.Parse(os.Args[2:]); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		return errors.New("exactly one model reference is required")
	}
	working, err := os.Getwd()
	if err != nil {
		return err
	}
	roots, err := dataroot.Resolve(working)
	if err != nil {
		return err
	}
	repository := strings.TrimSpace(*repoFlag)
	if repository == "" {
		repository = roots.Store
	}
	path := roots.ResolveModelPath(flags.Arg(0))
	switch verb {
	case "activate":
		if strings.TrimSpace(*reason) == "" {
			return errors.New("activate requires -reason: the decision event records why")
		}
		switch capability := recipe.Task(*task); capability {
		case recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq:
			return activateCapability(repository, path, *reason, capability)
		}
		return activate(repository, path, *reason)
	case "status":
		return status(repository, path, recipe.Task(*task))
	default:
		return fmt.Errorf("unknown verb %q", verb)
	}
}

// capabilityInventory: per-task artifact inventory. Forecast and seq2seq are
// standard HF safetensors directories (extra vendor sidecars ignored by the
// companion whitelist); tabular is the dual-head explicit file list.
func capabilityInventory(task recipe.Task, path string) (modelartifact.Inventory, error) {
	switch task {
	case recipe.TaskForecast, recipe.TaskSeq2Seq:
		repo, err := hfrepo.Open(path)
		if err != nil {
			return modelartifact.Inventory{}, err
		}
		defer repo.Close()
		return modelartifact.FromHFRepository(repo)
	case recipe.TaskTabular:
		return tabularInventory(path)
	}
	return modelartifact.Inventory{}, fmt.Errorf("no capability inventory for task %q", task)
}

// capabilityDefinition: per-task single-node host definition constructor.
func capabilityDefinition(task recipe.Task, modelID artifact.ID) (recipe.Definition, error) {
	switch task {
	case recipe.TaskForecast:
		return modelrecipe.ForecastDefinition(modelID)
	case recipe.TaskTabular:
		return modelrecipe.TabularDefinition(modelID)
	case recipe.TaskSeq2Seq:
		return modelrecipe.Seq2SeqDefinition(modelID)
	}
	return recipe.Definition{}, fmt.Errorf("no capability definition for task %q", task)
}

// tabularInventory: dual-head artifact (classification/ + regression/, each
// config.json + model.safetensors) — an explicit file list; no repository
// walker owns this layout.
func tabularInventory(path string) (modelartifact.Inventory, error) {
	var specs []modelartifact.FileSpec
	for _, head := range []string{tabularicl.TaskClassification, tabularicl.TaskRegression} {
		specs = append(specs,
			modelartifact.FileSpec{
				Path: filepath.Join(path, head, "config.json"),
				Name: head + "/config", Role: artifact.ComponentConfig,
			},
			modelartifact.FileSpec{
				Path: filepath.Join(path, head, "model.safetensors"),
				Name: head + "/weights", Role: artifact.ComponentWeights,
			},
		)
	}
	return modelartifact.FromFiles(path, specs)
}

// activateCapability drives the shared sealed lifecycle for capability-package
// models: inventory facts, then candidate -> validated -> active with the
// decision evidence. Capability packages derive dimensions from the artifact,
// so no profile document is published.
func activateCapability(repository, path, reason string, task recipe.Task) error {
	ctx := context.Background()
	inventory, err := capabilityInventory(task, path)
	if err != nil {
		return err
	}
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID := inventory.Manifest.ID
	batch, err := inventory.Batch("recipe/facts/" + modelID.String())
	if err != nil {
		return err
	}
	if _, err := store.Commit(ctx, batch); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	definition, err := capabilityDefinition(task, modelID)
	if err != nil {
		return err
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "recipe/candidate/"+definition.ID.String(), definition,
	); err != nil {
		return fmt.Errorf("publish candidate: %w", err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "recipe/validated/"+definition.ID.String(), definition,
		recipe.StatusValidated, nil, nil,
	); err != nil {
		return fmt.Errorf("transition validated: %w", err)
	}
	evidenceID, err := activationEvidence(ctx, store, definition, reason)
	if err != nil {
		return err
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "recipe/active/"+definition.ID.String(), definition,
		recipe.StatusActive, []artifact.ID{evidenceID}, nil,
	); err != nil {
		return fmt.Errorf("transition active: %w", err)
	}
	fmt.Printf("activated %s\n  task       %s\n  model      %s\n  recipe     %s\n  reason     %s\n",
		path, task, modelID, definition.ID, reason)
	return nil
}

func activate(repository, path, reason string) error {
	ctx := context.Background()
	file, err := gguf.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	inventory, err := modelartifact.FromGGUF(file)
	if err != nil {
		return err
	}
	spec, err := model.ReadSpec(file)
	if err != nil {
		return err
	}
	profile, ok := model.LookupArchitecture(spec.Architecture)
	if !ok {
		return fmt.Errorf("architecture %q has no registered profile", spec.Architecture)
	}
	profileDocument, err := modelrecipe.NewProfileDocument(profile)
	if err != nil {
		return err
	}
	document, err := modelrecipe.NewModelDefinitionFromGGUF(file, profileDocument, inventory.TensorInventory)
	if err != nil {
		return err
	}
	resolved, err := document.Resolve(profileDocument, inventory.TensorInventory)
	if err != nil {
		return err
	}
	store, err := repodb.Open(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	modelID := inventory.Manifest.ID
	if _, err := modelrecipe.PublishResolvedModelDefinition(
		ctx, store, "recipe/facts/"+modelID.String(), inventory, resolved,
	); err != nil {
		return fmt.Errorf("publish model facts: %w", err)
	}
	definition, err := modelrecipe.InferenceWithModelDefinition(
		modelID, resolved.Profile.ID, resolved.Document.ID, recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity,
	)
	if err != nil {
		return err
	}
	if _, _, err := modelrecipe.PublishCandidate(
		ctx, store, "recipe/candidate/"+definition.ID.String(), definition,
	); err != nil {
		return fmt.Errorf("publish candidate: %w", err)
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "recipe/validated/"+definition.ID.String(), definition,
		recipe.StatusValidated, nil, nil,
	); err != nil {
		return fmt.Errorf("transition validated: %w", err)
	}
	evidenceID, err := activationEvidence(ctx, store, definition, reason)
	if err != nil {
		return err
	}
	if _, _, err := modelrecipe.Transition(
		ctx, store, "recipe/active/"+definition.ID.String(), definition,
		recipe.StatusActive, []artifact.ID{evidenceID}, nil,
	); err != nil {
		return fmt.Errorf("transition active: %w", err)
	}
	fmt.Printf("activated %s\n  model      %s\n  definition %s\n  recipe     %s\n  reason     %s\n",
		path, modelID, resolved.Document.ID, definition.ID, reason)
	return nil
}

// activationEvidence records the decision basis: reason, decider commit, and
// the honest statement that serving-parity evidence follows as run records.
func activationEvidence(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	reason string,
) (artifact.ID, error) {
	decider := "unknown"
	if out, err := exec.Command("git", "rev-parse", "HEAD").Output(); err == nil {
		decider = strings.TrimSpace(string(out))
	}
	payload := []byte("activation decision\nreason: " + reason + "\ndecider_commit: " + decider +
		"\nfollow_up: serving parity evidence lands as run records against this recipe\n")
	id, err := artifact.IdentifyBytes(artifact.KindEvidence, payload)
	if err != nil {
		return artifact.ID{}, err
	}
	_, err = store.Commit(ctx, artifact.Batch{
		Key: "recipe/activation-evidence/" + definition.ID.String(),
		Contents: []artifact.Content{{
			Descriptor: artifact.Descriptor{ID: id, Size: uint64(len(payload))},
			Data:       payload,
		}},
	})
	if err != nil {
		return artifact.ID{}, err
	}
	return id, nil
}

func status(repository, path string, task recipe.Task) error {
	ctx := context.Background()
	var inventory modelartifact.Inventory
	switch task {
	case recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq:
		var err error
		inventory, err = capabilityInventory(task, path)
		if err != nil {
			return err
		}
	default:
		file, err := gguf.Open(path)
		if err != nil {
			return err
		}
		defer file.Close()
		inventory, err = modelartifact.FromGGUF(file)
		if err != nil {
			return err
		}
	}
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, inventory.Manifest.ID, task)
	if err != nil {
		return err
	}
	if !active {
		fmt.Printf("%s\n  model  %s\n  active %s recipe: ABSENT\n", path, inventory.Manifest.ID, task)
		return nil
	}
	fmt.Printf("%s\n  model  %s\n  recipe %s\n  tier   %s\n",
		path, inventory.Manifest.ID, activation.Definition.ID, activation.Tier)
	return nil
}
