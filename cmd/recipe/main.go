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
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/dataroot"
	"overgo/internal/gguf"
	"overgo/internal/model"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
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
		return activate(repository, path, *reason)
	case "status":
		return status(repository, path)
	default:
		return fmt.Errorf("unknown verb %q", verb)
	}
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

func status(repository, path string) error {
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
	store, err := repodb.OpenReadOnly(repository)
	if err != nil {
		return err
	}
	defer store.Close()
	activation, active, err := modelrecipe.ActiveRecord(ctx, store, inventory.Manifest.ID, recipe.TaskInference)
	if err != nil {
		return err
	}
	if !active {
		fmt.Printf("%s\n  model  %s\n  active inference recipe: ABSENT\n", path, inventory.Manifest.ID)
		return nil
	}
	fmt.Printf("%s\n  model  %s\n  recipe %s\n  tier   %s\n",
		path, inventory.Manifest.ID, activation.Definition.ID, activation.Tier)
	return nil
}
