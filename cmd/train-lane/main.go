// train-lane: the store-derived training smoke matrix. Every model with an
// active training-task recipe and present bytes gets one bounded recorded
// training step through the same workflow a user runs; models whose
// objective needs inputs a smoke cannot supply, and models whose bytes are
// absent, are reported by name, never green and never silently skipped.
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
)

// smokeCorpus is the bounded deterministic dataset for one training step:
// enough tokens to fill one short batch, identical on every run.
const smokeCorpus = "The quick brown fox jumps over the lazy dog. " +
	"Pack my box with five dozen liquor jugs. " +
	"Sphinx of black quartz, judge my vow. " +
	"How vexingly quick daft zebras jump. "

const trainLaneCatalogBound = 10_000

func main() { clioptions.MainNamed("train-lane", run) }

func run() error {
	steps := flag.Int("steps", 1, "bounded update steps per dense-workflow model")
	sequence := flag.Int("seq", 128, "maximum token sequence for the bounded dense step")
	maxWall := flag.Duration("max-wall", 10*time.Minute, "projected-wall bound per dense step; declared routes carry their own recorded bounds")
	routesPath := flag.String("routes", filepath.Join("docs", "training_routes.json"), "committed architecture-to-trainer route declarations")
	flag.Parse()
	catalog, err := loadRouteCatalog(*routesPath)
	if err != nil {
		return err
	}
	roots, err := dataroot.ResolveCurrent()
	if err != nil {
		return err
	}
	ctx := context.Background()
	reader, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		return err
	}
	entries, truncated, err := discovery.CapabilityCatalog(ctx, reader, trainLaneCatalogBound, nil)
	closeErr := reader.Close()
	if err != nil {
		return err
	}
	if closeErr != nil {
		return closeErr
	}
	if truncated {
		return runrecord.LaneError(runrecord.LaneFailed, "capability catalog truncated; the matrix would be incomplete")
	}
	corpus := filepath.Join(os.TempDir(), "overgo-train-lane-corpus.txt")
	if err := os.WriteFile(corpus, []byte(strings.Repeat(smokeCorpus, 32)), 0o644); err != nil {
		return err
	}
	t2vDir := resolveT2VDirectory(ctx, roots.Store, entries)
	trainable := 0
	passed, failed, unavailable, unsupported := 0, 0, 0, 0
	for _, entry := range entries {
		var training *discovery.Capability
		for index := range entry.Capabilities {
			if entry.Capabilities[index].Task == recipe.TaskTraining {
				training = &entry.Capabilities[index]
				break
			}
		}
		if training == nil {
			continue
		}
		trainable++
		if training.Stale != "" {
			unavailable++
			fmt.Printf("[train] %s STALE (%s)\n", entry.Model, training.Stale)
			continue
		}
		if !entry.Present {
			unavailable++
			fmt.Printf("[train] %s UNAVAILABLE (recorded location missing: %s)\n", entry.Model, entry.Location)
			continue
		}
		modelInput, inputErr := trainableInput(ctx, roots.Store, entry)
		if inputErr != nil {
			unavailable++
			fmt.Printf("[train] %s UNAVAILABLE (%s)\n", entry.Model, inputErr)
			continue
		}
		output, err := os.MkdirTemp("", "overgo-train-lane-*")
		if err != nil {
			return err
		}
		// The checkpoint target must not exist yet: publication owns its
		// creation, so the dense route gets a fresh subpath of the lane's
		// temporary directory.
		route, routeErr := resolveRoute(catalog, modelInput, roots.Store, t2vDir,
			denseArgv(roots.Store, training.Recipe.String(), modelInput, corpus,
				filepath.Join(output, "checkpoint"), *steps, *sequence, *maxWall))
		if routeErr != nil {
			unavailable++
			fmt.Printf("[train] %s UNAVAILABLE (%s)\n", entry.Model, routeErr)
			if err := os.RemoveAll(output); err != nil {
				return err
			}
			continue
		}
		began := time.Now()
		stepErr := trainStep(route)
		wall := time.Since(began)
		removeErr := os.RemoveAll(output)
		switch {
		case stepErr == nil:
			passed++
			fmt.Printf("[train] %s route=%s succeeded %7.1fs tier=%s\n", entry.Model, route.Name, wall.Seconds(), training.Tier)
		case isObjectiveUnsupported(stepErr):
			unsupported++
			fmt.Printf("[train] %s route=%s OBJECTIVE-UNSUPPORTED (%s)\n", entry.Model, route.Name, clioptions.Tail(stepErr.Error(), 200))
		default:
			failed++
			fmt.Printf("[train] %s route=%s failed    %7.1fs\n        %s\n", entry.Model, route.Name, wall.Seconds(), clioptions.Tail(stepErr.Error(), 600))
		}
		if removeErr != nil {
			return removeErr
		}
	}
	if trainable == 0 {
		fmt.Println("train-lane: 0 training-activated models; nothing to train (honest empty, not green)")
		return runrecord.LaneError(runrecord.LaneEmpty, "capability catalog has no training activations")
	}
	fmt.Printf("=== TRAIN %d passed / %d failed / %d unavailable / %d objective-unsupported of %d training-activated ===\n",
		passed, failed, unavailable, unsupported, trainable)
	fmt.Println("honesty: matrix derived from training activations; each step runs the recipe-authorized workflow and records its session observation; objective-unsupported rows need RL inputs a smoke cannot supply")
	if failed > 0 {
		return runrecord.LaneError(runrecord.LaneFailed, fmt.Sprintf("%d failed, %d unavailable", failed, unavailable))
	}
	if unavailable > 0 {
		return runrecord.LaneError(runrecord.LaneUnavailable, fmt.Sprintf("%d model artifacts unavailable", unavailable))
	}
	return nil
}

// trainableInput resolves the model input the routed trainer loads: a
// recorded weights file — GGUF or a raw checkpoint, each a different
// representation of the same data — or a recorded directory holding a
// model declaration. A model with neither on record is reported, never
// guessed.
func trainableInput(ctx context.Context, storePath string, entry discovery.CatalogEntry) (string, error) {
	reader, err := overgodb.OpenReadOnly(storePath)
	if err != nil {
		return "", err
	}
	defer reader.Close()
	if file, err := artifact.AvailablePath(ctx, reader, entry.Model, artifact.LocationFile); err == nil {
		switch strings.ToLower(filepath.Ext(file)) {
		case ".gguf", ".pt":
			return file, nil
		}
	}
	directory, err := artifact.AvailablePath(ctx, reader, entry.Model, artifact.LocationDirectory)
	if err != nil {
		return "", fmt.Errorf("no recorded model input: %v", err)
	}
	if _, err := os.Stat(filepath.Join(directory, "config.json")); err != nil {
		return "", fmt.Errorf("recorded location %s is not a trainable model input", directory)
	}
	return directory, nil
}

// denseArgv is the dense-workflow invocation: the recipe-authorized
// bounded step through cmd/train, lexically frozen so the update stays
// device-resident inside the lane's wall instead of grinding the host
// orthogonalization of the tied vocabulary matrix.
func denseArgv(store, recipeID, model, dataset, output string, steps, sequence int, maxWall time.Duration) []string {
	return []string{
		"go", "run", "./cmd/train",
		"-store", store, "-recipe", recipeID, "-model", model,
		"-dataset", dataset, "-out", output,
		"-steps", fmt.Sprintf("%d", steps), "-seq", fmt.Sprintf("%d", sequence),
		"-max-wall", maxWall.String(), "-freeze-lexical",
	}
}

// trainStep runs one bounded recorded training step through the routed
// trainer subprocess, supervised by the process owner; the trainer
// itself records the session observation to the store.
func trainStep(route trainerRoute) error {
	var combined bytes.Buffer
	var env []string
	if len(route.Env) > 0 {
		env = append(os.Environ(), route.Env...)
	}
	receipt, err := processcontrol.Run(context.Background(), processcontrol.Command{
		Path:   route.Argv[0],
		Args:   route.Argv[1:],
		Env:    env,
		Stdout: &combined, Stderr: &combined,
	})
	if err != nil {
		return fmt.Errorf("%v: %s", err, clioptions.Tail(combined.String(), 1200))
	}
	if receipt.ExitCode != 0 {
		return fmt.Errorf("exit status %d: %s", receipt.ExitCode, clioptions.Tail(combined.String(), 1200))
	}
	return nil
}

// isObjectiveUnsupported recognizes the exact recipe refusals for
// objectives whose required inputs — a frozen reference model or an
// objective scale — a bounded smoke cannot supply.
func isObjectiveUnsupported(err error) bool {
	message := err.Error()
	return strings.Contains(message, "DPO inputs differ from recipe") ||
		strings.Contains(message, "GRPO inputs differ from recipe")
}
