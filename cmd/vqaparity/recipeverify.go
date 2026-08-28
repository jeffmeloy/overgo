package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/cuda/driver"
	"overgo/internal/modelartifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/workflowruntime"
)

func codeRevision() (string, error) {
	if info, ok := debug.ReadBuildInfo(); ok {
		for _, setting := range info.Settings {
			if setting.Key == "vcs.revision" && len(setting.Value) == 40 {
				return setting.Value, nil
			}
		}
	}
	return runrecord.HeadCommit(".")
}

func cudaEnvironment() (runrecord.Environment, error) {
	library, err := driver.Open()
	if err != nil {
		return runrecord.Environment{}, err
	}
	defer library.Close()
	if err := library.Init(); err != nil {
		return runrecord.Environment{}, err
	}
	version, err := library.DriverVersion()
	if err != nil {
		return runrecord.Environment{}, err
	}
	device, err := library.DeviceInfo(0)
	if err != nil {
		return runrecord.Environment{}, err
	}
	host, err := os.Hostname()
	if err != nil {
		host = "unknown-host"
	}
	return runrecord.NewEnvironment(runrecord.Environment{
		Host: host, OS: runtime.GOOS, Arch: runtime.GOARCH, Device: device.Name,
		Backend: "cuda", Driver: version.String(), Runtime: runtime.Version(),
	})
}

func publishVQAVerification(
	ctx context.Context,
	store artifact.Repository,
	definition recipe.Definition,
	wall time.Duration,
	walls []workflowruntime.NodeWall,
) (modelrecipe.Verification, error) {
	environment, err := cudaEnvironment()
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	revision, err := codeRevision()
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	duration := uint64(max(wall.Nanoseconds(), 1))
	steps := []runrecord.GateStep{{
		Name: "rxbrain-canonical-vqa", Phase: runrecord.PhaseTest,
		Outcome: runrecord.StepSucceeded, DurationNS: duration,
	}}
	phases := map[recipe.NodeID]runrecord.Phase{"prepare": runrecord.PhasePrepare, "generate": runrecord.PhaseGenerate}
	for _, node := range walls {
		phase, mapped := phases[node.Node]
		if !mapped || node.WallNS == 0 {
			continue
		}
		steps = append(steps, runrecord.GateStep{
			Name: "node-" + string(node.Node), Phase: phase,
			Outcome: runrecord.StepSucceeded, DurationNS: node.WallNS,
		})
	}
	record, err := runrecord.NewGateRecord(
		definition.ID, environment.ID, revision, runrecord.OutcomeSucceeded, "", duration, steps,
	)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	batch, err := record.Batch("vqa/verification/" + definition.ID.String() + "/" + record.Result.ID.String())
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	environmentContent, err := environment.Content()
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	batch.Contents = append(batch.Contents, environmentContent)
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.Verification{}, err
	}
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}, nil
}

// verifyActivateVQA: real processor/device proof, promotion, active replay.
func verifyActivateVQA(l *campaignContext, repo, imagePath, question string) error {
	ctx := context.Background()
	store, err := openRecipeStore(repo)
	if err != nil {
		return err
	}
	defer store.Close()
	inventory, err := modelartifact.FromHFPath(l.modelDir)
	if err != nil {
		return err
	}
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskVQA, inventory.Manifest.ID)
	if err != nil {
		return err
	}
	batch, err := inventory.Batch("vqa/model/" + inventory.Manifest.ID.String())
	if err != nil {
		return err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return err
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		return err
	}
	verified, err := executeVQAProgram(
		l, ctx, store, inventory.Manifest.ID, program, imagePath, question, "VERIFY",
	)
	if err != nil {
		return err
	}
	if err := validateCanonicalVQA(l, verified, "VERIFY"); err != nil {
		return err
	}
	verification, err := publishVQAVerification(ctx, store, definition, verified.result.e2eWall, verified.walls)
	if err != nil {
		return err
	}
	if err := modelrecipe.ActivateCapability(
		ctx, store, definition, verification, recipe.EvidenceParity,
		"canonical RxBrain processor, exact golden prefix, coherent EOS answer",
	); err != nil {
		return err
	}
	activation, activeProgram, err := modelrecipe.ResolveActiveCapability(
		ctx, store, inventory.Manifest.ID, recipe.TaskVQA,
	)
	if err != nil {
		return err
	}
	if activation.Definition.ID != definition.ID {
		return errors.New("VQA verifier: activation resolved another recipe")
	}
	active, err := executeVQAProgram(
		l, ctx, store, inventory.Manifest.ID, activeProgram, imagePath, question, "ACTIVE",
	)
	if err != nil {
		return err
	}
	if err := validateCanonicalVQA(l, active, "ACTIVE"); err != nil {
		return err
	}
	if !slices.Equal(active.chain, verified.chain) || active.text != verified.text {
		return errors.New("VQA verifier: active replay differs from promotion run")
	}
	l.Log(fmt.Sprintf(
		"RECIPE verify LANE GREEN recipe=%s gate=%s run=%s verify=%s active=%s",
		definition.ID, verification.Gate, verification.Run,
		verified.result.e2eWall.Round(time.Millisecond), active.result.e2eWall.Round(time.Millisecond),
	))
	return nil
}
