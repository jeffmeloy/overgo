// Package modelrecipetest owns executable capability fixtures.
package modelrecipetest

import (
	"context"
	"fmt"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

const verificationCodeCommit = "0123456789abcdef0123456789abcdef01234567"

type Capability struct {
	Model   artifact.ID
	Program recipe.Program
	Runtime *workflowruntime.Runtime
}

func PublishVerification(
	ctx context.Context,
	store artifact.Repository,
	key string,
	recipeID artifact.ID,
) (modelrecipe.Verification, error) {
	environment, err := artifact.IdentifyBytes(artifact.KindEvidence, []byte(key+"/environment"))
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: key + "/environment",
		Artifacts: []artifact.Descriptor{{
			ID: environment, Size: uint64(len(key + "/environment")),
		}},
	}); err != nil {
		return modelrecipe.Verification{}, err
	}
	record, err := runrecord.NewGateRecord(
		recipeID, environment, verificationCodeCommit,
		runrecord.OutcomeSucceeded, "", 1,
		[]runrecord.GateStep{{
			Name: "verify", Phase: runrecord.PhaseValidate,
			Outcome: runrecord.StepSucceeded, DurationNS: 1,
		}},
	)
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	batch, err := record.Batch(key + "/gate")
	if err != nil {
		return modelrecipe.Verification{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return modelrecipe.Verification{}, err
	}
	return modelrecipe.Verification{Gate: record.Result.ID, Run: record.Run.ID}, nil
}

func NewCapability(t testing.TB, name string, task recipe.Task) Capability {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	check(t, err)
	t.Cleanup(func() { check(t, store.Close()) })
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	testutil.PublishArtifact(t, store, modelID)
	var compiled recipe.Definition
	if task == recipe.TaskImageGen {
		compiled, err = modelrecipe.OscillatorImageDefinition(modelID)
	} else if task == recipe.TaskVideoGen {
		compiled, err = modelrecipe.OscillatorVideoDefinition(modelID)
	} else {
		compiled, err = modelrecipe.CapabilityDefinition(task, modelID)
	}
	return newDefinedCapability(t, store, modelID, compiled, err)
}

func newDefinedCapability(t testing.TB, store artifact.Repository, modelID artifact.ID, compiled recipe.Definition, err error) Capability {
	t.Helper()
	check(t, err)
	program, err := modelrecipe.CompileCapability(compiled)
	check(t, err)
	runtime, err := workflowruntime.NewForProgram(store, program)
	check(t, err)
	return Capability{Model: modelID, Program: program, Runtime: runtime}
}

func check(t testing.TB, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func (f Capability) ExecuteScalar(key string, value any) (workflowruntime.Result, error) {
	definition := f.Program.Definition()
	if len(definition.Inputs) != 1 {
		return workflowruntime.Result{}, fmt.Errorf("model recipe fixture: want one input, got %d", len(definition.Inputs))
	}
	input := definition.Inputs[0]
	operation, err := workflowruntime.ExecutionID(definition.ID, key)
	if err != nil {
		return workflowruntime.Result{}, err
	}
	return f.Runtime.ExecuteProgram(context.Background(), key, operation, nil, f.Program, map[recipe.PortName]workflowruntime.Value{
		input.Name: {Kind: input.Data, Items: []workflowruntime.Datum{{Value: value}}},
	})
}

// MustExecuteScalar runs a scalar capability and returns its typed output.
func MustExecuteScalar[Value any](t testing.TB, fixture Capability, key string, input any) Value {
	t.Helper()
	result, err := fixture.ExecuteScalar(key, input)
	check(t, err)
	outputs := fixture.Program.Definition().Outputs
	if len(outputs) != 1 {
		t.Fatalf("model recipe fixture: want one output, got %d", len(outputs))
	}
	return Output[Value](t, result, outputs[0].Name)
}

func Output[Value any](t testing.TB, result workflowruntime.Result, name recipe.PortName) Value {
	t.Helper()
	datum, one := result.Outputs[name].Single()
	value, typed := datum.Value.(Value)
	if !one || !typed || !result.Commit.Valid() {
		t.Fatalf("model recipe output %q = (%T, %v, %v), commit=%v", name, datum.Value, one, typed, result.Commit)
	}
	return value
}
