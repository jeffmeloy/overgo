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
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

type Capability struct {
	Model   artifact.ID
	Program recipe.Program
	Runtime *workflowruntime.Runtime
}

func NewCapability(t testing.TB, name string, task recipe.Task) Capability {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	check(t, err)
	t.Cleanup(func() { check(t, store.Close()) })
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	testutil.PublishArtifact(t, store, modelID)
	compiled, err := modelrecipe.CapabilityDefinition(task, modelID)
	check(t, err)
	program, err := modelrecipe.CompileCapability(compiled)
	check(t, err)
	runtime, err := workflowruntime.NewWithCatalog(store, modelrecipe.Catalog())
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
	return f.Runtime.ExecuteProgram(context.Background(), key, f.Program, map[recipe.PortName]workflowruntime.Value{
		input.Name: {Kind: input.Data, Items: []workflowruntime.Datum{{Value: value}}},
	})
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
