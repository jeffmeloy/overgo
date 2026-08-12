package capabilityruntime

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

type scalarRequest struct {
	Value int `json:"value"`
}

func capabilityFixture(t *testing.T, name string, task recipe.Task) (*repodb.Store, artifact.ID, recipe.Program) {
	t.Helper()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	modelID := testutil.ArtifactID(t, artifact.KindModel, name)
	testutil.PublishArtifact(t, store, modelID)
	definition, err := modelrecipe.CapabilityDefinition(task, modelID)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	return store, modelID, program
}

func TestJSONScalarExecutesIdentityBoundProgram(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "scalar-model", recipe.TaskImageGen)
	execute := JSONScalar[scalarRequest, int, int](
		"scalar", func(request scalarRequest) error {
			if request.Value <= 0 {
				return errors.New("positive value required")
			}
			return nil
		},
		func(path string, request scalarRequest) (int, error) { return len(path) + request.Value, nil },
		func(runtime *workflowruntime.Runtime, bound artifact.ID, model int) error {
			return workflowruntime.RegisterJSONStage[scalarRequest, int](
				runtime, modelrecipe.ModuleImageGenerate, bound,
				artifact.JSONContract(artifact.KindOutput, "test.scalar-output.v1"),
				func(request scalarRequest) (int, error) { return request.Value + model, nil },
			)
		},
	)
	got, err := execute(context.Background(), store, "abc", modelID, program, `{"value":4}`)
	if err != nil {
		t.Fatal(err)
	}
	if got != 11 {
		t.Fatalf("output = %v, want 11", got)
	}
	if _, err := execute(context.Background(), store, "abc", modelID, program, `{"value":4,"extra":1}`); err == nil {
		t.Fatal("unknown input field accepted")
	}
	if program.Definition().Task != recipe.TaskImageGen {
		t.Fatalf("task = %s", program.Definition().Task)
	}
}

func TestExecuteVQABindsInputsAndOutputLineage(t *testing.T) {
	store, modelID, program := capabilityFixture(t, "vqa-model", recipe.TaskVQA)
	image := VQAImage{Data: []byte("image")}
	answer, err := ExecuteVQA(
		context.Background(), store, modelID, program, "vqa/runtime", image, "where?",
		func(_ context.Context, gotImage VQAImage, question string) (string, error) {
			if string(gotImage.Data) != string(image.Data) || question != "where?" {
				t.Fatalf("input = (%q, %q)", gotImage.Data, question)
			}
			return "there", nil
		},
	)
	if err != nil || answer != "there" {
		t.Fatalf("answer = (%q, %v)", answer, err)
	}
	if _, err := ExecuteVQA(
		context.Background(), store, modelID, program, "vqa/invalid", VQAImage{}, "where?",
		func(context.Context, VQAImage, string) (string, error) { return "", nil },
	); err == nil {
		t.Fatal("empty image accepted")
	}
}
