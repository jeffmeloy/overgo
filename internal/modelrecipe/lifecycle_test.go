package modelrecipe

import (
	"context"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/model"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

func TestLifecyclePromotionAndSupersession(t *testing.T) {
	ctx := context.Background()
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	modelID, _ := artifact.IdentifyBytes(artifact.KindModel, []byte("lifecycle-model"))
	evidenceID, _ := artifact.IdentifyBytes(artifact.KindEvidence, []byte("validation"))
	if _, err := store.Commit(ctx, artifact.Batch{
		Key: "fixture/lifecycle/facts",
		Artifacts: []artifact.Descriptor{
			{ID: modelID, Size: uint64(len("lifecycle-model"))},
			{ID: evidenceID, Size: uint64(len("validation"))},
		},
	}); err != nil {
		t.Fatal(err)
	}
	first, err := Inference(modelID, recipe.PlacementHost)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/lifecycle/first/candidate", first); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/first/validated", first, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/first/active", first, recipe.StatusActive, []artifact.ID{evidenceID}, nil); err != nil {
		t.Fatal(err)
	}
	active, ok, err := Active(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || active.ID != first.ID {
		t.Fatalf("first active = (%s, %v, %v)", active.ID, ok, err)
	}

	second, err := Inference(modelID, recipe.PlacementDevice)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := PublishCandidate(ctx, store, "fixture/lifecycle/second/candidate", second); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(ctx, store, "fixture/lifecycle/second/validated", second, recipe.StatusValidated, nil, nil); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Transition(
		ctx, store, "fixture/lifecycle/second/active", second, recipe.StatusActive,
		[]artifact.ID{evidenceID}, &first.ID,
	); err != nil {
		t.Fatal(err)
	}
	active, ok, err = Active(ctx, store, modelID, recipe.TaskInference)
	if err != nil || !ok || active.ID != second.ID {
		t.Fatalf("second active = (%s, %v, %v)", active.ID, ok, err)
	}
	oldStatus, err := currentEvent(ctx, store, first.ID)
	if err != nil || oldStatus.To != recipe.StatusSuperseded {
		t.Fatalf("old status = (%+v, %v)", oldStatus, err)
	}
	compiled, ok, err := CompileActive(
		ctx, store, modelID,
		model.Spec{CommonSpec: model.CommonSpec{Architecture: "llama"}}, model.Weights{},
	)
	if err != nil || !ok || compiled.Recipe.ID != second.ID {
		t.Fatalf("compiled active = (%s, %v, %v)", compiled.Recipe.ID, ok, err)
	}
}
