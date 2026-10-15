package main

import (
	"context"
	"os"
	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/parity"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
	"overgo/internal/vqaserve"
	"path/filepath"
	"slices"
	"testing"
)

// TestParityCommandsRequireFreshExecution holds both parity commands' shared
// executor to producing current tokens and stage measurements on every invocation.
func TestParityCommandsRequireFreshExecution(t *testing.T) {
	// Serial: this test replaces the package's prepare and answer callbacks.
	oldPrepare, oldAnswer := prepareVQA, answerVQA
	t.Cleanup(func() { prepareVQA, answerVQA = oldPrepare, oldAnswer })
	prepared, answered := 0, 0
	prepareVQA = func(_ string, image []byte, question string) (vqaserve.Prepared, error) {
		if string(image) != "image" || question != "where?" {
			t.Fatalf("unexpected input %q %q", image, question)
		}
		prepared++
		return vqaserve.Prepared{}, nil
	}
	answerVQA = func(_ context.Context, _ string, _ vqaserve.Prepared, _ int, _ func(string)) (vqaserve.Result, string, error) {
		answered++
		return vqaserve.Result{Generated: []int{answered}}, "there", nil
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	model := testutil.ArtifactID(t, artifact.KindModel, "vqa-model")
	testutil.PublishArtifact(t, store, model)
	definition, err := modelrecipe.CapabilityDefinition(recipe.TaskVQA, model)
	if err != nil {
		t.Fatal(err)
	}
	program, err := modelrecipe.CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	directory := t.TempDir()
	image := filepath.Join(directory, "image")
	if err := os.WriteFile(image, []byte("image"), 0600); err != nil {
		t.Fatal(err)
	}
	campaign := &campaignContext{Campaign: parity.NewCampaign(filepath.Join(directory, "parity.log"))}
	for _, command := range []struct {
		name string
		tags []string
	}{
		{"runRecipeServe", []string{"RUN1", "RUN2"}}, {"verifyActivateVQA", []string{"VERIFY", "ACTIVE"}},
	} {
		t.Run(command.name, func(t *testing.T) {
			for range 2 {
				for _, tag := range command.tags {
					before := answered
					got, err := executeVQAProgram(campaign, t.Context(), store, model, program, image, "where?", tag)
					if err != nil || prepared != before+1 || answered != before+1 || got.text != "there" || !slices.Equal(got.chain, []int{before + 1}) || len(got.walls) != 2 {
						t.Fatalf("%s: prepares=%d answers=%d result=%+v error=%v; want fresh callbacks, current tokens and both stage measurements", tag, prepared, answered, got, err)
					}
				}
			}
		})
	}
}
