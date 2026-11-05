package trainingworkflow

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/trainingprogram"
)

// TestAResumeRefusesAnAverageItsCheckpointDoesNotName saves an averaged run's checkpoint, which names the digest of
// each file of its average, and resumes from it as it is changed: a flipped byte in the average's weights refuses the
// resume, naming the file, and so does the average's state gone, where a resume once went on with no average and no
// error. The checkpoint as it was saved still resumes.
func TestAResumeRefusesAnAverageItsCheckpointDoesNotName(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	run := func(output string, request TextHeldoutRequest) error {
		request.Alias, request.ModelDirectory, request.OutputDirectory, request.Steps, request.Host, request.Average = text.alias, text.model, filepath.Join(text.root, output), 12, true, 0.9
		_, err := RunTextHeldout(t.Context(), text.store, request)
		return err
	}
	testcheck.NoError(t, run("saved", TextHeldoutRequest{CheckpointEvery: 4, CheckpointKeep: 3}))
	checkpoint := filepath.Join(text.root, "saved"+stepCheckpointInfix+"000004")
	weights, state := filepath.Join(checkpoint, averageDirectory, averageWeightsFile), filepath.Join(checkpoint, averageDirectory, averageStateFile)
	saved, err := os.ReadFile(weights)
	testcheck.NoError(t, err)
	flipped := append([]byte(nil), saved...)
	flipped[len(flipped)-1] ^= 0xff
	testcheck.NoError(t, os.WriteFile(weights, flipped, 0o600))
	if err := run("from-flipped", TextHeldoutRequest{Resume: checkpoint}); err == nil || !strings.Contains(err.Error(), averageDirectory+"/"+averageWeightsFile) {
		t.Fatalf("a resume from an average with a flipped byte answered %v, want a refusal naming its weights", err)
	}
	testcheck.NoError(t, os.WriteFile(weights, saved, 0o600))
	kept, err := os.ReadFile(state)
	testcheck.NoError(t, err)
	testcheck.NoError(t, os.Remove(state))
	if err := run("from-stateless", TextHeldoutRequest{Resume: checkpoint}); err == nil || !strings.Contains(err.Error(), averageDirectory+"/"+averageStateFile) {
		t.Fatalf("a resume from an average whose state is gone answered %v, want a refusal naming it", err)
	}
	testcheck.NoError(t, os.WriteFile(state, kept, 0o600))
	if err := run("from-saved", TextHeldoutRequest{Resume: checkpoint}); err != nil {
		t.Fatalf("a resume from the checkpoint as it was saved: %v", err)
	}
}
