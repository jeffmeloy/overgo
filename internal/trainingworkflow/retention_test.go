package trainingworkflow

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/peft"
	"overgo/internal/trainingprogram"
)

// stepCheckpoints lists the names of the checkpoints a run to output saved along
// the way, in order.
func stepCheckpoints(t *testing.T, output string) []string {
	t.Helper()
	entries, err := os.ReadDir(filepath.Dir(output))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, entry := range entries {
		if strings.HasPrefix(entry.Name(), filepath.Base(output)+stepCheckpointInfix) {
			names = append(names, entry.Name())
		}
	}
	slices.Sort(names)
	return names
}

// publishedWeights is the bytes of the weights a run published at output.
func publishedWeights(t *testing.T, output string) []byte {
	t.Helper()
	path, err := trainingprogram.CheckpointWeightsPath(output)
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// TestCheckpointRetention runs a text objective that saves a checkpoint every
// second update and keeps two, and checks what a run that saves checkpoints must:
// only the older ones are deleted and nothing else beside the output is; a run
// resumed from a kept checkpoint ends holding exactly the weights, and for an
// adapter exactly the factors, an uninterrupted run does, under a moving average
// that continues from the checkpoint's rather than starting over; and a
// checkpoint that follows the run's last update has nothing left to resume.
func TestCheckpointRetention(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	const steps = 10
	run := func(t *testing.T, name string, request TextHeldoutRequest) string {
		t.Helper()
		output := filepath.Join(text.root, name)
		request.Alias, request.ModelDirectory, request.OutputDirectory, request.Steps, request.Host = text.alias, text.model, output, steps, true
		if _, err := RunTextHeldout(ctx, text.store, request); err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		return output
	}

	t.Run("retention deletes only older checkpoints", func(t *testing.T) {
		output := filepath.Join(text.root, "kept")
		// Names beside the output that are not a checkpoint of this run, which retention must leave be.
		for _, bystander := range []string{"kept" + stepCheckpointInfix + "abc", "kept" + stepCheckpointInfix + "00000012", "kept-other", "other" + stepCheckpointInfix + "000002"} {
			if err := os.Mkdir(filepath.Join(text.root, bystander), 0o700); err != nil {
				t.Fatal(err)
			}
		}
		run(t, "kept", TextHeldoutRequest{CheckpointEvery: 2, CheckpointKeep: 2})
		var kept []string
		for _, name := range stepCheckpoints(t, output) {
			if _, err := trainingprogram.LoadCheckpoint(filepath.Join(text.root, name)); err == nil {
				kept = append(kept, strings.TrimPrefix(name, "kept"+stepCheckpointInfix))
			}
		}
		if !slices.Equal(kept, []string{"000006", "000008"}) {
			t.Fatalf("checkpoints kept %v, want the newest two of those after updates 2, 4, 6 and 8: 000006 and 000008", kept)
		}
		for _, bystander := range []string{"kept" + stepCheckpointInfix + "abc", "kept" + stepCheckpointInfix + "00000012", "kept-other", "other" + stepCheckpointInfix + "000002"} {
			if _, err := os.Stat(filepath.Join(text.root, bystander)); err != nil {
				t.Errorf("retention removed %s, which is no checkpoint of the run", bystander)
			}
		}
		if _, err := trainingprogram.LoadCheckpoint(output); err != nil {
			t.Errorf("the run's own checkpoint: %v", err)
		}
		if names := stepCheckpoints(t, output); slices.Contains(names, "kept"+stepCheckpointInfix+"000010") {
			t.Error("a checkpoint was saved after the last update, which is the run's own")
		}
	})

	t.Run("a full run resumes where it stopped, average and all", func(t *testing.T) {
		whole := run(t, "whole", TextHeldoutRequest{Average: 0.9})
		interrupted := run(t, "interrupted", TextHeldoutRequest{Average: 0.9, CheckpointEvery: 4, CheckpointKeep: 3})
		if !slices.Equal(publishedWeights(t, whole), publishedWeights(t, interrupted)) {
			t.Fatal("saving checkpoints changed the weights the run published")
		}
		for _, at := range []string{"000004", "000008"} {
			resumed := run(t, "resumed-"+at, TextHeldoutRequest{Average: 0.9, Resume: filepath.Join(text.root, "interrupted"+stepCheckpointInfix+at)})
			if !slices.Equal(publishedWeights(t, whole), publishedWeights(t, resumed)) {
				t.Errorf("a run resumed from update %s published other weights than the uninterrupted run", at)
			}
		}
		// A checkpoint of a run that kept no average gives the resumed run none to continue; and
		// the average a checkpoint holds is continued even when the resuming run does not name one.
		restarted := run(t, "resumed-no-flag", TextHeldoutRequest{Resume: filepath.Join(text.root, "interrupted"+stepCheckpointInfix+"000004")})
		if !slices.Equal(publishedWeights(t, whole), publishedWeights(t, restarted)) {
			t.Error("a run resumed from a checkpoint with an average dropped it")
		}
	})

	t.Run("an adapter run resumes where it stopped, average and all", func(t *testing.T) {
		lora := func() *LoRA { return &LoRA{Rank: 4, Alpha: 8, Seed: 1} }
		whole := run(t, "lora-whole", TextHeldoutRequest{Average: 0.9, LoRA: lora()})
		run(t, "lora-interrupted", TextHeldoutRequest{Average: 0.9, LoRA: lora(), CheckpointEvery: 4, CheckpointKeep: 3})
		resumed := run(t, "lora-resumed", TextHeldoutRequest{Average: 0.9, LoRA: lora(), Resume: filepath.Join(text.root, "lora-interrupted"+stepCheckpointInfix+"000004")})
		wholeAdapter, err := peft.Load(filepath.Join(whole, LoRAAdapterDirectory))
		if err != nil {
			t.Fatal(err)
		}
		resumedAdapter, err := peft.Load(filepath.Join(resumed, LoRAAdapterDirectory))
		if err != nil {
			t.Fatal(err)
		}
		for name, module := range wholeAdapter.Modules {
			other := resumedAdapter.Modules[name]
			if !slices.Equal(module.Down, other.Down) || !slices.Equal(module.Up, other.Up) {
				t.Fatalf("%s: the resumed adapter differs from the uninterrupted one", name)
			}
		}
		if !slices.Equal(publishedWeights(t, whole), publishedWeights(t, resumed)) {
			t.Error("the resumed adapter run published other weights than the uninterrupted one")
		}
	})

	t.Run("what cannot resume is refused", func(t *testing.T) {
		for name, request := range map[string]TextHeldoutRequest{
			"a finished run's own checkpoint": {Resume: filepath.Join(text.root, "whole")},
			"no checkpoint there":             {Resume: filepath.Join(text.root, "absent")},
			"a negative interval":             {CheckpointEvery: -1},
			"no number to keep":               {CheckpointEvery: 2},
		} {
			request.Alias, request.ModelDirectory, request.OutputDirectory, request.Steps, request.Host = text.alias, text.model, filepath.Join(text.root, "refused-"+strings.ReplaceAll(name, " ", "-")), steps, true
			if _, err := RunTextHeldout(ctx, text.store, request); err == nil {
				t.Errorf("%s was accepted", name)
			}
		}
	})
}
