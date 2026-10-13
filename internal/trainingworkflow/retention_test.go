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

// TestStepCheckpointsIgnoreAnotherRunsDirectories holds a run's saves to the run: a
// directory an earlier run left beside the output, numbered above or below the
// run's own saves, neither outranks them in retention nor is deleted by it; a
// name the run would save to that is already taken refuses the run before it
// trains, naming it; an output written with a trailing separator saves beside
// itself, not inside; and a run resumed from a step that is not the newest
// continues once the later steps it would write over are gone, leaving the step
// it resumed from.
func TestStepCheckpointsIgnoreAnotherRunsDirectories(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	const steps = 10
	run := func(output string, request TextHeldoutRequest) error {
		request.Alias, request.ModelDirectory, request.OutputDirectory, request.Steps, request.Host = text.alias, text.model, output, steps, true
		_, err := RunTextHeldout(t.Context(), text.store, request)
		return err
	}
	exists := func(path string) bool {
		_, err := os.Stat(path)
		return err == nil
	}
	stale := func(output string, numbers ...string) []string {
		var made []string
		for _, number := range numbers {
			path := output + stepCheckpointInfix + number
			if err := os.Mkdir(path, 0o700); err != nil {
				t.Fatal(err)
			}
			made = append(made, path)
		}
		return made
	}

	t.Run("a stale directory neither outranks the run's saves nor is deleted by them", func(t *testing.T) {
		output := filepath.Join(text.root, "fresh")
		older := stale(output, "000200", "000700")
		if err := run(output, TextHeldoutRequest{CheckpointEvery: 2, CheckpointKeep: 1}); err != nil {
			t.Fatal(err)
		}
		for _, path := range older {
			if !exists(path) {
				t.Errorf("the run deleted %s, which an earlier run left", path)
			}
		}
		if got, want := stepCheckpoints(t, output), []string{"fresh-step-000008", "fresh-step-000200", "fresh-step-000700"}; !slices.Equal(got, want) {
			t.Fatalf("step directories after the run %v, want its newest save beside the stale two %v", got, want)
		}
	})

	t.Run("a name the run would save to that is taken refuses it before it trains", func(t *testing.T) {
		output := filepath.Join(text.root, "taken")
		taken := stale(output, "000004")
		err := run(output, TextHeldoutRequest{CheckpointEvery: 2, CheckpointKeep: 2})
		if err == nil || !strings.Contains(err.Error(), taken[0]) {
			t.Fatalf("the run was %v, want it refused naming %s", err, taken[0])
		}
		if exists(output) || !exists(taken[0]) || len(stepCheckpoints(t, output)) != 1 {
			t.Fatalf("a refused run left %v, want only the directory that was there", stepCheckpoints(t, output))
		}
	})

	t.Run("an output with a trailing separator saves beside itself", func(t *testing.T) {
		output := filepath.Join(text.root, "slashed")
		if err := run(output+string(filepath.Separator), TextHeldoutRequest{CheckpointEvery: 4, CheckpointKeep: 2}); err != nil {
			t.Fatal(err)
		}
		if got, want := stepCheckpoints(t, output), []string{"slashed-step-000004", "slashed-step-000008"}; !slices.Equal(got, want) {
			t.Fatalf("step directories %v, want %v beside the output", got, want)
		}
		if _, err := trainingprogram.LoadCheckpoint(output); err != nil {
			t.Fatal(err)
		}
		if exists(filepath.Join(output, "-step-000004")) {
			t.Error("a checkpoint was saved inside the output")
		}
	})

	t.Run("a run resumed from a step that is not the newest continues once the later steps are gone", func(t *testing.T) {
		output := filepath.Join(text.root, "resumable")
		if err := run(output, TextHeldoutRequest{CheckpointEvery: 2, CheckpointKeep: 4}); err != nil {
			t.Fatal(err)
		}
		if err := os.RemoveAll(output); err != nil {
			t.Fatal(err)
		}
		from := output + stepCheckpointInfix + "000004"
		err := run(output, TextHeldoutRequest{CheckpointEvery: 2, CheckpointKeep: 1, Resume: from})
		if err == nil || !strings.Contains(err.Error(), output+stepCheckpointInfix+"000006") {
			t.Fatalf("the resumed run was %v, want it refused naming the first later step", err)
		}
		for _, later := range []string{"000006", "000008"} {
			if err := os.RemoveAll(output + stepCheckpointInfix + later); err != nil {
				t.Fatal(err)
			}
		}
		if err := run(output, TextHeldoutRequest{CheckpointEvery: 2, CheckpointKeep: 1, Resume: from}); err != nil {
			t.Fatal(err)
		}
		if got, want := stepCheckpoints(t, output), []string{"resumable-step-000002", "resumable-step-000004", "resumable-step-000008"}; !slices.Equal(got, want) {
			t.Fatalf("step directories %v, want the step resumed from, the earlier run's older one and the resumed run's newest save %v", got, want)
		}
	})
}

// TestCheckpointRetention runs a text objective that saves a checkpoint every
// second update and keeps two, and checks what a run that saves checkpoints must:
// only the older ones are deleted and nothing else beside the output is; a run
// resumed from a kept checkpoint ends holding exactly the weights, and for an
// adapter exactly the factors, an uninterrupted run does, under a moving average
// that continues from the checkpoint's rather than starting over; a
// checkpoint that follows the run's last update has nothing left to resume;
// and a proof keeps the checkpoints it saves along the way, which a stopped
// run resumes from, though it keeps no file of its own checkpoint.
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
		var progress strings.Builder
		restarted := run(t, "resumed-no-flag", TextHeldoutRequest{Resume: filepath.Join(text.root, "interrupted"+stepCheckpointInfix+"000004"), Progress: &progress})
		if !slices.Equal(publishedWeights(t, whole), publishedWeights(t, restarted)) {
			t.Error("a run resumed from a checkpoint with an average dropped it")
		}
		if !strings.Contains(progress.String(), "resuming continues the checkpoint's moving average, at decay 0.9 after 4 updates, not the decay 0 this run asked for") {
			t.Errorf("a resume that overrides the average it asked for did not say so; progress was %q", progress.String())
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

	t.Run("a proof keeps what it saves along the way, and nothing at its output", func(t *testing.T) {
		// A checkpoint saved along the way is what a stopped run resumes from.
		// It is not the checkpoint a proof's claim names, so it stays, as the
		// run was asked; the proof's own checkpoint leaves no file.
		output := run(t, "proved", TextHeldoutRequest{Proof: true, CheckpointEvery: 4, CheckpointKeep: 1})
		if _, err := trainingprogram.LoadCheckpoint(stepCheckpointDirectory(output, 8)); err != nil {
			t.Errorf("the checkpoint the proof saved after update 8: %v", err)
		}
		if _, err := os.Stat(output); !os.IsNotExist(err) {
			t.Errorf("the proof left its output at %s (%v), want no file of its own checkpoint kept", output, err)
		}
	})
}
