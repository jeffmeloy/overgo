package trainingworkflow

import (
	"encoding/json"
	"math"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/peft"
	"overgo/internal/safetensors"
	"overgo/internal/trainingprogram"
)

// TestTextObjectiveTrainsALoRAAdapter trains a tiny dense model's LoRA adapter
// under a token-prediction objective and judges the adapted model on the
// held-out records: the verdict passes, the checkpoint's parameters are the
// adapter's factors and not the model's, the adapter is in the checkpoint as a
// PEFT directory, and merging that adapter into the base checkpoint gives
// exactly the weights the checkpoint holds, so the file means what was judged.
func TestTextObjectiveTrainsALoRAAdapter(t *testing.T) {
	var documents []string
	for index := range 12 {
		documents = append(documents, strings.Repeat("abcd", 5)[:6+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	output := filepath.Join(text.root, "trained")
	var steps int
	result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: output, Steps: 48, Host: true,
		LoRA: &LoRA{Rank: 4, Alpha: 8, Seed: 1}, Step: func(int, int, float64) { steps++ },
	})
	if err != nil || steps != 48 {
		t.Fatalf("LoRA run = (%+v, %v) after %d reported steps, want a run of forty-eight", result, err, steps)
	}
	var verdict struct {
		BaselineValue  float64 `json:"baseline_value"`
		CandidateValue float64 `json:"candidate_value"`
	}
	content, found, err := artifact.ReadContent(ctx, text.store, result.Verdict)
	if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil {
		t.Fatalf("verdict %s content = (found %t, %v)", result.Verdict, found, err)
	}
	t.Logf("held-out token accuracy %.4f -> %.4f (passed %t)", verdict.BaselineValue, verdict.CandidateValue, result.Passed)
	if !result.Passed {
		t.Fatalf("the adapted model did not beat the base on held-out records: %.4f -> %.4f", verdict.BaselineValue, verdict.CandidateValue)
	}

	checkpoint, err := trainingprogram.LoadCheckpoint(output)
	if err != nil {
		t.Fatal(err)
	}
	adapter, err := peft.Load(filepath.Join(output, LoRAAdapterDirectory))
	if err != nil {
		t.Fatal(err)
	}
	factors := 0
	for _, module := range adapter.Modules {
		factors += module.Rank * (module.In + module.Out)
	}
	if len(adapter.Modules) != 7 || checkpoint.ParameterCount != factors || adapter.Config.BaseModel != "model" || adapter.Config.Rank != 4 {
		t.Fatalf("adapter has %d modules, %d factor parameters against the checkpoint's %d, base %q rank %d; want seven modules and the factors as the parameters",
			len(adapter.Modules), factors, checkpoint.ParameterCount, adapter.Config.BaseModel, adapter.Config.Rank)
	}
	merged := filepath.Join(t.TempDir(), "merged")
	if _, err := peft.Merge(adapter, text.model, merged); err != nil {
		t.Fatal(err)
	}
	fromAdapter, err := safetensors.OpenSource(merged)
	if err != nil {
		t.Fatal(err)
	}
	defer fromAdapter.Close()
	fromCheckpoint, err := safetensors.OpenSource(output)
	if err != nil {
		t.Fatal(err)
	}
	defer fromCheckpoint.Close()
	for name, tensor := range fromCheckpoint.Tensors {
		got, err := safetensors.ReadF32(fromAdapter.Tensors[name])
		if err != nil {
			t.Fatal(err)
		}
		want, err := safetensors.ReadF32(tensor)
		if err != nil {
			t.Fatal(err)
		}
		for index := range want {
			if math.Abs(float64(got[index]-want[index])) > 1e-5 {
				t.Fatalf("%s[%d]: merged adapter %v, checkpoint %v", name, index, got[index], want[index])
			}
		}
	}
}

// TestLoRARunRefusesWhatItCannotRun: a LoRA run trains on the host and has no
// lexical table to freeze or device to train on.
func TestLoRARunRefusesWhatItCannotRun(t *testing.T) {
	var documents []string
	for index := range 12 {
		documents = append(documents, strings.Repeat("abcd", 5)[:6+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	for name, request := range map[string]TextHeldoutRequest{
		"device":         {Host: false, LoRA: &LoRA{Rank: 2, Alpha: 4}},
		"frozen lexical": {Host: false, FreezeLexical: true, LoRA: &LoRA{Rank: 2, Alpha: 4}},
	} {
		request.Alias, request.ModelDirectory, request.OutputDirectory, request.Steps = text.alias, text.model, filepath.Join(text.root, "refused-"+strings.ReplaceAll(name, " ", "-")), 2
		if _, err := RunTextHeldout(t.Context(), text.store, request); err == nil || !strings.Contains(err.Error(), "LoRA run trains on the host") {
			t.Errorf("%s: err = %v, want the host-only refusal", name, err)
		}
	}
	if _, err := RunTextHeldout(t.Context(), text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "too-wide"), Steps: 2, Host: true, LoRA: &LoRA{Rank: 99, Alpha: 4},
	}); err == nil || !strings.Contains(err.Error(), "exceeds") {
		t.Errorf("a rank wider than the projections: err = %v", err)
	}
}

// TestContinuedAdapterIsJudgedAgainstTheTrueBase holds the verdict of a run that continues an
// adapter to its baseline: the base model as it was before any adapter, which a first run of the
// same objective measured, and not the model with the prior adapter applied, which would read the
// continuation as base to trained and the prior adapter's gain as nothing the run did.
func TestContinuedAdapterIsJudgedAgainstTheTrueBase(t *testing.T) {
	var documents []string
	for index := range 12 {
		documents = append(documents, strings.Repeat("abcd", 5)[:6+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	values := func(result HeldoutResult) (baseline, candidate float64) {
		t.Helper()
		var verdict struct {
			BaselineValue  float64 `json:"baseline_value"`
			CandidateValue float64 `json:"candidate_value"`
		}
		content, found, err := artifact.ReadContent(ctx, text.store, result.Verdict)
		if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil {
			t.Fatalf("verdict content = (found %t, %v)", found, err)
		}
		return verdict.BaselineValue, verdict.CandidateValue
	}
	first := filepath.Join(text.root, "first")
	firstResult, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: first, Steps: 16, Host: true, LoRA: &LoRA{Rank: 4, Alpha: 8, Seed: 1},
	})
	if err != nil {
		t.Fatal(err)
	}
	base, trained := values(firstResult)
	if base == trained {
		t.Fatalf("the first run's adapter left the held-out accuracy at %v: the fixture cannot tell a base baseline from an adapted one", base)
	}
	secondResult, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "second"), Steps: 4, Host: true,
		LoRA: &LoRA{From: filepath.Join(first, LoRAAdapterDirectory)},
	})
	if err != nil {
		t.Fatal(err)
	}
	if again, _ := values(secondResult); again != base {
		t.Fatalf("the continued run's baseline is %v, want the true base's %v, not the first adapter's %v", again, base, trained)
	}
}

// TestLoRARunContinuesAnAdapterFromPEFT: a run given the PEFT directory a
// previous run left starts from that adapter's factors, trains them further and
// leaves its own adapter in its own checkpoint; naming a rank beside the adapter
// is refused, since the adapter has its own.
func TestLoRARunContinuesAnAdapterFromPEFT(t *testing.T) {
	var documents []string
	for index := range 12 {
		documents = append(documents, strings.Repeat("abcd", 5)[:6+index])
	}
	text := registerTextObjective(t, documents, trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	first := filepath.Join(text.root, "first")
	if _, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: first, Steps: 8, Host: true, LoRA: &LoRA{Rank: 4, Alpha: 8, Seed: 1},
	}); err != nil {
		t.Fatal(err)
	}
	started, err := peft.Load(filepath.Join(first, LoRAAdapterDirectory))
	if err != nil {
		t.Fatal(err)
	}
	second := filepath.Join(text.root, "second")
	if _, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: second, Steps: 8, Host: true, LoRA: &LoRA{From: filepath.Join(first, LoRAAdapterDirectory)},
	}); err != nil {
		t.Fatal(err)
	}
	continued, err := peft.Load(filepath.Join(second, LoRAAdapterDirectory))
	if err != nil {
		t.Fatal(err)
	}
	if len(continued.Modules) != len(started.Modules) || continued.Config.Rank != started.Config.Rank || continued.Config.Alpha != started.Config.Alpha {
		t.Fatalf("continued adapter = %+v with %d modules, want the first run's %+v with %d", continued.Config, len(continued.Modules), started.Config, len(started.Modules))
	}
	moved := false
	for name, module := range continued.Modules {
		moved = moved || !slices.Equal(module.Up, started.Modules[name].Up)
	}
	if !moved {
		t.Fatal("the continued run left every factor as it found it")
	}
	if _, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
		Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "third"), Steps: 2, Host: true,
		LoRA: &LoRA{From: filepath.Join(first, LoRAAdapterDirectory), Rank: 2},
	}); err == nil || !strings.Contains(err.Error(), "the adapter's own rank") {
		t.Errorf("a rank beside an adapter: err = %v", err)
	}
}
