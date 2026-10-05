package trainingworkflow

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/peft"
	"overgo/internal/trainingprogram"
)

// TestLoRAScalingVariantsTrainEndToEnd trains the same adapter, at the same rank
// and steps, plainly, rank-stabilized and with the up factor learning four times
// faster, and reads each held-out verdict: every run publishes one and an adapter
// that says how it was scaled, and a run that continues an adapter, which has its
// own scaling, is refused the options. The accuracies are logged: on this tiny
// fixture they are one run each of one seed, evidence of nothing about a model.
func TestLoRAScalingVariantsTrainEndToEnd(t *testing.T) {
	text := registerTextObjective(t, averageDocuments(), trainingprogram.MetricTokenAccuracy)
	ctx := t.Context()
	for _, variant := range []struct {
		name  string
		lora  LoRA
		rs    bool
		alpha float64
	}{
		{"plain", LoRA{Rank: 4, Alpha: 8, Seed: 1}, false, 8},
		{"rank-stabilized", LoRA{Rank: 4, Alpha: 8, Seed: 1, RSLoRA: true}, true, 8},
		{"LoRA+ at four", LoRA{Rank: 4, Alpha: 8, Seed: 1, UpRate: 4}, false, 32},
	} {
		output := filepath.Join(text.root, strings.ReplaceAll(variant.name, " ", "-"))
		lora := variant.lora
		result, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
			Alias: text.alias, ModelDirectory: text.model, OutputDirectory: output, Steps: 48, Host: true, LoRA: &lora,
		})
		if err != nil || !result.Verdict.Valid() {
			t.Fatalf("%s: run = (%+v, %v), want a verdict", variant.name, result, err)
		}
		var verdict struct {
			BaselineValue  float64 `json:"baseline_value"`
			CandidateValue float64 `json:"candidate_value"`
		}
		content, found, err := artifact.ReadContent(ctx, text.store, result.Verdict)
		if err != nil || !found || json.Unmarshal(content.Data, &verdict) != nil {
			t.Fatalf("%s: verdict content = (found %t, %v)", variant.name, found, err)
		}
		t.Logf("%s: held-out token accuracy %.4f -> %.4f", variant.name, verdict.BaselineValue, verdict.CandidateValue)
		adapter, err := peft.Load(filepath.Join(output, LoRAAdapterDirectory))
		if err != nil {
			t.Fatal(err)
		}
		if adapter.Config.RSLoRA != variant.rs || adapter.Config.Alpha != variant.alpha {
			t.Errorf("%s: adapter is rslora=%t at alpha %v, want rslora=%t at alpha %v", variant.name, adapter.Config.RSLoRA, adapter.Config.Alpha, variant.rs, variant.alpha)
		}
	}
	for name, lora := range map[string]LoRA{
		"rslora beside an adapter": {From: filepath.Join(text.root, "plain", LoRAAdapterDirectory), RSLoRA: true},
		"LoRA+ beside an adapter":  {From: filepath.Join(text.root, "plain", LoRAAdapterDirectory), UpRate: 4},
	} {
		if _, err := RunTextHeldout(ctx, text.store, TextHeldoutRequest{
			Alias: text.alias, ModelDirectory: text.model, OutputDirectory: filepath.Join(text.root, "refused-"+strings.ReplaceAll(name, " ", "-")), Steps: 2, Host: true, LoRA: &lora,
		}); err == nil || !strings.Contains(err.Error(), "the adapter's own rank, alpha, targets and scaling") {
			t.Errorf("%s: err = %v, want the refusal naming the adapter's own scaling", name, err)
		}
	}
}
