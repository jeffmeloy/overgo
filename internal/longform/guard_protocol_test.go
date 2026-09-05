package longform

import (
	"encoding/json"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/tokenizer"
)

func TestGuardGenerationProtocol(t *testing.T) {
	eog := []tokenizer.TokenID{2, 3}
	logits := []float32{1, 2, 4, 3}
	raw, err := generationOptions(RawContinuation, DeclaredFloors().OutputTokens)
	if err != nil {
		t.Fatal(err)
	}
	if token, err := raw.Sampler.Sample(logits); err != nil || token != int(eog[0]) {
		t.Fatalf("natural EOS changed: token=%d err=%v", token, err)
	}
	guard, err := generationOptions(GuardContinuation, DeclaredFloors().OutputTokens)
	if err != nil {
		t.Fatal(err)
	}
	if token, err := guard.Sampler.Sample(logits); err != nil || token != int(eog[0]) {
		t.Fatalf("guard changed greedy selection: token=%d err=%v", token, err)
	}
	if !raw.Sampler.IsRawGreedy() || !guard.Sampler.IsRawGreedy() || !guard.DeviceGreedy || !raw.DeviceGreedy ||
		raw.ContinueAfterEOG || !guard.ContinueAfterEOG || guard.MaxNewTokens != DeclaredFloors().OutputTokens {
		t.Fatal("guard changed the sampler/device path or failed to bind the full budget")
	}
	if _, err := generationOptions("unknown", DeclaredFloors().OutputTokens); err == nil {
		t.Fatal("unknown protocol accepted")
	}
	model, _, err := artifact.Identify(artifact.KindModel, strings.NewReader("guard protocol fixture"))
	if err != nil {
		t.Fatal(err)
	}
	rawInputs := BindInputs(model, "same corpus", eog, RawContinuation)
	guardInputs := BindInputs(model, "same corpus", eog, GuardContinuation)
	if !rawInputs.valid() || !guardInputs.valid() || rawInputs == guardInputs ||
		rawInputs.CorpusDigest != guardInputs.CorpusDigest || rawInputs.TokenDigest != guardInputs.TokenDigest {
		t.Fatal("protocol failed to distinguish otherwise identical inputs")
	}
	data, err := json.Marshal(guardInputs)
	if err != nil {
		t.Fatal(err)
	}
	var decoded Inputs
	if err := json.Unmarshal(data, &decoded); err != nil || decoded != guardInputs {
		t.Fatalf("protocol identity changed in storage: %v", err)
	}
	for _, protocol := range []Protocol{"", "unknown"} {
		decoded.Protocol = protocol
		if decoded.valid() {
			t.Fatal("unbound stored protocol accepted")
		}
	}
	reference := Result{Inputs: rawInputs, Floors: DeclaredFloors()}
	fresh := reference
	fresh.Inputs = guardInputs
	if verdict := Compare(reference, fresh, reference.Floors, reference.Floors.CheckRungCeiling); verdict.Passed ||
		len(verdict.Reasons) != 1 || !strings.Contains(verdict.Reasons[0], "comparison inputs differ") {
		t.Fatalf("different generation protocols compared as identical: %v", verdict)
	}
}
