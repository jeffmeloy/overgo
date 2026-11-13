package adaptiveparity

import (
	"os"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/recipe"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestInferenceTextAndStructuredMatrix(t *testing.T) {
	snapshot := textStructuredSnapshot(t)
	wantStates := map[string]PromotionState{
		"dense-carbon": PromotionParity, "dense-minicpm5": PromotionWallLead,
		"dense-qwen25": PromotionParity, "hybrid-qwen35": PromotionTradeoff,
		"forecast-timesfm":       PromotionWallLead,
		"tabular-classification": PromotionWallLead, "tabular-regression": PromotionWallLead,
		"seq2seq-needle": PromotionWallLead,
		"embedding":      PromotionRefused, "rerank": PromotionRefused,
	}
	if len(snapshot.Capabilities) != len(wantStates) {
		t.Fatalf("text/structured rows = %d, want %d", len(snapshot.Capabilities), len(wantStates))
	}
	for _, capability := range snapshot.Capabilities {
		want, ok := wantStates[capability.ID]
		if !ok || capability.State != want {
			t.Fatalf("row %q state = %q, want %q", capability.ID, capability.State, want)
		}
		delete(wantStates, capability.ID)
	}
	if len(wantStates) != 0 {
		t.Fatalf("missing text/structured rows: %v", wantStates)
	}
}

func textStructuredSnapshot(t *testing.T) Snapshot {
	t.Helper()
	raw, err := os.ReadFile(testutil.FixturePath(t, "adaptive_text_structured_snapshot.json"))
	if err != nil {
		t.Fatalf("UNAVAILABLE: text/structured snapshot absent; parity NOT verified: %v", err)
	}
	snapshot, _, err := Normalize(raw)
	testcheck.NoError(t, err)
	return snapshot
}

func TestInferenceModalityMatrixDerivedFromRecipes(t *testing.T) {
	model := testutil.ArtifactID(t, artifact.KindModel, "modality-model")
	profile := testutil.ArtifactID(t, artifact.KindProfile, "image-profile")
	definitions := make([]recipe.Definition, 0, 7)
	for _, task := range []recipe.Task{
		recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskVQA,
	} {
		definition := testcheck.Must(modelrecipe.CapabilityDefinition(task, model))(t)
		definitions = append(definitions, definition)
	}
	oscillatorImage := testcheck.Must(modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, model, artifact.ID{}))(t)
	definitions = append(definitions, oscillatorImage)
	latentImage := testcheck.Must(modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentImagePrepare, model, profile))(t)
	definitions = append(definitions, latentImage)

	required := []Contract{
		{Task: recipe.TaskForecast, Signature: Signature{Inputs: []Modality{ModalityTimeSeries}, Outputs: []Modality{ModalityTimeSeries}}},
		{Task: recipe.TaskTabular, Signature: Signature{Inputs: []Modality{ModalityTable}, Outputs: []Modality{ModalityTable}}},
		{Task: recipe.TaskSeq2Seq, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityText}}},
		{Task: recipe.TaskSpeech, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityAudio}}},
		{Task: recipe.TaskImageGen, Signature: Signature{Inputs: []Modality{ModalityTable}, Outputs: []Modality{ModalityImage}}},
		{Task: recipe.TaskImageGen, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityImage}}},
		{Task: recipe.TaskVQA, Signature: Signature{Inputs: []Modality{ModalityImage, ModalityText}, Outputs: []Modality{ModalityText}}},
		{Task: recipe.TaskVideoGen, Signature: Signature{Inputs: []Modality{ModalityText}, Outputs: []Modality{ModalityVideo}}},
	}
	rows := testcheck.Must(InferenceModalityMatrix(definitions, required))(t)
	if len(rows) != len(required) {
		t.Fatalf("matrix rows = %d, want %d", len(rows), len(required))
	}
	refused := 0
	for _, row := range rows {
		if row.Status == ContractRefused {
			refused++
			if row.Task != recipe.TaskVideoGen || row.Recipe.Valid() || row.Reason == "" {
				t.Fatalf("invalid refusal row: %+v", row)
			}
			continue
		}
		if row.Status != ContractSupported || !row.Recipe.Valid() {
			t.Fatalf("invalid supported row: %+v", row)
		}
	}
	if refused != 1 {
		t.Fatalf("refused rows = %d, want 1", refused)
	}
}
