package adaptiveparity

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
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

func TestInferenceTextAndStructuredEvidenceIdentity(t *testing.T) {
	if testing.Short() {
		t.Skip(testskip.ShortIntegration)
	}
	if os.Getenv("OVERGO_ADAPTIVE_PARITY") != "1" {
		t.Skip("set OVERGO_ADAPTIVE_PARITY=1 to verify external evidence identities")
	}
	snapshot := textStructuredSnapshot(t)
	store, err := overgodb.Open(filepath.Join(testutil.RepoRoot(t), "overgodb-store"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	goldenPaths := map[string]string{
		"dense-carbon": "carbon_serving_golden.json", "dense-minicpm5": "minicpm5_serving_golden.json",
		"dense-qwen25": "qwen25_serving_golden.json", "hybrid-qwen35": "qwen35_4b_serving_golden.json",
		"forecast-timesfm": "timesfm_golden.json", "tabular-classification": "tabfm_predict_golden.json",
		"tabular-regression": "tabfm_predict_golden.json", "seq2seq-needle": "needle_forward_oracle.json",
	}
	for _, capability := range snapshot.Capabilities {
		if capability.State == PromotionRefused {
			continue
		}
		if len(capability.Artifacts) != 1 || len(capability.Corpora) != 1 || len(capability.Goldens) != 1 {
			t.Fatalf("row %q evidence cardinality is not singular", capability.ID)
		}
		artifactPath, err := artifact.AvailablePath(t.Context(), store, capability.Artifacts[0].Identity, artifact.LocationFile)
		if err != nil {
			t.Fatal(err)
		}
		assertFileIdentity(t, artifactPath, capability.Artifacts[0].Identity)
		golden := testutil.FixturePath(t, goldenPaths[capability.ID])
		assertFileIdentity(t, golden, capability.Corpora[0].Identity)
		assertFileIdentity(t, golden, capability.Goldens[0].Identity)
	}
}

func textStructuredSnapshot(t *testing.T) Snapshot {
	t.Helper()
	raw, err := os.ReadFile(testutil.FixturePath(t, "adaptive_text_structured_snapshot.json"))
	if err != nil {
		t.Fatalf("UNAVAILABLE: text/structured snapshot absent; parity NOT verified: %v", err)
	}
	snapshot, _, err := Normalize(raw)
	if err != nil {
		t.Fatal(err)
	}
	return snapshot
}

func assertFileIdentity(t *testing.T, path string, want artifact.ID) {
	t.Helper()
	file, err := os.Open(path)
	if err != nil {
		t.Fatalf("UNAVAILABLE: evidence file %s absent; parity NOT verified: %v", path, err)
	}
	got, _, identifyErr := artifact.Identify(want.Kind(), file)
	closeErr := file.Close()
	if identifyErr != nil {
		t.Fatal(identifyErr)
	}
	if closeErr != nil {
		t.Fatal(closeErr)
	}
	if got != want {
		t.Fatalf("evidence file %s identity = %s, want %s", path, got, want)
	}
}

func TestInferenceModalityMatrixDerivedFromRecipes(t *testing.T) {
	model := identify(t, artifact.KindModel, "modality-model")
	profile := identify(t, artifact.KindProfile, "image-profile")
	definitions := make([]recipe.Definition, 0, 7)
	for _, task := range []recipe.Task{
		recipe.TaskForecast, recipe.TaskTabular, recipe.TaskSeq2Seq,
		recipe.TaskSpeech, recipe.TaskVQA,
	} {
		definition, err := modelrecipe.CapabilityDefinition(task, model)
		if err != nil {
			t.Fatal(err)
		}
		definitions = append(definitions, definition)
	}
	oscillatorImage, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleOscillatorImagePrepare, model, artifact.ID{})
	if err != nil {
		t.Fatal(err)
	}
	definitions = append(definitions, oscillatorImage)
	latentImage, err := modelrecipe.GenerationDefinition(modelrecipe.ModuleLatentImagePrepare, model, profile)
	if err != nil {
		t.Fatal(err)
	}
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
	rows, err := InferenceModalityMatrix(definitions, required)
	if err != nil {
		t.Fatal(err)
	}
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
