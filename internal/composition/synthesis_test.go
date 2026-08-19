package composition

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/organ"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/safetensors"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

const synthesisComponentTensor = "model.layers.0.mlp.gate_proj.weight"

func TestComponentProposalToPromotionPipeline(t *testing.T) {
	root := t.TempDir()
	targetDir, donorDir := filepath.Join(root, "target"), filepath.Join(root, "donor")
	writeSynthesisModel(t, targetDir, 3)
	writeSynthesisModel(t, donorDir, 5)
	target, err := identifyDenseModel(targetDir)
	if err != nil {
		t.Fatal(err)
	}
	donor, err := identifyDenseModel(donorDir)
	if err != nil {
		t.Fatal(err)
	}
	contract := organ.Classify(synthesisComponentTensor, "f32", "", "text", "")
	index, err := NewHypervectorIndex([]CatalogComponent{{Model: donor, Name: synthesisComponentTensor, Contract: contract}})
	if err != nil {
		t.Fatal(err)
	}
	hits, err := index.Search(CatalogComponent{Model: target, Name: synthesisComponentTensor, Contract: contract}, 1)
	if err != nil {
		t.Fatal(err)
	}
	ranker, err := TrainProposalRanker(nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := NewBridgeProposal(target, Candidates(hits, target), ranker,
		"go test ./internal/composition -run '^TestComponentProposalToPromotionPipeline$' -count=1",
		"bridge requires measured held-out evidence")
	if err != nil {
		t.Fatal(err)
	}
	store, err := repodb.Open(filepath.Join(root, "repodb"))
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	rankerContent, _ := ranker.Content()
	proposalContent, _ := proposal.Content()
	if _, err := store.Commit(context.Background(), artifact.Batch{
		Key: "fixture/component-proposal", Contents: []artifact.Content{rankerContent, proposalContent},
		Artifacts: []artifact.Descriptor{{ID: target}, {ID: donor}},
		Lineage:   append(ranker.Lineage(), proposal.Lineage()...),
	}); err != nil {
		t.Fatal(err)
	}
	derivation := testutil.ArtifactID(t, artifact.KindEvidence, "synthesis decider")
	if _, err := store.Commit(context.Background(), artifact.Batch{Key: "fixture/synthesis-decider",
		Artifacts: []artifact.Descriptor{{ID: derivation}}}); err != nil {
		t.Fatal(err)
	}
	outcome, err := SynthesizeBridge(store, proposal, Config{
		TargetDir: targetDir, DonorDir: donorDir, GraftLayer: -1, DonorLayer: -1,
		Seeds: []int64{7}, Steps: 1, BaseLR: 0.01, Momentum: 0.9,
		Train: [][]int{{1, 2, 3, 4}, {2, 3, 4, 1}}, HeldOut: []int{1, 3, 2, 4},
	}, recipe.Decider{CodeCommit: strings.Repeat("a", 40), Derivation: derivation})
	if err != nil {
		t.Fatal(err)
	}
	want := recipe.DecisionRefused
	if outcome.Result.Ship {
		want = recipe.DecisionAccepted
	}
	if outcome.Decision.Outcome != want || outcome.Record.TrainingPlan != outcome.Result.Recipe ||
		outcome.Record.Parents[0] != target || outcome.Result.DonorTensor != synthesisComponentTensor {
		t.Fatalf("outcome=%+v", outcome)
	}
	if _, found, err := store.Content(context.Background(), outcome.Result.Recipe); err != nil || !found {
		t.Fatalf("compiled recipe absent: found=%t err=%v", found, err)
	}
}

func TestComponentSynthesisLoadsOnlySelectedDonor(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "donor")
	writeSynthesisModel(t, directory, 11)
	component, err := loadDonorComponent(directory, -1, synthesisComponentTensor)
	if err != nil {
		t.Fatal(err)
	}
	retained := len(component.gate) + len(component.up) + len(component.down)
	source, err := safetensors.OpenSource(directory)
	if err != nil {
		t.Fatal(err)
	}
	defer source.Close()
	all := 0
	for _, tensor := range source.Tensors {
		all += int(tensor.Elements())
	}
	if component.gateName != synthesisComponentTensor || component.upName == "" || component.downName == "" ||
		len(component.gate) == 0 || len(component.up) == 0 || len(component.down) == 0 || retained >= all {
		t.Fatalf("selected donor = names %q/%q/%q geometry %dx%d retained %d of %d",
			component.gateName, component.upName, component.downName,
			component.hidden, component.intermediate, retained, all)
	}
}

func writeSynthesisModel(t *testing.T, directory string, seed int64) {
	t.Helper()
	weights, shapes := testutil.DenseCausalWeights(t, testutil.DenseCausalSpec{
		Vocab: 8, Hidden: 8, Heads: 2, HeadDim: 4, KVHeads: 1, Intermediate: 16, Layers: 1, Seed: seed,
	})
	if err := os.MkdirAll(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := safetensors.Save(filepath.Join(directory, trainingprogram.CheckpointWeights), weights, shapes, nil); err != nil {
		t.Fatal(err)
	}
	config := `{"model_type":"llama","num_attention_heads":2,"head_dim":4,"rope_theta":10000.0,"rms_norm_eps":1e-6,"tie_word_embeddings":true}`
	if err := os.WriteFile(filepath.Join(directory, "config.json"), []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
}
