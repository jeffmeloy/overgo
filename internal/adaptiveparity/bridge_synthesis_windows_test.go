//go:build windows

package adaptiveparity_test

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/composition"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
	"overgo/internal/testutil"
	"overgo/internal/trainingprogram"
)

// Bridge-synthesis protocol, written before the run (Probe Discipline): the
// pipeline row, not a tier re-ranking. Given a blocked proposal naming the
// Carbon-500M middle-MLP donor for the Qwen2.5-0.5B target, SynthesizeBridge
// derives the adapter geometry, trains the bridge with everything else frozen
// under a SHORT recorded protocol (2 seeds x 4 steps, lr-scale 0.1 -- pipeline
// validation budget, deliberately below the retired tier's re-ranking
// budgets), commits the generation record, and emits a typed decision bound
// to the proposal: promotion evidence on SHIP, a refusal with the measured
// reason otherwise. The adapter tier remains retired; a REFUSE here is the
// expected physics and proves the pipeline emits it faithfully.
const (
	synthesisWindow = 64
	synthesisSteps  = 4
)

// TestBridgeSynthesisProducesEvidenceOrRefusal passes when the pipeline ran
// to a committed decision -- either verdict -- and fails only when it could
// not execute.
func TestBridgeSynthesisProducesEvidenceOrRefusal(t *testing.T) {
	cudatest.RequireProbe(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	targetDir := filepath.Join(roots.Models, "Qwen2.5-0.5B")
	donorDir := filepath.Join(roots.Models, "Carbon-500M")
	for _, directory := range []string{targetDir, donorDir} {
		if _, err := os.Stat(directory); err != nil {
			t.Skipf("UNAVAILABLE: %s absent; bridge synthesis NOT verified", directory)
		}
	}
	tokens := qwenAdaptiveProfileTokens
	if len(tokens) < 3*synthesisWindow {
		t.Fatalf("recorded profile stream too short: %d tokens", len(tokens))
	}
	store, err := repodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	target := synthesisModelID(t, targetDir)
	donor := synthesisModelID(t, donorDir)
	ranker, err := composition.TrainProposalRanker(nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	proposal, err := composition.NewBridgeProposal(
		target,
		[]composition.BridgeCandidate{{Donor: donor, Component: "model.layers.14.mlp.gate_proj.weight", Distance: 0.25}},
		ranker,
		"go test ./internal/adaptiveparity -run '^TestBridgeSynthesisProducesEvidenceOrRefusal$' -count=1",
		"adapter tier retired; synthesis validates the pipeline, promotion requires the experiment plane",
	)
	if err != nil {
		t.Fatal(err)
	}
	proposalContent, err := proposal.Content()
	if err != nil {
		t.Fatal(err)
	}
	rankerContent, err := ranker.Content()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := store.Commit(t.Context(), artifact.Batch{
		Key:       "synthesis/fixture",
		Artifacts: []artifact.Descriptor{{ID: target}, {ID: donor}},
		Contents:  []artifact.Content{rankerContent, proposalContent},
		Lineage:   proposal.Lineage(),
	}); err != nil {
		t.Fatal(err)
	}
	decider := recipe.Decider{
		CodeCommit: "0123456789abcdef0123456789abcdef01234567",
		Derivation: testutil.ArtifactID(t, artifact.KindEvidence, "synthesis-authority"),
	}
	started := time.Now()
	outcome, err := composition.SynthesizeBridge(store, proposal, composition.Config{
		TargetDir: targetDir, DonorDir: donorDir,
		GraftLayer: -1, DonorLayer: -1,
		Seeds: []int64{7, 11}, Steps: synthesisSteps,
		BaseLR: 0, LRScale: 0.1, Momentum: 0.9,
		Train: [][]int{
			tokens[:synthesisWindow],
			tokens[synthesisWindow : 2*synthesisWindow],
		},
		HeldOut: tokens[2*synthesisWindow : 3*synthesisWindow],
	}, decider)
	if err != nil {
		t.Fatal(err)
	}
	verdict := "REFUSE"
	if outcome.Result.Ship {
		verdict = "SHIP"
	}
	if outcome.Result.Ship && outcome.Decision.Outcome != recipe.DecisionAccepted {
		t.Fatalf("ship without promotion evidence: %+v", outcome.Decision)
	}
	if !outcome.Result.Ship && outcome.Decision.Outcome != recipe.DecisionRefused {
		t.Fatalf("refusal without a refusal decision: %+v", outcome.Decision)
	}
	if outcome.Decision.Subject != proposal.ID || outcome.Decision.Decider != decider {
		t.Fatalf("decision binding = %+v, want the proposal subject and decider identity", outcome.Decision)
	}
	stored, ok, err := store.Content(t.Context(), outcome.Decision.ID)
	if err != nil || !ok {
		t.Fatalf("synthesis decision not durable: (%v, %v)", ok, err)
	}
	parsed, err := recipe.ParseDecision(stored.Data)
	if err != nil || parsed.ID != outcome.Decision.ID {
		t.Fatalf("decision roundtrip = (%+v, %v)", parsed, err)
	}
	t.Logf("bridge synthesis verdict: %s decision=%s record=%s", verdict, outcome.Decision.ID, outcome.Record.ID)
	t.Logf("reason: %s", outcome.Decision.Reason)
	t.Logf("budget: 2 seeds x %d steps at lr-scale 0.1, wall %s, host reference", synthesisSteps, time.Since(started))
}

func synthesisModelID(t *testing.T, directory string) artifact.ID {
	t.Helper()
	file, err := os.Open(filepath.Join(directory, trainingprogram.CheckpointWeights))
	if err != nil {
		t.Fatal(err)
	}
	id, _, err := artifact.Identify(artifact.KindModel, file)
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		t.Fatal(err)
	}
	return id
}
