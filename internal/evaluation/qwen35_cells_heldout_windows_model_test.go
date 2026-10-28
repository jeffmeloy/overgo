//go:build windows && modeltest

package evaluation

import (
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// TestQwen35CellsHoldHeldOut: Qwen3.8-27B answers the held-out parts its measurement run asks through the chat
// template, MMLU-Pro (its scorecard's quality) and IFEval, as well over its attention layers' int8 cells (cells-qwen35:
// a code and a 64-value group's fp16 scale where the F32 cache keeps the value) as over the F32 cache: of the cases the
// two paths answer differently, those only the F32 cache answers right do not outnumber those only the cells do past
// chance (cellsChance, the sign test gemma-4-12B's cells were held to). Each path answers into a store of its own.
func TestQwen35CellsHoldHeldOut(t *testing.T) {
	// Serial: both paths share the one card and the model is loaded alone.
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	// The model is read where the roots find it first, beside the store before the configured root.
	var path string
	for _, root := range roots.ModelRoots() {
		if candidate := filepath.Join(root, kQuantModel); path == "" {
			if _, err := os.Stat(candidate); err == nil {
				path = candidate
			}
		}
	}
	if path == "" {
		t.Skipf("Qwen3.8-27B's held-out check needs %s beneath one of %q", kQuantModel, roots.ModelRoots())
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	suites := map[string]CompiledSuite{}
	for _, family := range []string{"mmlu-pro", "ifeval"} {
		authorities := ExactAuthorities{
			ModelDefinition: planID(t, artifact.KindModelDefinition, "qwen3.8-27b"), RuntimeRecipe: planID(t, artifact.KindRecipe, "qwen3.8-27b"),
			CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "device"),
			Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
		}
		derived, _, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, family)
		if err != nil || len(derived) != 1 {
			t.Fatalf("the store derived %d suites for %s: %v", len(derived), family, err)
		}
		suites[family] = derived[0]
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	right := map[bool]map[string]map[string]bool{}
	for _, f32 := range []bool{false, true} {
		runner, err := modelcli.OpenRunner(t.Context(), roots.Store, path, inference.OpenOptions{F32Cache: f32})
		if err != nil {
			t.Fatal(err)
		}
		right[f32] = map[string]map[string]bool{}
		for family, suite := range suites {
			right[f32][family] = answerCells(t, suite, cellsRuntime{runner})
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for family := range suites {
		losses, wins := 0, 0
		for answered, f32Right := range right[true][family] {
			switch cellsRight := right[false][family][answered]; {
			case f32Right && !cellsRight:
				losses++
			case cellsRight && !f32Right:
				wins++
			}
		}
		chance := signTest(losses, wins)
		t.Logf("%s: the F32 cache alone answers %d right, the cells alone %d (p %.3g)", family, losses, wins, chance)
		if losses > wins && chance < cellsChance {
			t.Errorf("%s: the cells lose %d cases the F32 cache answers and win %d, a lean chance gives %.3g", family, losses, wins, chance)
		}
	}
}
