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

// kQuantModel is Qwen3.8-27B's UD Q6_K file beneath a models root, as its scorecard row serves it: its Q6_K, Q5_K and
// Q4_K projections of one to eight columns, a decode token's, a verify window's and an option's continuation, take
// Strata's native MMVQ (executor.kQuantProducts), q8_1 activations with an fp16 block scale where the q8-input span
// kernels keep an F32 one.
var kQuantModel = filepath.Join("Qwen3.8-27B-UD-Q6_K", "Qwen3.8-27B-UD-Q6_K.gguf")

// TestKQuantProductsHoldHeldOut: Qwen3.8-27B answers the held-out parts its measurement run asks through the chat
// template, MMLU-Pro (its scorecard's quality) and IFEval, as well through the native MMVQ as through the span kernels:
// of the cases the two paths answer differently, those only the span kernels answer right do not outnumber those only
// the native products do past chance (cellsChance, the two-sided sign test the int8 cells were held to). Each path
// answers into a store of its own, so neither reads the other's answers.
func TestKQuantProductsHoldHeldOut(t *testing.T) {
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
	for _, span := range []bool{false, true} {
		runner, err := modelcli.OpenRunner(t.Context(), roots.Store, path, inference.OpenOptions{SpanKernels: span})
		if err != nil {
			t.Fatal(err)
		}
		right[span] = map[string]map[string]bool{}
		for family, suite := range suites {
			right[span][family] = answerCells(t, suite, cellsRuntime{runner})
		}
		if err := runner.Close(); err != nil {
			t.Fatal(err)
		}
	}
	for family := range suites {
		losses, wins := 0, 0
		for answered, spanRight := range right[true][family] {
			switch nativeRight := right[false][family][answered]; {
			case spanRight && !nativeRight:
				losses++
			case nativeRight && !spanRight:
				wins++
			}
		}
		chance := signTest(losses, wins)
		t.Logf("%s: the span kernels alone answer %d right, the native products alone %d (p %.3g)", family, losses, wins, chance)
		if losses > wins && chance < cellsChance {
			t.Errorf("%s: the native products lose %d cases the span kernels answer and win %d, a lean chance gives %.3g", family, losses, wins, chance)
		}
	}
}
