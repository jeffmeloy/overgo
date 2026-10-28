//go:build windows && modeltest

package evaluation

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
)

// qwen35CellsModels are the qwen35 family's served models whose attention layers keep their K/V as int8 cells
// (cells-qwen35), each a file beneath the roots that hold it: Qwen3.8-27B's and Qwen3.5-9B's beneath a models root,
// Qwen3.5-4B's conversion beneath the checkpoints root.
func qwen35CellsModels(roots dataroot.Roots) []struct {
	name, file string
	under      []string
} {
	return []struct {
		name, file string
		under      []string
	}{
		{"Qwen3.8-27B", kQuantModel, roots.ModelRoots()},
		{"Qwen3.5-9B", "Qwen3.5-9B-Q8_0.gguf", roots.ModelRoots()},
		{"Qwen3.5-4B", filepath.Join("overgo-hfconvert", "Qwen3.5-4B-f16.gguf"), []string{roots.Checkpoints}},
	}
}

// TestQwen35CellsHoldHeldOut: each of the qwen35 family's served models answers the held-out parts its measurement run
// asks through the chat template, MMLU-Pro and IFEval, as well over its attention layers' int8 cells (cells-qwen35: a
// code and a 64-value group's fp16 scale where the F32 cache keeps the value) as over the F32 cache: of the cases the
// two paths answer differently, those only the F32 cache answers right do not outnumber those only the cells do past
// chance (cellsChance, the sign test gemma-4-12B's cells were held to). Each path answers into a store of its own.
func TestQwen35CellsHoldHeldOut(t *testing.T) {
	// Serial: both paths share the one card and each model is loaded alone.
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	type family struct {
		name   string
		suites map[string]CompiledSuite
	}
	suitesOf := func(model string) map[string]CompiledSuite {
		suites := map[string]CompiledSuite{}
		for _, family := range []string{"mmlu-pro", "ifeval"} {
			authorities := ExactAuthorities{
				ModelDefinition: planID(t, artifact.KindModelDefinition, model), RuntimeRecipe: planID(t, artifact.KindRecipe, model),
				CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "device"),
				Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
			}
			derived, _, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, family)
			if err != nil || len(derived) != 1 {
				t.Fatalf("the store derived %d suites for %s: %v", len(derived), family, err)
			}
			suites[family] = derived[0]
		}
		return suites
	}
	models := qwen35CellsModels(roots)
	derived := make([]family, len(models))
	for index, model := range models {
		derived[index] = family{name: model.name, suites: suitesOf(strings.ToLower(model.name))}
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	for index, model := range models {
		t.Run(model.name, func(t *testing.T) {
			// The model is read where its roots find it first.
			var path string
			for _, root := range model.under {
				if candidate := filepath.Join(root, model.file); path == "" {
					if _, err := os.Stat(candidate); err == nil {
						path = candidate
					}
				}
			}
			if path == "" {
				t.Skipf("%s's held-out check needs %s beneath one of %q", model.name, model.file, model.under)
			}
			suites := derived[index].suites
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
		})
	}
}
