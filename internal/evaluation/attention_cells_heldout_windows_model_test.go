//go:build windows && modeltest && pairtest

package evaluation

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// cellsModels are gemma-4-12B-it-fp8's conversions as their scorecard rows serve them, beneath the checkpoints root:
// every attention layer of each keeps its K/V as Strata's int8 cells (widen-kv-op), a code and a 64-value group's
// fp16 scale where the F32 cache keeps the value.
var cellsModels = []string{
	filepath.Join("overgo-hfconvert", "gemma-4-12B-it-fp8-bf16.gguf"),
	filepath.Join("overgo-hfconvert", "gemma-4-12B-it-fp8-native.gguf"),
}

// cellsChance is the sign test's level: the cases one path answers right and the other wrong lean against the cells
// past chance when a split as uneven comes less often than this. The tensor-core prompt attention's 8 BBH cases to
// none (p 0.008) declined; gemma-4-12B-it-fp8's cells parted from the F32 cache on 19 of BBH's 577, 10 to 9 (p 1).
const cellsChance = 0.05

// cellsRuntime answers a case through the runner as the suite's protocol asks: a raw completion's continuations
// scored, or one user turn through the chat template.
type cellsRuntime struct{ *inference.Runner }

func (r cellsRuntime) ShapeChatPrompt(prompt string) (string, error) {
	return r.FormatChatWithOptions(
		[]inference.ChatMessage{{Role: inference.ChatRoleUser, Content: prompt}},
		inference.ChatFormatOptions{AddGenerationPrompt: true},
	)
}

// TestAttentionCellsHoldHeldOut: gemma-4-12B answers the held-out parts its measurement run scores, BBH by raw
// completion and IFEval through the chat template, as well over its int8 attention cells as over the F32 cache: of the
// cases the two paths answer differently, those only the F32 cache answers right do not outnumber those only the cells
// do past chance. Each path answers into a store of its own, so neither reads the other's answers.
func TestAttentionCellsHoldHeldOut(t *testing.T) {
	// Serial: both paths share the one card and each model is loaded alone.
	cudatest.Require(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	parts := map[string]prompting{"bbh": PromptingRawCompletion, "ifeval": PromptingChatTemplate}
	suites := map[string]CompiledSuite{}
	store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
	for family, asked := range parts {
		authorities := ExactAuthorities{
			ModelDefinition: planID(t, artifact.KindModelDefinition, "gemma-4-12b"), RuntimeRecipe: planID(t, artifact.KindRecipe, "gemma-4-12b"),
			CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "device"),
			Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: asked},
		}
		derived, _, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, family)
		if err != nil || len(derived) != 1 {
			t.Fatalf("the store derived %d suites for %s: %v", len(derived), family, err)
		}
		suites[family] = derived[0]
	}
	testcheck.NoError(t, store.Close())
	for _, name := range cellsModels {
		path := filepath.Join(roots.Checkpoints, name)
		if _, err := os.Stat(path); err != nil {
			t.Skipf("gemma-4-12B's held-out check needs %s: %v", path, err)
		}
		t.Run(filepath.Base(name), func(t *testing.T) {
			right := map[bool]map[string]map[string]bool{}
			for _, f32 := range []bool{false, true} {
				runner := testcheck.Must(modelcli.OpenRunner(t.Context(), roots.Store, path, inference.OpenOptions{F32Cache: f32}))(t)
				t.Cleanup(func() {
					if err := runner.Close(); err != nil {
						t.Error(err)
					}
				})
				right[f32] = map[string]map[string]bool{}
				for family, suite := range suites {
					right[f32][family] = answerCells(t, suite, cellsRuntime{runner})
				}
				testcheck.NoError(t, runner.Close())
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
				chance := min(1, 2*tensorstats.ExactPairedBinary(uint64(max(losses, wins)), uint64(min(losses, wins)), 0).Tail)
				t.Logf("%s: the F32 cache alone answers %d right, the cells alone %d (p %.3g)", family, losses, wins, chance)
				if losses > wins && chance < cellsChance {
					t.Errorf("%s: the cells lose %d cases the F32 cache answers and win %d, a lean chance gives %.3g", family, losses, wins, chance)
				}
			}
		})
	}
}

// answerCells scores the suite under the runtime into a store of its own and returns, by case, whether it answered
// right: a grouped choice's selected candidate its answer, an instruction prompt's every rule followed strictly.
func answerCells(t *testing.T, suite CompiledSuite, runtime Runtime) map[string]bool {
	t.Helper()
	answers := testcheck.Must(overgodb.Open(t.TempDir()))(t)
	defer answers.Close()
	publishPlanFixtureAuthorities(t, answers, suite.plan)
	result := testcheck.Must(suite.execute(context.WithoutCancel(t.Context()), answers, runtime))(t)
	content := testcheck.Must(artifact.RequireTypedContent(t.Context(), answers, result.Report))(t)
	var report struct {
		Observations []struct {
			Name     string
			Selected *int
			Answer   *int
			Strict   []bool
		}
	}
	if err := json.Unmarshal(content.Data, &report); err != nil || len(report.Observations) != int(suite.descriptor.Cases) {
		t.Fatalf("the report holds %d of %d cases: %v", len(report.Observations), suite.descriptor.Cases, err)
	}
	right := make(map[string]bool, len(report.Observations))
	for _, observation := range report.Observations {
		switch {
		case observation.Selected != nil && observation.Answer != nil:
			right[observation.Name] = *observation.Selected == *observation.Answer
		default:
			followed := len(observation.Strict) > 0
			for _, strict := range observation.Strict {
				followed = followed && strict
			}
			right[observation.Name] = followed
		}
	}
	return right
}
