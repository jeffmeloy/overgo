//go:build windows && modeltest && pairtest

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
	"overgo/internal/tensorstats"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// promptTensorCoreModels are a dense model of each architecture the profiles admit to the tensor-core prompt attention
// besides qwen4exp, as their scorecard rows serve them, each with the protocol its IFEval reading is scored by.
var promptTensorCoreModels = []struct {
	name, path string
	ifeval     prompting
}{
	{"Qwen2.5-0.5B (qwen2)", `overgo-hfconvert/Qwen2.5-0.5B-f16.gguf`, PromptingChatTemplate},
	{"MiniCPM5-1B (llama)", `MiniCPM5-1B-native-chat.gguf`, PromptingChatTemplate},
	{"gemma-4-E4B-it (gemma4)", `overgo-hfconvert/gemma-4-E4B-it-bf16.gguf`, PromptingChatTemplate},
	{"Unlimited-OCR (deepseek2-ocr)", `overgo-hfconvert/Unlimited-OCR-bf16.gguf`, PromptingRawCompletion},
}

// TestPromptTensorCoresHoldHeldOut: a dense model of each admitted architecture answers BBH by raw completion and IFEval
// by its scored protocol as well with its prompts' attention on the tensor-core kernel as on the F32 paths: of the cases
// the two paths answer differently, those only the F32 paths answer right do not outnumber those only the kernel does
// past chance (cellsChance). Each path answers into a store of its own, so neither reads the other's answers. On
// 2026-10-31 the kernel with its products split into FP16 roundings and their remainders lost 13 cases and gained 20
// over these models' BBH, IFEval and MMLU-Pro, where one rounding of each lost MiniCPM5's IFEval 4 prompts to 1.
func TestPromptTensorCoresHoldHeldOut(t *testing.T) {
	// Serial: both paths share the one card and each model is loaded alone.
	cudatest.Require(t)
	roots := testcheck.Must(dataroot.Resolve(testutil.RepoRoot(t)))(t)
	// derive compiles the family's suite under the protocol asked, once for each model's parts.
	derive := func(t *testing.T, family string, asked prompting) CompiledSuite {
		t.Helper()
		store := testcheck.Must(overgodb.OpenReadOnly(roots.Store))(t)
		defer store.Close()
		authorities := ExactAuthorities{
			ModelDefinition: planID(t, artifact.KindModelDefinition, "prompt-tensor-cores"), RuntimeRecipe: planID(t, artifact.KindRecipe, "prompt-tensor-cores"),
			CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "device"),
			Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: asked},
		}
		derived, _, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, family)
		if err != nil || len(derived) != 1 {
			t.Fatalf("the store derived %d suites for %s: %v", len(derived), family, err)
		}
		return derived[0]
	}
	for _, served := range promptTensorCoreModels {
		file := roots.ResolveModelPath(served.path)
		if _, err := os.Stat(file); err != nil {
			file = filepath.Join(roots.Checkpoints, filepath.FromSlash(served.path))
		}
		if _, err := os.Stat(file); err != nil {
			t.Fatalf("%s's held-out check needs %s: %v", served.name, served.path, err)
		}
		t.Run(served.name, func(t *testing.T) {
			parts := map[string]CompiledSuite{"bbh": derive(t, "bbh", PromptingRawCompletion), "ifeval": derive(t, "ifeval", served.ifeval)}
			right := map[bool]map[string]map[string]bool{}
			for _, tensorCores := range []bool{false, true} {
				runner := testcheck.Must(modelcli.OpenRunner(t.Context(), roots.Store, file, inference.OpenOptions{PromptTensorCores: &tensorCores}))(t)
				t.Cleanup(func() {
					if err := runner.Close(); err != nil {
						t.Error(err)
					}
				})
				right[tensorCores] = map[string]map[string]bool{}
				for family, suite := range parts {
					var runtime Runtime = runner
					if family == "ifeval" && served.ifeval == PromptingChatTemplate {
						runtime = cellsRuntime{runner}
					}
					right[tensorCores][family] = answerCells(t, suite, runtime)
				}
				testcheck.NoError(t, runner.Close())
			}
			for family := range parts {
				losses, wins := 0, 0
				for answered, f32Right := range right[false][family] {
					switch kernelRight := right[true][family][answered]; {
					case f32Right && !kernelRight:
						losses++
					case kernelRight && !f32Right:
						wins++
					}
				}
				chance := min(1, 2*tensorstats.ExactPairedBinary(uint64(max(losses, wins)), uint64(min(losses, wins)), 0).Tail)
				t.Logf("%s: the F32 paths alone answer %d right, the tensor cores alone %d (p %.3g)", family, losses, wins, chance)
				if losses > wins && chance < cellsChance {
					t.Errorf("%s: the tensor cores lose %d cases the F32 paths answer and win %d, a lean chance gives %.3g", family, losses, wins, chance)
				}
			}
		})
	}
}
