package repoanalysis

import (
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// reductionsDisplaced is what each finding of the register's test reductions (compact-tests/sqa-reductions) removes,
// by the file that held it: the declaration, local, call or line that stood for the repeat its finding names. A file
// that is gone holds none of it.
var reductionsDisplaced = map[string]map[string][]string{
	"SQA-310": {
		"internal/repoanalysis/production_repeats_test.go": {"func TestParameterRepeatsStayUnderTheirCeiling(", "func TestSurfaceRepeatsStayUnderTheirCeiling(", "linkedPackages(t, root)"},
		"internal/repoanalysis/local_forwarders_test.go":   {"func TestLocalForwardersStayUnderTheirCeiling(", "linkedPackages(t, root)"},
		"internal/repoanalysis/test_repeats_test.go":       {"func TestNoTestBodyRepeatsThreeTimes(", "func TestNoTestBodyOfThreeValuesRepeatsThreeTimes(", "loadModule("},
	},
	"SQA-276": {
		"internal/modelcli/flashnext_tensor_cores_model_test.go": {"func TestFlashNextServesPromptTensorCores("},
		"internal/modelcli/dense_tensor_cores_model_test.go":     {"TestFlashNextServesPromptTensorCores"},
	},
	"SQA-237": {"internal/inference/sampled_top_k_windows_test.go": {"func TestSampledToolRequestDrawsFromTheTopKUntilItsTrigger("}},
	"SQA-236": {
		"internal/evaluation/attention_cells_heldout_windows_model_test.go":     {"signTest("},
		"internal/evaluation/qwen35_cells_heldout_windows_model_test.go":        {"signTest("},
		"internal/evaluation/kquant_products_heldout_windows_model_test.go":     {"signTest("},
		"internal/evaluation/prompt_tensor_cores_heldout_windows_model_test.go": {"signTest(", "rawRuntime"},
	},
	"SQA-235": {"internal/inference/decode_session_identity_test.go": {"func TestDecodeSessionIdentityDistinguishesSourcePageCapacity("}},
	"SQA-253": {"internal/overgodb/query.go": {"descriptorCandidateCount"}},
	"SQA-269": {"cmd/validate/onboard_test.go": {"func TestOnboardingAgainRetiresOnlyAConversionStillServing("}},
	"SQA-270": {
		"cmd/validate/only_model_test.go": {"func TestOneModelsCellsOfOneValidationAreMeasuredAlone("},
		"skill.md":                        {"TestOneModelsCellsOfOneValidationAreMeasuredAlone"},
	},
	"SQA-216": {"internal/modelartifact/tensor_budget_test.go": {"openGGUF("}},
	"SQA-207": {"internal/inference/speculating_frames_windows_test.go": {"func TestPromptCachedRequestsReplayTheirFrames("}},
	"SQA-222": {"internal/scorecard/captioned_generation_test.go": {"func TestADeclaredNotApplicableQualityIsNotWorkToDefine("}},
	"SQA-225": {"internal/latentvideo/reference_edit_runtime_cuda_windows_test.go": {"priorFrameHash"}},
	"SQA-252": {
		"internal/server/held_open_header_test.go": {"func TestActivityStreamNamesItselfHeldOpen("},
		"internal/server/one_event_stream_test.go": {"bufio.NewScanner("},
	},
	"SQA-240": {"internal/modeldevice/device_load_test.go": {`"load survived a failed write"`}},
}

// TestTheRegistersTestReductionsLand holds the checkout to the register's test reductions: none of what any of its
// findings displaced stands in its file again. The findings are closed and gone from the register; the checkout shows
// the repeat gone.
func TestTheRegistersTestReductionsLand(t *testing.T) {
	t.Parallel()
	for id, files := range reductionsDisplaced {
		for file, displaced := range files {
			content, err := os.ReadFile(filepath.Join(checkoutRoot, filepath.FromSlash(file)))
			if errors.Is(err, fs.ErrNotExist) {
				continue
			}
			testcheck.NoError(t, err)
			for _, text := range displaced {
				if strings.Contains(string(content), text) {
					t.Errorf("%s: %s holds %q again, which the reduction removed", id, file, text)
				}
			}
		}
	}
}
