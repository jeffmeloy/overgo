package repoanalysis

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// reductionsRow is the plan step that lands the register's test reductions.
const reductionsRow = "compact-tests/sqa-reductions"

// reductionsDisplaced is what each of reductionsRow's findings removes, by the file that held it: the declaration,
// local, call or line that stood for the repeat its finding names. A file that is gone holds none of it.
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

// TestTheRegistersTestReductionsLand holds reductionsRow to the review register, docs/sqa_findings.json, and to the
// checkout: each finding whose plan references name it and that stays open is triaged implemented and names what it
// displaced in reductionsDisplaced, and none of what any of them displaced stands in its file again. The reviewer
// resolves a finding; the checkout shows the repeat gone.
func TestTheRegistersTestReductionsLand(t *testing.T) {
	t.Parallel()
	data, err := os.ReadFile(filepath.Join(checkoutRoot, "docs", "sqa_findings.json"))
	testcheck.NoError(t, err)
	var register struct {
		Findings []struct {
			ID     string   `json:"id"`
			Status string   `json:"status"`
			Rows   []string `json:"plan_references"`
		} `json:"findings"`
		Triage struct {
			Findings map[string]struct {
				Action string `json:"action"`
			} `json:"findings"`
		} `json:"master_lead_triage"`
	}
	testcheck.NoError(t, json.Unmarshal(data, &register))
	known := map[string]bool{}
	for _, finding := range register.Findings {
		known[finding.ID] = true
		if finding.Status != "open" || !slices.Contains(finding.Rows, reductionsRow) {
			continue
		}
		if action := register.Triage.Findings[finding.ID].Action; action != "implemented" {
			t.Errorf("%s names %s and stays open with its triage %q, want it implemented", finding.ID, reductionsRow, action)
		}
		if reductionsDisplaced[finding.ID] == nil {
			t.Errorf("%s names %s, but reductionsDisplaced names nothing it removes", finding.ID, reductionsRow)
		}
	}
	for id, files := range reductionsDisplaced {
		if !known[id] {
			t.Errorf("reductionsDisplaced names %s, which the register does not hold", id)
		}
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
