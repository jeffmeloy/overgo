package plan

import (
	"fmt"
	"slices"
	"strings"
	"testing"
)

// The optimized campaign keeps prerequisite evidence ahead of changes that
// invalidate it, without putting independent host work behind GPU campaigns.
// These are safety relationships, not a copy of the full dispatch order.
func optimizedValidationProblems(document Plan) []string {
	var problems []string
	if err := Validate(document); err != nil {
		return []string{err.Error()}
	}
	prerequisites := map[string][]string{
		"model-regression-baseline/do":                {"validation-readiness/measurement-contract", "model-regression-baseline/prepare-guard", "model-regression-baseline/repair-throughput"},
		"model-regression-baseline/complete-coverage": {"baseline-repair-order/do", "model-regression-baseline/prepare-guard"},
		"model-regression-baseline/repair-throughput": {"model-regression-baseline/complete-coverage"},
		"model-regression-baseline/prepare-guard":     {"validation-readiness/measurement-contract"},
		"model-regression-baseline/full-catalog":      {"model-regression-baseline/do", "gemma-12b-accuracy/do", "modality-verification/media-report", "device-memory-retention/do", "decode-attention-per-key-cost/do"},
		"device-memory-retention/do":                  {"model-regression-gate/do"},
		"decode-attention-per-key-cost/do":            {"model-regression-gate/do"},
		"gemma-12b-accuracy/do":                       {"model-regression-gate/do"},
		"model-regression-gate/do":                    {"model-regression-baseline/do"},
		"model-regression-gate/coverage-inventory":    {"model-regression-baseline/do"},
		"model-regression-gate/coverage-selection":    {"model-regression-gate/coverage-inventory"},
		"model-regression-gate/coverage-acquisition":  {"model-regression-gate/coverage-selection"},
		"simplify-prefill-paths/do":                   {"model-regression-gate/do", "model-regression-baseline/full-catalog", "modality-verification/media-report", "benchmark-completion/mmlu-pro-pass"},
		"simplify-execution-core/do":                  {"simplify-prefill-paths/do"},
		"final-model-validation/do":                   {"simplify-execution-core/do", "simplify-command-surface/do", "simplify-checkpoint-locations/do", "boundary-hardening/cross-origin", "boundary-hardening/argv-output"},
		"benchmark-27b/mmlu-pro-pass":                 {"benchmark-completion/mmlu-pro-pass"},
		"benchmark-completion/published-comparison":   {"benchmark-27b/mmlu-pro-pass", "validation-automation/benchmark-protocol"},
		"validation-automation/benchmark-protocol":    {"model-regression-baseline/complete-coverage"},
		"simplify-checkpoint-locations/do":            {"model-regression-gate/do", "modality-verification/media-report", "benchmark-completion/mmlu-pro-pass"},
		"simplify-command-surface/do":                 {"model-regression-gate/do", "modality-verification/media-report", "benchmark-completion/mmlu-pro-pass", "failure-recovery/control-plane-drills", "failure-recovery/gate-recovery-drill"},
		"simplify-capability-report/do":               {"final-model-validation/do", "benchmark-completion/published-comparison", "modality-verification/media-report"},
	}
	for after, before := range prerequisites {
		step, retained := retainedCampaignStep(document, after)
		if !retained {
			continue
		}
		for _, prerequisite := range before {
			// A completed prerequisite is pruned, but its reference must
			// remain for store/Git proof at dispatch.
			if !slices.Contains(step.DependsOn, prerequisite) {
				problems = append(problems, fmt.Sprintf("%s must follow %s", after, prerequisite))
			}
		}
	}
	_, closing := retainedCampaignStep(document, "campaign-closeout/closeout")
	if len(document.Items) > 0 && !closing {
		problems = append(problems, "unfinished campaign has no closeout row")
	}
	for _, item := range document.Items {
		for _, step := range item.Steps {
			ref := item.ID + "/" + step.ID
			if closing && ref != "campaign-closeout/closeout" && !campaignDependsOn(document, "campaign-closeout/closeout", ref, map[string]bool{}) {
				problems = append(problems, "closeout does not wait for "+ref)
			}
			if ref == "benchmark-27b/mmlu-pro-pass" && item.Owner != "operator" {
				problems = append(problems, "full 27B benchmark must remain operator-owned")
			}
			if strings.Contains(step.Verify, "UNBOUND") || strings.Contains(step.Verify, "./cmd/model-regress") || strings.HasPrefix(step.Verify, "go build ") || step.Verify == "go run ./cmd/plan -status" {
				problems = append(problems, "non-acceptance verifier for "+ref)
			}
		}
	}
	for _, host := range []string{"boundary-hardening/cross-origin", "boundary-hardening/argv-output", "validation-readiness/failure-diagnostics", "validation-automation/gate-scope-efficiency", "validation-automation/benchmark-protocol", "failure-recovery/rollout-plan-author", "failure-recovery/gate-recovery-drill", "modality-verification/capability-census"} {
		for _, benchmark := range []string{"benchmark-completion/mmlu-pro-pass", "benchmark-27b/mmlu-pro-pass", "benchmark-completion/published-comparison"} {
			if campaignDependsOn(document, host, benchmark, map[string]bool{}) {
				problems = append(problems, host+" unnecessarily waits for "+benchmark)
			}
		}
	}
	for _, automation := range []string{"validation-automation/gate-scope-efficiency", "validation-automation/benchmark-protocol"} {
		for _, delayed := range []string{"model-regression-baseline/do", "modality-verification/media-report"} {
			if campaignDependsOn(document, automation, delayed, map[string]bool{}) {
				problems = append(problems, automation+" unnecessarily waits for "+delayed)
			}
		}
		if campaignDependsOn(document, "benchmark-completion/mmlu-pro-pass", automation, map[string]bool{}) {
			problems = append(problems, "independent quality pass waits for "+automation)
		}
	}
	for _, initial := range []string{"benchmark-completion/mmlu-pro-pass", "modality-verification/text-and-vision", "modality-verification/image-and-video", "modality-verification/speech-ocr-tabular-forecast"} {
		for _, deferred := range []string{"validation-readiness/measurement-contract", "device-memory-retention/do", "decode-attention-per-key-cost/do", "gemma-12b-accuracy/do", "model-regression-baseline/do", "simplify-prefill-paths/do", "simplify-execution-core/do", "simplify-command-surface/do", "simplify-checkpoint-locations/do", "final-model-validation/do"} {
			if campaignDependsOn(document, initial, deferred, map[string]bool{}) {
				problems = append(problems, initial+" must not wait for "+deferred)
			}
		}
	}
	for _, pair := range [][2]string{{"device-memory-retention/do", "decode-attention-per-key-cost/do"}, {"modality-verification/text-and-vision", "modality-verification/image-and-video"}, {"modality-verification/image-and-video", "modality-verification/speech-ocr-tabular-forecast"}} {
		if campaignDependsOn(document, pair[0], pair[1], map[string]bool{}) || campaignDependsOn(document, pair[1], pair[0], map[string]bool{}) {
			problems = append(problems, "independent work serialized: "+pair[0]+" and "+pair[1])
		}
	}
	for _, guard := range []string{"model-regression-baseline/do", "model-regression-gate/do"} {
		for _, later := range []string{"model-regression-baseline/full-catalog", "modality-verification/media-report", "device-memory-retention/do", "decode-attention-per-key-cost/do", "gemma-12b-accuracy/do"} {
			if campaignDependsOn(document, guard, later, map[string]bool{}) {
				problems = append(problems, guard+" must protect rather than wait for "+later)
			}
		}
	}
	for _, repair := range []string{"model-regression-baseline/complete-coverage", "model-regression-baseline/repair-throughput"} {
		for _, blocked := range []string{"model-regression-baseline/do", "model-regression-gate/do"} {
			if campaignDependsOn(document, repair, blocked, map[string]bool{}) {
				problems = append(problems, repair+" cannot wait for the healthy baseline it must repair")
			}
		}
	}
	slices.Sort(problems)
	return problems
}

func TestOptimizedValidationCampaign(t *testing.T) {
	document := loadCampaignPlan(t)
	if !strings.Contains(document.Campaign, "bounded capability gains") {
		return // Other campaigns bind their own structure in campaign_structure_test.go.
	}
	for _, problem := range optimizedValidationProblems(document) {
		t.Error(problem)
	}
}

func TestOptimizedValidationRejectsUnsafeOrdering(t *testing.T) {
	for _, test := range []struct {
		name   string
		target string
		change func(*Plan)
	}{
		{"automation delayed by model campaign", "validation-automation/gate-scope-efficiency", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "validation-automation" {
					d.Items[i].Steps[0].DependsOn = []string{"model-regression-baseline/do"}
				}
			}
		}},
		{"quality delayed by unrelated automation", "benchmark-completion/mmlu-pro-pass", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "benchmark-completion" {
					d.Items[i].Steps[0].DependsOn = append(d.Items[i].Steps[0].DependsOn, "validation-automation/gate-scope-efficiency")
				}
			}
		}},
		{"baseline repair deadlock", "model-regression-baseline/repair-throughput", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "model-regression-baseline" {
					for j := range d.Items[i].Steps {
						if d.Items[i].Steps[j].ID == "repair-throughput" {
							d.Items[i].Steps[j].DependsOn = append(d.Items[i].Steps[j].DependsOn, "model-regression-gate/do")
						}
					}
				}
			}
		}},
		{"publication bypasses producer", "model-regression-baseline/do", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID != "model-regression-baseline" {
					continue
				}
				for j := range d.Items[i].Steps {
					if d.Items[i].Steps[j].ID == "do" {
						d.Items[i].Steps[j].DependsOn = slices.DeleteFunc(d.Items[i].Steps[j].DependsOn, func(dependency string) bool {
							return dependency == "model-regression-baseline/prepare-guard"
						})
					}
				}
			}
		}},
		{"guard delayed by media", "model-regression-baseline/do", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "model-regression-baseline" {
					d.Items[i].Steps[0].DependsOn = append(d.Items[i].Steps[0].DependsOn, "modality-verification/media-report")
				}
			}
		}},
		{"benchmark delayed by contract", "benchmark-completion/mmlu-pro-pass", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "benchmark-completion" {
					d.Items[i].Steps[0].DependsOn = append(d.Items[i].Steps[0].DependsOn, "validation-readiness/measurement-contract")
				}
			}
		}},
		{"independent defects serialized", "decode-attention-per-key-cost/do", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "decode-attention-per-key-cost" {
					d.Items[i].Steps[0].DependsOn = []string{"device-memory-retention/do"}
				}
			}
		}},
		{"baseline bypass", "simplify-prefill-paths/do", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "simplify-prefill-paths" {
					d.Items[i].Steps[0].DependsOn = nil
				}
			}
		}},
		{"premature closeout", "simplify-capability-report/do", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "campaign-closeout" {
					d.Items[i].Steps[0].DependsOn = []string{"failure-recovery/gate-recovery-drill"}
				}
			}
		}},
		{"unattended 27B", "benchmark-27b/mmlu-pro-pass", func(d *Plan) {
			for i := range d.Items {
				if d.Items[i].ID == "benchmark-27b" {
					d.Items[i].Owner = ""
				}
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			document := loadCampaignPlan(t)
			if !strings.Contains(document.Campaign, "bounded capability gains") {
				return
			}
			// Closed rows legitimately leave the plan; mutations are only
			// meaningful while their targets remain.
			if _, ok := retainedCampaignStep(document, test.target); !ok {
				return
			}
			test.change(&document)
			if len(optimizedValidationProblems(document)) == 0 {
				t.Fatal("unsafe plan mutation was accepted")
			}
		})
	}
}
