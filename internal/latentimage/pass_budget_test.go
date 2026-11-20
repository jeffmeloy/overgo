package latentimage

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// passBudget is ComfyUI's MOSS pass split into overgo's parts, each part's bar beside it, and the workflow's terms the
// pass at which overgo matches ComfyUI's whole workflow is read from (fixtures/comfy/moss-ultra/pass_budget.json).
type passBudget struct {
	Measured    string  `json:"measured"`
	Passes      int     `json:"passes"`
	ComfyPassMS float64 `json:"comfy_pass_ms"`
	Workflow    struct {
		ComfyS           float64 `json:"comfy_s"`
		Passes           int     `json:"passes"`
		Sections         int     `json:"sections"`
		OutsideSamplingS float64 `json:"outside_sampling_s"`
		MergesS          float64 `json:"merges_s"`
		DepthS           float64 `json:"depth_s"`
	} `json:"workflow"`
	Parts []struct {
		Part          string  `json:"part"`
		ComfyMS       float64 `json:"comfy_ms"`
		BarMS         float64 `json:"bar_ms"`
		Row           string  `json:"row"`
		ComfyKernels  string  `json:"comfy_kernels"`
		OvergoMS      float64 `json:"overgo_ms"`
		OvergoKernels string  `json:"overgo_kernels"`
	} `json:"parts"`
}

// TestThePassBudgetReachesComfyUI holds the per-pass rows' bars to ComfyUI's measured pass (moss-fusion/budget):
// ComfyUI's parts, read kernel by kernel from its own text to image pass at 1024 x 576, add up to no more than its
// pass; each part's bar is no tighter than ComfyUI's own part, which a bar below it would ask of overgo with no
// measurement that it can be reached; and the bars add up to at most the pass at which overgo matches ComfyUI's whole
// workflow, its time outside sampling, the adapter merges and a fresh depth map taken as their rows set them. So it
// is known before more is built that the rows, each reaching its bar, reach ComfyUI's time.
func TestThePassBudgetReachesComfyUI(t *testing.T) {
	t.Parallel()
	var budget passBudget
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(testutil.RepoRoot(t), "fixtures", "comfy", "moss-ultra", "pass_budget.json")))(t), &budget))
	workflow := budget.Workflow
	if budget.Passes <= 0 || workflow.Passes <= 0 || len(budget.Parts) == 0 {
		t.Fatalf("the budget %+v names no passes or parts", budget)
	}
	matching := 1000 * (workflow.ComfyS - float64(workflow.Sections)*workflow.OutsideSamplingS - workflow.MergesS - workflow.DepthS) / float64(workflow.Passes)
	var comfy, bars float64
	for _, part := range budget.Parts {
		if part.BarMS < part.ComfyMS {
			t.Errorf("%s's bar is %.2f ms, under ComfyUI's own %.2f ms", part.Part, part.BarMS, part.ComfyMS)
		}
		comfy += part.ComfyMS
		bars += part.BarMS
	}
	if comfy > budget.ComfyPassMS {
		t.Errorf("ComfyUI's parts add up to %.2f ms, more than its pass of %.2f ms", comfy, budget.ComfyPassMS)
	}
	if bars > matching {
		t.Errorf("the parts' bars add up to %.2f ms a pass; overgo matches ComfyUI's workflow at %.2f ms", bars, matching)
	}
	t.Logf("ComfyUI's parts %.2f ms of its %.2f ms pass; the bars %.2f ms against the matching pass of %.2f ms", comfy, budget.ComfyPassMS, bars, matching)
}
