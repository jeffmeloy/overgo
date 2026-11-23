package server

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"testing"

	"overgo/internal/latentimage"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// workflowCounterpart is what overgo holds for what a MOSS Ultra workflow node does: the page's elements, as its
// scripts write them, and the methods of the picture a run returns.
type workflowCounterpart struct {
	page, result []string
}

// mossWorkflowCounterparts are what overgo holds for each thing a MOSS Ultra section's output node does (ComfyUI's
// output_node classes: SaveImage, PreviewImage, PreviewAny, Image Comparer (rgthree), CreateBoundingBoxes).
var mossWorkflowCounterparts = map[string]workflowCounterpart{
	// The section's finished picture, the run's picture the page shows.
	"finished picture saved": {page: []string{"workbench-output"}},
	// The picture as drawn before its sharpening, which a sharpened run returns first (moss-sharpened-pair).
	"drawn picture saved": {page: []string{"workbench-output"}, result: []string{"BeforeFinishing"}},
	// "Prompt used" and "Negative prompt used", which the run's picture carries and the page shows (moss-prompts-used).
	"prompt used shown":          {page: []string{"Prompt used"}, result: []string{"DrawnFrom"}},
	"negative prompt used shown": {page: []string{"Negative prompt used"}, result: []string{"DrawnFrom"}},
	// The section's comparers, the page's A/B compare.
	"pictures compared": {page: []string{"ab-compare"}},
	// A depth section's PreviewImage of its depth map, the page's depth preview.
	"depth map previewed": {page: []string{"depth-preview"}},
	// The Ideogram builder's region canvas, the layout's box canvas.
	"layout regions drawn": {page: []string{"box-canvas"}},
}

// workflowNode is a node of a section's request as ComfyUI ran it.
type workflowNode struct {
	Inputs map[string]any `json:"inputs"`
	Class  string         `json:"class_type"`
	Meta   struct {
		Title string `json:"title"`
	} `json:"_meta"`
}

// mossOutputNode says what an output node of a section's request does, among mossWorkflowCounterparts; empty where it is
// none of them.
func mossOutputNode(node workflowNode, nodes map[string]workflowNode) string {
	source := func(of workflowNode, input string) workflowNode {
		link, _ := of.Inputs[input].([]any)
		if len(link) == 0 {
			return workflowNode{}
		}
		id, _ := link[0].(string)
		return nodes[id]
	}
	switch node.Class {
	case "SaveImage":
		if source(node, "images").Class == "ImageSharpen" {
			return "finished picture saved"
		}
		return "drawn picture saved"
	case "PreviewAny":
		switch node.Meta.Title {
		case "Prompt used":
			return "prompt used shown"
		case "Negative prompt used":
			return "negative prompt used shown"
		}
	case "PreviewImage":
		// The previewed picture is followed back through what scales it to the node that made it.
		for made := source(node, "images"); made.Class != ""; made = source(made, "image") {
			if strings.Contains(made.Class, "Depth") {
				return "depth map previewed"
			}
		}
	case "Image Comparer (rgthree)":
		return "pictures compared"
	case "CreateBoundingBoxes":
		return "layout regions drawn"
	}
	return ""
}

// TestEveryMOSSWorkflowOutputHasACounterpart holds moss-workflow-surface: every node of each MOSS Ultra section's
// request (fixtures/comfy/moss-ultra/requests) whose class ComfyUI marks an output node, the nodes by which the
// workflow saves, previews, compares or shows something, is one of what overgo holds a counterpart for
// (mossWorkflowCounterparts), and each counterpart is there: its page elements in the generation page's scripts and its
// methods on the picture a run returns. The sections had been held to their pictures alone, and what the workflow shows
// beside them read from a few of its nodes, so its twelve prompt displays and six saves of the unsharpened picture went
// without a counterpart and no check said so; a new output node, or a counterpart taken away, is refused here.
func TestEveryMOSSWorkflowOutputHasACounterpart(t *testing.T) {
	t.Parallel()
	repository := testutil.RepoRoot(t)
	fixture := filepath.Join(repository, "fixtures", "comfy", "moss-ultra")
	var definitions map[string]struct {
		Output bool `json:"output_node"`
	}
	testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(filepath.Join(fixture, "object_info.json")))(t), &definitions))
	var scripts strings.Builder
	for _, script := range []string{"workflow.js", filepath.Join("mod", "generation.js")} {
		scripts.Write(testcheck.Must(os.ReadFile(filepath.Join(repository, "internal", "server", "webui", script)))(t))
	}
	picture := reflect.TypeFor[latentimage.EncodedImage]()
	for name, counterpart := range mossWorkflowCounterparts {
		for _, element := range counterpart.page {
			if !strings.Contains(scripts.String(), element) {
				t.Errorf("%s's counterpart, the page's %q, is not in the generation page's scripts", name, element)
			}
		}
		for _, method := range counterpart.result {
			if _, held := picture.MethodByName(method); !held {
				t.Errorf("%s's counterpart, the picture's %s, is not a method of latentimage.EncodedImage", name, method)
			}
		}
	}
	requests := testcheck.Must(filepath.Glob(filepath.Join(fixture, "requests", "*.json")))(t)
	if len(requests) != 6 {
		t.Fatalf("the fixture holds %d section requests; want the workflow's six", len(requests))
	}
	done := map[string]int{}
	for _, path := range requests {
		var nodes map[string]workflowNode
		testcheck.NoError(t, json.Unmarshal(testcheck.Must(os.ReadFile(path))(t), &nodes))
		ids := slices.Sorted(func(yield func(string) bool) {
			for id := range nodes {
				if !yield(id) {
					return
				}
			}
		})
		for _, id := range ids {
			node := nodes[id]
			if !definitions[node.Class].Output {
				continue
			}
			what := mossOutputNode(node, nodes)
			if _, held := mossWorkflowCounterparts[what]; !held {
				t.Errorf("%s node %s (%s %q) saves, previews, compares or shows something overgo holds no counterpart for", filepath.Base(path), id, node.Class, node.Meta.Title)
				continue
			}
			done[what]++
		}
	}
	t.Logf("the workflow's output nodes, by what they do: %v", done)
}
