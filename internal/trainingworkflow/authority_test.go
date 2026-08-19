package trainingworkflow

import (
	"go/ast"
	"path"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

func TestRecipePlanOwnsAllTrainingExecution(t *testing.T) {
	const workflowPackage = "overgo/internal/trainingworkflow"
	wantWorkflowCallers := []string{"cmd/train/main.go", "internal/server/training_workspace.go"}
	directTraining := map[string]map[string]bool{
		"overgo/internal/adaptertrain":    {"NewTrainer": true},
		"overgo/internal/densecausal":     {"Train": true, "TrainDPOBatchesResume": true, "TrainDeviceResident": true},
		"overgo/internal/diffusionimage":  {"NewTrainer": true, "TrainBatch": true, "TrainOTBatch": true},
		"overgo/internal/hybridtrain":     {"TrainHost": true, "TrainDeviceResident": true},
		"overgo/internal/oscillatorimage": {"NewTrainer": true},
		"overgo/internal/scratchmodel":    {"TrainShared": true, "TrainResident": true},
		"overgo/internal/seq2seq":         {"NewTrainer": true},
		"overgo/internal/seriesforecast":  {"NewTrainer": true},
		"overgo/internal/speechsynth":     {"NewTrainer": true},
		"overgo/internal/thoughtbank":     {"NewTrainer": true},
	}

	snapshot, err := repoanalysis.DiscoverGo("../..", "cmd", "internal/server")
	if err != nil {
		t.Fatal(err)
	}
	var workflowCallers []string
	for _, source := range snapshot.Files {
		if source.Test {
			continue
		}
		file, err := source.Syntax()
		if err != nil {
			t.Fatal(err)
		}
		imports := make(map[string]string, len(file.Imports))
		for _, spec := range file.Imports {
			imported, err := strconv.Unquote(spec.Path.Value)
			if err != nil {
				t.Fatal(err)
			}
			name := path.Base(imported)
			if spec.Name != nil {
				name = spec.Name.Name
			}
			imports[name] = imported
		}
		ast.Inspect(file, func(node ast.Node) bool {
			call, ok := node.(*ast.CallExpr)
			if !ok {
				return true
			}
			selector, ok := call.Fun.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			qualifier, ok := selector.X.(*ast.Ident)
			if !ok {
				return true
			}
			imported := imports[qualifier.Name]
			if imported == workflowPackage && selector.Sel.Name == "Execute" {
				workflowCallers = append(workflowCallers, source.Path)
			}
			if strings.HasPrefix(source.Path, "cmd/") && !strings.Contains(source.Path, "-probe/") &&
				directTraining[imported][selector.Sel.Name] {
				t.Errorf("%s calls %s.%s outside the recipe workflow", source.Path, qualifier.Name, selector.Sel.Name)
			}
			return true
		})
	}
	slices.Sort(workflowCallers)
	if !slices.Equal(workflowCallers, wantWorkflowCallers) {
		t.Fatalf("training workflow callers=%v want %v", workflowCallers, wantWorkflowCallers)
	}
}
