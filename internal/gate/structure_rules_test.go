package gate

import (
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/repoanalysis"
)

// The structure rules below parse other packages' sources. They live here
// so the packages they hold import no Go parser in their tests, which
// would make each a source reader the gate selects on every Go change.

// repositoryRoot resolves the checkout these rules read.
func repositoryRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}

// TestTrainCommandUsesNativeWorkflow holds the train command to the recipe
// workflow: it imports and executes trainingworkflow, never a trainer.
func TestTrainCommandUsesNativeWorkflow(t *testing.T) {
	t.Parallel()
	file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(repositoryRoot(t), "cmd", "train", "main.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	imports := map[string]bool{}
	for _, spec := range file.Imports {
		path, err := strconv.Unquote(spec.Path.Value)
		if err != nil {
			t.Fatal(err)
		}
		imports[path] = true
	}
	if !imports["overgo/internal/trainingworkflow"] || imports["overgo/internal/densecausal"] || imports["overgo/internal/trainingprogram"] {
		t.Fatalf("command imports=%v", imports)
	}
	executesWorkflow := false
	ast.Inspect(file, func(node ast.Node) bool {
		call, ok := node.(*ast.CallExpr)
		if !ok {
			return true
		}
		selector, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			return true
		}
		owner, ownerOK := selector.X.(*ast.Ident)
		if ownerOK && owner.Name == "trainingworkflow" && selector.Sel.Name == "Execute" {
			executesWorkflow = true
		}
		return true
	})
	if !executesWorkflow {
		t.Fatal("command does not execute the native training workflow")
	}
}

// TestNoDirectCompositionConstructors holds inference to its guarded
// composition constructors.
func TestNoDirectCompositionConstructors(t *testing.T) {
	t.Parallel()
	directory := filepath.Join(repositoryRoot(t), "internal", "inference")
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	protected := map[string]map[string]struct{}{
		"ProductionComposition":         {"OpenProductionComposition": {}},
		"ExternalCrossAttentionProgram": {"OpenExternalCrossAttention": {}},
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		file, parseErr := parser.ParseFile(token.NewFileSet(), filepath.Join(directory, name), nil, 0)
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		for _, declaration := range file.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || !function.Name.IsExported() || function.Type.Results == nil {
				continue
			}
			for _, result := range function.Type.Results.List {
				resultName := compositionResultType(result.Type)
				allowed, guarded := protected[resultName]
				if !guarded {
					continue
				}
				if _, ok := allowed[function.Name.Name]; !ok {
					t.Errorf("internal/inference/%s exports direct %s constructor %s", name, resultName, function.Name.Name)
				}
			}
		}
	}
}

func compositionResultType(expression ast.Expr) string {
	switch value := expression.(type) {
	case *ast.Ident:
		return value.Name
	case *ast.StarExpr:
		return compositionResultType(value.X)
	case *ast.SelectorExpr:
		return value.Sel.Name
	default:
		return ""
	}
}

// TestInferenceDecodeOutputOwnedByPolicy holds inference's decode paths to
// the compiled output policy: no other file switches on the output mode or
// retains a displaced decode wrapper.
func TestInferenceDecodeOutputOwnedByPolicy(t *testing.T) {
	t.Parallel()
	directory := filepath.Join(repositoryRoot(t), "internal", "inference")
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") || name == "decode_output.go" {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(directory, name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		ast.Inspect(file, func(node ast.Node) bool {
			switch statement := node.(type) {
			case *ast.SwitchStmt:
				if selector, ok := statement.Tag.(*ast.SelectorExpr); ok && selector.Sel.Name == "mode" {
					t.Errorf("%s retains a decode-output mode switch", name)
				}
			case *ast.FuncDecl:
				if statement.Name.Name == "forwardDeviceCachedGreedyBatchLocked" ||
					statement.Name.Name == "forwardDeviceCachedTopKBatchLocked" {
					t.Errorf("%s retains displaced decode wrapper %s", name, statement.Name.Name)
				}
			}
			return true
		})
	}
}

// TestRecipePlanOwnsAllTrainingExecution holds training execution to the
// recipe workflow: the commands call no trainer directly, and exactly the
// declared callers execute the workflow.
func TestRecipePlanOwnsAllTrainingExecution(t *testing.T) {
	t.Parallel()
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
	snapshot, err := repoanalysis.DiscoverGo(repositoryRoot(t), "cmd", "internal/server")
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

// TestCapabilityEpisodeProjectionAuthorityRatchet holds every dataset
// projection to its runrecord adapter: no other production file calls the
// dataset package's projection constructors.
func TestCapabilityEpisodeProjectionAuthorityRatchet(t *testing.T) {
	t.Parallel()
	forbidden := map[string]bool{
		"CompileCapabilityEpisodeProjection":  true,
		"LoadCapabilityEpisodeProjection":     true,
		"ParseCapabilityEpisodeProjection":    true,
		"CompileInteractionArcProjection":     true,
		"LoadInteractionArcProjection":        true,
		"LoadInteractionArc":                  true,
		"LoadInteractionArcMeasurement":       true,
		"LoadInteractionArcMeasurementPolicy": true,
		"LoadInteractionArcSelection":         true,
		"NewInteractionArcMeasurement":        true,
		"NewInteractionCallReference":         true,
		"SelectInteractionArcsOnce":           true,
	}
	snapshot, err := repoanalysis.DiscoverGo(repositoryRoot(t), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	for _, source := range snapshot.Files {
		if source.Test || source.Path == "internal/runrecord/dataset_projection_adapter.go" {
			continue
		}
		parsed, parseErr := source.Syntax()
		if parseErr != nil {
			t.Fatal(parseErr)
		}
		aliases := map[string]bool{}
		for _, spec := range parsed.Imports {
			importPath, unquoteErr := strconv.Unquote(spec.Path.Value)
			if unquoteErr != nil || importPath != "overgo/internal/dataset" {
				continue
			}
			name := "dataset"
			if spec.Name != nil {
				name = spec.Name.Name
			}
			aliases[name] = true
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			call, isCall := node.(*ast.CallExpr)
			if !isCall {
				return true
			}
			name := ""
			switch function := call.Fun.(type) {
			case *ast.SelectorExpr:
				identifier, isIdentifier := function.X.(*ast.Ident)
				if isIdentifier && aliases[identifier.Name] {
					name = function.Sel.Name
				}
			case *ast.Ident:
				if parsed.Name.Name == "dataset" || aliases["."] {
					name = function.Name
				}
			}
			if forbidden[name] {
				t.Errorf("dataset projection authority bypass at %s:%d: %s", source.Path, source.Line(call.Pos()), name)
			}
			return true
		})
	}
}

// TestScratchRuntimeOwnsNoPrimitive holds the scratch model's production
// runtime: no compiled file owns an optimizer or host-state primitive, no
// oracle or host runtime compiles into production, and the resident Step
// runs the program instead of hard-coded forward and backward passes.
func TestScratchRuntimeOwnsNoPrimitive(t *testing.T) {
	t.Parallel()
	directory := filepath.Join(repositoryRoot(t), "internal", "scratchmodel")
	pkg, err := build.Default.ImportDir(directory, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range pkg.GoFiles {
		if strings.Contains(name, "oracle") || strings.HasPrefix(name, "host_") {
			t.Fatalf("oracle runtime compiled into production: %s", name)
		}
		data, err := os.ReadFile(filepath.Join(directory, name))
		if err != nil {
			t.Fatal(err)
		}
		text := string(data)
		if strings.Contains(text, "func muonUpdate") || strings.Contains(text, "type value struct") || strings.Contains(text, "type hostState") {
			t.Fatalf("production scratch runtime owns forbidden primitive in %s", name)
		}
	}
	file, err := parser.ParseFile(token.NewFileSet(), filepath.Join(directory, "resident_cuda_windows.go"), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	foundProgramRun, foundHardCoded := false, false
	for _, declaration := range file.Decls {
		function, ok := declaration.(*ast.FuncDecl)
		if !ok || function.Name.Name != "Step" || function.Recv == nil {
			continue
		}
		ast.Inspect(function.Body, func(node ast.Node) bool {
			selector, ok := node.(*ast.SelectorExpr)
			if !ok {
				return true
			}
			foundProgramRun = foundProgramRun || selector.Sel.Name == "Run"
			foundHardCoded = foundHardCoded || selector.Sel.Name == "forward" || selector.Sel.Name == "backward"
			return true
		})
	}
	if !foundProgramRun || foundHardCoded {
		t.Fatalf("resident Step program authority=%t hard-coded forward/backward=%t", foundProgramRun, foundHardCoded)
	}
}
