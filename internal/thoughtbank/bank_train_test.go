package thoughtbank

import (
	"go/ast"
	"go/parser"
	"go/token"
	"math/rand"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/checkpointtest"
	"overgo/internal/trainingprogram"
)

// TestBankTrainerCheckpointRoundTripIsBitExact steps the bank trainer once
// and holds the checkpoint it publishes to its bank tensors and optimizer
// state bit for bit.
func TestBankTrainerCheckpointRoundTripIsBitExact(t *testing.T) {
	rng := rand.New(rand.NewSource(20260930))
	weights, input, memory, rows, slots := newFastWeightBankFixture(rng, false)
	config, err := trainingprogram.BuiltinOptimizerPolicy().Config(len(weights.FWA) + len(weights.FWB) + len(weights.FWO) + len(weights.NormWeight))
	if err != nil {
		t.Fatal(err)
	}
	trainer, err := weights.NewTrainer(config)
	if err != nil {
		t.Fatal(err)
	}
	defer trainer.Close()
	example := BankTrainingExample{Input: input, Memory: memory, Target: randSlice(rng, len(input), 1), Rows: rows, Slots: slots}
	before, err := trainer.Loss(example)
	if err != nil {
		t.Fatal(err)
	}
	result, err := trainer.Step(example)
	if err != nil || result.Loss != before || !(result.GradientL2 > 0) {
		t.Fatalf("step = %+v (%v), want the loss before it (%g) and a gradient", result, err, before)
	}
	program, err := trainingprogram.CompileObjectiveProgram(trainingprogram.ObjectiveTokenPrediction, nil, trainer.OptimizerPlan())
	if err != nil {
		t.Fatal(err)
	}
	checkpointtest.RequireRoundTrip(t, program, trainer, map[string][]float32{
		"bank.fw_a": weights.FWA, "bank.fw_b": weights.FWB, "bank.fw_o": weights.FWO, "bank.norm": weights.NormWeight,
	})
}

// TestBankTrainerHasAUserPath holds the bank trainer to a user path: a
// cmd/train route runs a function that loads a thoughtbank checkpoint,
// builds its bank's trainer and steps it on bank training examples.
func TestBankTrainerHasAUserPath(t *testing.T) {
	directory := filepath.Join("..", "..", "cmd", "train")
	entries, err := os.ReadDir(directory)
	if err != nil {
		t.Fatal(err)
	}
	files := token.NewFileSet()
	routes := map[string]string{}
	reaching := map[string]bool{}
	for _, entry := range entries {
		name := entry.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(files, filepath.Join(directory, name), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		imports := false
		for _, spec := range parsed.Imports {
			imports = imports || spec.Path.Value == strconv.Quote("overgo/internal/thoughtbank")
		}
		ast.Inspect(parsed, func(node ast.Node) bool {
			switch node := node.(type) {
			case *ast.ValueSpec:
				if len(node.Names) == 1 && node.Names[0].Name == "trainRoutes" && len(node.Values) == 1 {
					for _, element := range node.Values[0].(*ast.CompositeLit).Elts {
						pair := element.(*ast.KeyValueExpr)
						route, err := strconv.Unquote(pair.Key.(*ast.BasicLit).Value)
						if err != nil {
							t.Fatal(err)
						}
						routes[pair.Value.(*ast.Ident).Name] = route
					}
				}
			case *ast.FuncDecl:
				if imports && node.Body != nil {
					reaching[node.Name.Name] = calls(node.Body, "thoughtbank", "LoadCheckpoint") && calls(node.Body, "", "NewTrainer") &&
						names(node.Body, "thoughtbank", "BankTrainingExample")
				}
			}
			return true
		})
	}
	for function, route := range routes {
		if reaching[function] {
			t.Logf("route %s (%s) trains a thoughtbank checkpoint's bank", route, function)
			return
		}
	}
	t.Fatalf("no cmd/train route of %v loads a thoughtbank checkpoint and steps its bank trainer", routes)
}

// calls reports whether body calls name, as qualifier.name when qualifier is
// set and as a method of any value otherwise.
func calls(body ast.Node, qualifier, name string) (found bool) {
	ast.Inspect(body, func(node ast.Node) bool {
		if call, ok := node.(*ast.CallExpr); ok {
			if selector, ok := call.Fun.(*ast.SelectorExpr); ok && selector.Sel.Name == name {
				owner, isPackage := selector.X.(*ast.Ident)
				found = found || qualifier == "" || isPackage && owner.Name == qualifier
			}
		}
		return !found
	})
	return found
}

// names reports whether body names qualifier.name.
func names(body ast.Node, qualifier, name string) (found bool) {
	ast.Inspect(body, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok && selector.Sel.Name == name {
			owner, isPackage := selector.X.(*ast.Ident)
			found = found || isPackage && owner.Name == qualifier
		}
		return !found
	})
	return found
}
