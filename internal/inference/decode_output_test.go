package inference

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"

	"overgo/internal/tensor"
)

func TestCompiledOutputPolicyOwnsAllDecodePaths(t *testing.T) {
	const (
		vocabulary     = uint32(32)
		candidateCount = uint32(4)
	)
	graph := deviceBatchGraph{
		logits: &tensor.Tensor{}, selection: &tensor.Tensor{}, candidates: &tensor.Tensor{},
	}
	tests := []struct {
		mode deviceOutputMode
		topK uint32
		want *tensor.Tensor
	}{
		{mode: deviceOutputLogits, want: graph.logits},
		{mode: deviceOutputGreedy, want: graph.selection},
		{mode: deviceOutputGreedySpan, want: graph.selection},
		{mode: deviceOutputTopK, topK: candidateCount, want: graph.candidates},
	}
	for _, test := range tests {
		plan, err := compileDeviceOutputPlan(test.mode, test.topK, vocabulary)
		if err != nil {
			t.Fatal(err)
		}
		if plan.graphOutput(graph) != test.want {
			t.Fatalf("mode %d selected the wrong graph output", test.mode)
		}
	}
	for _, invalid := range []struct {
		mode deviceOutputMode
		topK uint32
	}{
		{mode: deviceOutputLogits, topK: candidateCount},
		{mode: deviceOutputGreedySpan, topK: candidateCount},
		{mode: deviceOutputTopK},
		{mode: deviceOutputTopK, topK: vocabulary + 1},
		{mode: deviceOutputGreedySpan + 1},
	} {
		if _, err := compileDeviceOutputPlan(invalid.mode, invalid.topK, vocabulary); err == nil {
			t.Fatalf("invalid output policy accepted: %+v", invalid)
		}
	}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		name := entry.Name()
		if entry.IsDir() || !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") || name == "decode_output.go" {
			continue
		}
		file, err := parser.ParseFile(token.NewFileSet(), name, nil, 0)
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
