package codeprofile

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/gosource"
	"overgo/internal/repoanalysis"
)

func TestDeclaredInternalOperationConsumer(t *testing.T) {
	root := filepath.Join("..", "..")
	paths := []string{"cmd/train/main.go", "internal/trainingworkflow/workflow.go", "internal/trainingworkflow/audio.go"}
	base, err := repoanalysis.LoadGo(root, paths)
	if err != nil {
		t.Fatal(err)
	}
	if len(base.Files) != len(paths) {
		t.Fatal("native training dispatch source is missing")
	}
	selection, err := gosource.HostBuildSelection(root, "./cmd/train", "./internal/trainingworkflow")
	if err != nil {
		t.Fatal(err)
	}
	// The real command owns several training modes. An internal operation
	// remains a production consumer without a dedicated executable.
	const operation = "executeAudioTraining"
	const testPath = "internal/trainingworkflow/consumer_test.go"
	selection.Files[testPath] = true
	selection.Packages[testPath] = selection.Packages[paths[1]]
	declarations, edges, _, err := ProductionConsumerGraph(base, selection)
	if err != nil {
		t.Fatal(err)
	}
	var target ConsumerDeclaration
	for _, declaration := range declarations {
		if declaration.File == paths[2] && declaration.Name == operation {
			target = declaration
		}
	}
	if target.Name == "" || target.ProductionReferences == 0 || target.TestReferences != 0 || target.Boundary != "" {
		t.Fatalf("real internal operation lacks production use: %+v", target)
	}
	if !consumerContractPath(edges, "cmd/train/main.go", "main", target) {
		for _, edge := range edges {
			if edge.From.Name == "main" || edge.From.Name == "run" || edge.From.Name == "Execute" {
				t.Logf("edge %s.%s -> %s.%s (%s)", edge.From.Package, edge.From.Name, edge.To.Package, edge.To.Name, edge.Kind)
			}
		}
		t.Fatal("internal operation has no path from the real command entry")
	}
	if len(NewUnconsumedSurface(nil, []ConsumerDeclaration{target})) != 0 {
		t.Fatal("existing gate rule requires a redundant executable")
	}
	workflow, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(paths[1])))
	if err != nil {
		t.Fatal(err)
	}
	const call = "return executeAudioTraining(ctx, request)"
	if strings.Count(string(workflow), call) != 1 {
		t.Fatal("training operation dispatch changed; review its new owner")
	}
	for _, fixture := range []struct {
		name, replacement, test string
		syntacticOnly           bool
		remove                  bool
	}{
		{name: "last real consumer removed", replacement: "return Result{}, nil"},
		{name: "stale target", replacement: "return retiredAudioTraining(ctx, request)"},
		{name: "missing target", replacement: call, remove: true},
		{name: "test-only consumer", replacement: "return Result{}, nil", test: "package trainingworkflow\nfunc TestOnly() { executeAudioTraining(nil, Request{}) }\n"},
		{name: "fake registration", replacement: "_ = \"executeAudioTraining\"; return Result{}, nil"},
		{name: "disconnected caller", replacement: "return Result{}, nil", syntacticOnly: true},
	} {
		t.Run(fixture.name, func(t *testing.T) {
			contents := map[string][]byte{paths[1]: []byte(strings.Replace(string(workflow), call, fixture.replacement, 1))}
			if fixture.syntacticOnly {
				contents[paths[1]] = append(contents[paths[1]], []byte("\nfunc disconnected(ctx context.Context, request Request) (Result, error) { return executeAudioTraining(ctx, request) }\n")...)
			}
			if fixture.remove {
				contents[paths[2]] = []byte("package trainingworkflow\n")
			}
			if fixture.test != "" {
				contents[testPath] = []byte(fixture.test)
			}
			candidate, err := base.Overlay(contents)
			if err != nil {
				t.Fatal(err)
			}
			observed, graph, _, err := ProductionConsumerGraph(candidate, selection)
			if err != nil {
				t.Fatal(err)
			}
			if consumerContractPath(graph, "cmd/train/main.go", "main", target) {
				t.Fatal("invalid declaration retained the real dispatch path")
			}
			for _, declaration := range observed {
				if declaration.File != target.File || declaration.Name != target.Name {
					continue
				}
				if fixture.syntacticOnly {
					if declaration.ProductionReferences == 0 {
						t.Fatal("counterexample no longer exercises syntactic reference credit")
					}
					continue
				}
				if declaration.ProductionReferences != 0 || len(NewUnconsumedSurface(nil, []ConsumerDeclaration{declaration})) != 1 {
					t.Fatalf("unsupported operation received consumer credit: %+v", declaration)
				}
				if fixture.test != "" && declaration.TestReferences == 0 {
					t.Fatal("test-only reference lost")
				}
			}
		})
	}
}

// Follow the owner's recorded edges. This bounded contract is not a new
// production selector, dynamic-execution proof, or blanket consumer exemption.
func consumerContractPath(edges []ConsumerReference, file, entry string, target ConsumerDeclaration) bool {
	seen := map[string]bool{}
	var pending []ConsumerDeclaration
	for _, edge := range edges {
		if edge.From.File == file && edge.From.Name == entry {
			pending = append(pending, edge.From)
		}
	}
	for len(pending) != 0 {
		node := pending[0]
		pending = pending[1:]
		key := declarationIdentity(node)
		if seen[key] {
			continue
		}
		seen[key] = true
		if key == declarationIdentity(target) {
			return true
		}
		for _, edge := range edges {
			if declarationIdentity(edge.From) == key {
				pending = append(pending, edge.To)
			}
		}
	}
	return false
}
