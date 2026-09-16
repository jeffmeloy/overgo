package audioparity

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"strings"
	"testing"
)

// TestAudioMeasurementOwner holds the measured CPU speech child runs to one
// test: every test function that reaches the measurement fixture, directly
// or through a helper of these files, is the fitness acceptance, and the
// profile child re-executes that same test. A second owner would run the
// same held-out measurement, repeats or profile again in parallel, as the
// envelope, simplification and benchmark-coverage acceptances did.
func TestAudioMeasurementOwner(t *testing.T) {
	t.Parallel()
	fileSet := token.NewFileSet()
	functions := map[string]*ast.FuncDecl{}
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(fileSet, entry.Name(), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range parsed.Decls {
			if function, ok := declaration.(*ast.FuncDecl); ok && function.Body != nil {
				functions[function.Name.Name] = function
			}
		}
	}
	const fixture = "newAudioMeasurementFixture"
	reaches := map[string]bool{}
	var resolve func(name string, path map[string]bool) bool
	resolve = func(name string, path map[string]bool) bool {
		if done, known := reaches[name]; known {
			return done
		}
		function, declared := functions[name]
		if !declared || path[name] {
			return false
		}
		path[name] = true
		found := false
		ast.Inspect(function.Body, func(node ast.Node) bool {
			identifier, ok := node.(*ast.Ident)
			if ok && (identifier.Name == fixture || identifier.Name != name && resolve(identifier.Name, path)) {
				found = true
			}
			return !found
		})
		delete(path, name)
		reaches[name] = found
		return found
	}
	var owners []string
	for name := range functions {
		if strings.HasPrefix(name, "Test") && resolve(name, map[string]bool{}) {
			owners = append(owners, name)
		}
	}
	if len(owners) != 1 || owners[0] != audioProfileOwner {
		t.Fatalf("measured child runs are owned by %v, want %s alone", owners, audioProfileOwner)
	}
	if _, declared := functions[audioProfileOwner]; !declared {
		t.Fatalf("the profile child's owner %s is not declared", audioProfileOwner)
	}
}
