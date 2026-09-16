package webuilane

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// modelWork names the calls that build, serve, download or hash a model in
// the browser tests; a page acceptance may reach none of them.
var modelWork = map[string]bool{
	"prepareBrowserJourney": true, "prepareBrowserJourneyStore": true, "smallestDeclaredProjector": true,
	"laneHubServer": true, "modelswap.ServerLauncher": true, "newLiveASRFixture": true,
}

// TestBrowserLanePageOnly holds the browser lane to page work: every test
// the lane runs by its prefix reaches no model build, serve, download or
// hash, directly or through a helper of its test files, and every journey
// that does is named by the journey prefix and skips outside the lane's
// journey mode, where the serving owners run it.
func TestBrowserLanePageOnly(t *testing.T) {
	t.Parallel()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fileSet := token.NewFileSet()
	functions := map[string]*ast.FuncDecl{}
	for _, dir := range []string{"internal/server", "internal/webuilane", "internal/audioparity"} {
		entries, err := os.ReadDir(filepath.Join(root, dir))
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if !strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			parsed, err := parser.ParseFile(fileSet, filepath.Join(root, dir, entry.Name()), nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, declaration := range parsed.Decls {
				if function, ok := declaration.(*ast.FuncDecl); ok && function.Recv == nil {
					functions[function.Name.Name] = function
				}
			}
		}
	}
	// A function reaches model work when its body names it or calls a
	// function of these files that does; the closure follows the calls.
	reaches := map[string]bool{}
	via := map[string]string{}
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
			switch typed := node.(type) {
			case *ast.SelectorExpr:
				if pkg, ok := typed.X.(*ast.Ident); ok && modelWork[pkg.Name+"."+typed.Sel.Name] {
					found = true
				}
			case *ast.Ident:
				if modelWork[typed.Name] || (typed.Name != name && resolve(typed.Name, path)) {
					via[name] = typed.Name
					found = true
				}
			}
			return !found
		})
		delete(path, name)
		reaches[name] = found
		return found
	}
	pageTests, journeys := 0, 0
	for name, function := range functions {
		switch {
		case strings.HasPrefix(name, BrowserTestPrefix):
			pageTests++
			if resolve(name, map[string]bool{}) {
				chain := name
				for step := via[name]; step != "" && chain != step; step = via[step] {
					chain += " -> " + step
				}
				t.Errorf("%s runs in the page lane yet reaches model work (%s); name it %s* and let the serving owners run it", name, chain, ModelJourneyPrefix)
			}
		case strings.HasPrefix(name, ModelJourneyPrefix):
			journeys++
			if !resolve(name, map[string]bool{}) {
				t.Errorf("%s is a journey that reaches no model work; a page acceptance belongs under %s*", name, BrowserTestPrefix)
			}
			source := fileSet.Position(function.Pos()).Filename
			text, err := os.ReadFile(source)
			if err != nil {
				t.Fatal(err)
			}
			body := string(text[fileSet.Position(function.Body.Lbrace).Offset:fileSet.Position(function.Body.Rbrace).Offset])
			if !strings.Contains(body, "webuilane.ModelJourneyEnvironment") && !strings.Contains(body, "ModelJourneyEnvironment)") {
				t.Errorf("%s does not skip outside the lane's journey mode", name)
			}
		}
	}
	if pageTests == 0 || journeys == 0 {
		t.Fatalf("browser tests: %d page acceptances, %d model journeys", pageTests, journeys)
	}
	t.Logf("browser lane: %d page acceptances run against retained receipts; %d model journeys run with the serving owners", pageTests, journeys)
}
