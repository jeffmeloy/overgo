package gate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/webuilane"
)

// The ratchets below read other packages' test sources. They live here, in
// the package that already parses Go, so the measured suites they hold
// import no Go parser themselves: a package whose tests parse Go is a
// source reader the gate selects on every Go change.

// testFunctions parses the test files of the repository directories given
// and indexes their top-level functions by name.
func testFunctions(t *testing.T, dirs ...string) (*token.FileSet, map[string]*ast.FuncDecl) {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	fileSet := token.NewFileSet()
	functions := map[string]*ast.FuncDecl{}
	for _, dir := range dirs {
		entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(dir)))
		if err != nil {
			t.Fatal(err)
		}
		for _, entry := range entries {
			if !strings.HasSuffix(entry.Name(), "_test.go") {
				continue
			}
			parsed, err := parser.ParseFile(fileSet, filepath.Join(root, filepath.FromSlash(dir), entry.Name()), nil, 0)
			if err != nil {
				t.Fatal(err)
			}
			for _, declaration := range parsed.Decls {
				function, ok := declaration.(*ast.FuncDecl)
				if !ok || function.Body == nil {
					continue
				}
				// A method joins under its bare name unless a function holds it.
				if _, held := functions[function.Name.Name]; function.Recv == nil || !held {
					functions[function.Name.Name] = function
				}
			}
		}
	}
	return fileSet, functions
}

// reachResolver reports whether a function names a marked identifier or
// package selector, directly or through the functions it calls; via records
// the step each function took, for the failure message.
func reachResolver(functions map[string]*ast.FuncDecl, marked map[string]bool) (func(name string) bool, map[string]string) {
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
				if pkg, ok := typed.X.(*ast.Ident); ok && marked[pkg.Name+"."+typed.Sel.Name] {
					via[name] = pkg.Name + "." + typed.Sel.Name
					found = true
				}
			case *ast.Ident:
				if marked[typed.Name] || (typed.Name != name && resolve(typed.Name, path)) {
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
	return func(name string) bool { return resolve(name, map[string]bool{}) }, via
}

// reachChain renders the call steps from a test to the marked work.
func reachChain(via map[string]string, name string) string {
	chain := name
	for step := via[name]; step != "" && chain != step; step = via[step] {
		chain += " -> " + step
	}
	return chain
}

// TestAudioMeasurementOwner holds the measured CPU speech child runs to one
// test: every audioparity test that reaches the measurement fixture,
// directly or through a helper of its files, is the fitness acceptance, and
// the profile child re-executes that same test. A second owner would run
// the same held-out measurement, repeats or profile again in parallel, as
// the envelope, simplification and benchmark-coverage acceptances did.
func TestAudioMeasurementOwner(t *testing.T) {
	t.Parallel()
	fileSet, functions := testFunctions(t, "internal/audioparity")
	reaches, _ := reachResolver(functions, map[string]bool{"newAudioMeasurementFixture": true})
	const owner = "TestAudioResourceFitnessAcceptance"
	var owners []string
	for name := range functions {
		if strings.HasPrefix(name, "Test") && reaches(name) {
			owners = append(owners, name)
		}
	}
	if len(owners) != 1 || owners[0] != owner {
		t.Fatalf("measured child runs are owned by %v, want %s alone", owners, owner)
	}
	if !strings.Contains(sourceOf(t, fileSet, functions["profile"]), "audioProfileOwner") || !strings.Contains(sourceOf(t, fileSet, functions[owner]), "audioProfileRequestEnvironment") {
		t.Fatalf("the profile child is not re-executed as %s", owner)
	}
}

// TestNativeFixturePublishedOnce holds the native speech fixture to one
// publication per capability-runtime test binary: exactly one test reaches
// PublishNative, directly or through a helper of its files, and the
// fresh-process restart child re-executes that same test, so the registered
// model and its captures are copied into a store once, not once per test.
func TestNativeFixturePublishedOnce(t *testing.T) {
	t.Parallel()
	fileSet, functions := testFunctions(t, "internal/capabilityruntime")
	reaches, _ := reachResolver(functions, map[string]bool{"speechrecognitiontest.PublishNative": true})
	const owner = "TestNativeASRSessionAcceptance"
	var publishers []string
	for name := range functions {
		if strings.HasPrefix(name, "Test") && reaches(name) {
			publishers = append(publishers, name)
		}
	}
	if len(publishers) != 1 || publishers[0] != owner {
		t.Fatalf("the native fixture is published by %v, want %s alone", publishers, owner)
	}
	if !strings.Contains(sourceOf(t, fileSet, functions[owner]), "-test.run=^"+owner+"$") {
		t.Fatalf("%s does not re-execute itself as the fresh-process restart child", owner)
	}
}

// browserModelWork names the calls that build, serve, download or hash a
// model in the browser tests; a page acceptance may reach none of them.
var browserModelWork = map[string]bool{
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
	fileSet, functions := testFunctions(t, "internal/server", "internal/webuilane", "internal/audioparity")
	reaches, via := reachResolver(functions, browserModelWork)
	pageTests, journeys := 0, 0
	for name, function := range functions {
		switch {
		case strings.HasPrefix(name, webuilane.BrowserTestPrefix):
			pageTests++
			if reaches(name) {
				t.Errorf("%s runs in the page lane yet reaches model work (%s); name it %s* and let the serving owners run it", name, reachChain(via, name), webuilane.ModelJourneyPrefix)
			}
		case strings.HasPrefix(name, webuilane.ModelJourneyPrefix):
			journeys++
			if !reaches(name) {
				t.Errorf("%s is a journey that reaches no model work; a page acceptance belongs under %s*", name, webuilane.BrowserTestPrefix)
			}
			if body := sourceOf(t, fileSet, function); !strings.Contains(body, "webuilane.ModelJourneyEnvironment") && !strings.Contains(body, "ModelJourneyEnvironment)") {
				t.Errorf("%s does not skip outside the lane's journey mode", name)
			}
		}
	}
	if pageTests == 0 || journeys == 0 {
		t.Fatalf("browser tests: %d page acceptances, %d model journeys", pageTests, journeys)
	}
	t.Logf("browser lane: %d page acceptances run against retained receipts; %d model journeys run with the serving owners", pageTests, journeys)
}

// sourceOf returns a function's body text.
func sourceOf(t *testing.T, fileSet *token.FileSet, function *ast.FuncDecl) string {
	t.Helper()
	if function == nil {
		t.Fatal("function is not declared")
	}
	text, err := os.ReadFile(fileSet.Position(function.Pos()).Filename)
	if err != nil {
		t.Fatal(err)
	}
	return string(text[fileSet.Position(function.Body.Lbrace).Offset:fileSet.Position(function.Body.Rbrace).Offset])
}
