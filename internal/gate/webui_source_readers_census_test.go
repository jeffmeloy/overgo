package gate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/plan"
)

// TestServerTestsDeclareEveryWebUISourceRead holds the server's tests to
// declaring each read of the served web UI source: every test that touches
// the embedded source, a webui/ path or a served page or asset, directly or
// through a helper, is named in the server's webuiSourceReaders inventory
// with the class that justifies reading source instead of driving the page,
// and the inventory names no test that no longer reads it. Browser legs,
// which drive the page, are not source readers.
func TestServerTestsDeclareEveryWebUISourceRead(t *testing.T) {
	t.Parallel()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	dir := filepath.Join(root, "internal", "server")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	files := token.NewFileSet()
	functions := map[string]*ast.FuncDecl{}
	var inventory []string
	var pending map[string]string
	for _, entry := range entries {
		if !strings.HasSuffix(entry.Name(), "_test.go") {
			continue
		}
		parsed, err := parser.ParseFile(files, filepath.Join(dir, entry.Name()), nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, declaration := range parsed.Decls {
			switch declaration := declaration.(type) {
			case *ast.FuncDecl:
				if declaration.Recv == nil {
					functions[declaration.Name.Name] = declaration
				}
			case *ast.GenDecl:
				for _, spec := range declaration.Specs {
					value, ok := spec.(*ast.ValueSpec)
					if !ok || len(value.Names) != len(value.Values) {
						continue
					}
					for index, name := range value.Names {
						if name.Name == "webuiSourceReaders" {
							inventory, pending = inventoryEntries(t, value.Values[index])
						}
					}
				}
			}
		}
	}
	if len(inventory) == 0 {
		t.Fatal("internal/server declares no webuiSourceReaders inventory")
	}
	// A helper that reads the source makes its callers readers.
	helpers := map[string]bool{"webuiJavaScript": true}
	for name, function := range functions {
		if !strings.HasPrefix(name, "Test") && readsWebUISource(function.Body, helpers) {
			helpers[name] = true
		}
	}
	var readers []string
	for name, function := range functions {
		if strings.HasPrefix(name, "Test") && !strings.HasPrefix(name, "TestWebUIBrowser") &&
			!strings.HasPrefix(name, "TestModelJourney") && readsWebUISource(function.Body, helpers) {
			readers = append(readers, name)
		}
	}
	for _, reader := range readers {
		if !slices.Contains(inventory, reader) {
			t.Errorf("%s reads the served web UI source but webuiSourceReaders does not declare why", reader)
		}
	}
	for _, declared := range inventory {
		if !slices.Contains(readers, declared) {
			t.Errorf("webuiSourceReaders declares %s, which no longer reads the served source", declared)
		}
	}
	// A pending needle names the open row whose leg replaces it.
	document, err := plan.Load(filepath.Join(root, "docs", "plan.json"))
	if err != nil {
		t.Fatal(err)
	}
	open := map[string]bool{}
	for _, item := range document.Items {
		open[item.ID] = item.Status == plan.StatusOpen
	}
	for test, row := range pending {
		if !open[row] {
			t.Errorf("%s is pending %s, which is not an open plan row", test, row)
		}
	}
}

// inventoryEntries answers the inventory literal's test names and, for each
// entry pending a leg row, the row its pendingLeg call names.
func inventoryEntries(t *testing.T, value ast.Expr) ([]string, map[string]string) {
	t.Helper()
	literal, ok := value.(*ast.CompositeLit)
	if !ok {
		t.Fatal("webuiSourceReaders is not a map literal")
	}
	var keys []string
	pending := map[string]string{}
	for _, element := range literal.Elts {
		pair, ok := element.(*ast.KeyValueExpr)
		if !ok {
			t.Fatal("webuiSourceReaders holds an element that is not a key and value")
		}
		name := stringLiteral(t, pair.Key)
		keys = append(keys, name)
		if call, ok := pair.Value.(*ast.CallExpr); ok {
			if function, ok := call.Fun.(*ast.Ident); ok && function.Name == "pendingLeg" && len(call.Args) == 1 {
				pending[name] = stringLiteral(t, call.Args[0])
			}
		}
	}
	return keys, pending
}

// stringLiteral answers a string literal's value.
func stringLiteral(t *testing.T, expression ast.Expr) string {
	t.Helper()
	literal, ok := expression.(*ast.BasicLit)
	if !ok || literal.Kind != token.STRING {
		t.Fatal("webuiSourceReaders names a test or a row other than by a string literal")
	}
	value, err := strconv.Unquote(literal.Value)
	if err != nil {
		t.Fatal(err)
	}
	return value
}

// servedAssetExtensions are the extensions of the page's served assets.
var servedAssetExtensions = []string{".js", ".css", ".html"}

// readsWebUISource: the body touches the embedded web UI, a webui/ path, a
// helper that does, or requests the served page or one of its assets.
func readsWebUISource(body *ast.BlockStmt, helpers map[string]bool) bool {
	found := false
	ast.Inspect(body, func(node ast.Node) bool {
		switch node := node.(type) {
		case *ast.Ident:
			found = found || node.Name == "webuiFS" || helpers[node.Name]
		case *ast.BasicLit:
			if node.Kind != token.STRING {
				return true
			}
			// A webui path (or a path element joined to one), or a served asset's path however the request is
			// made (a local helper, a table of cases).
			text, err := strconv.Unquote(node.Value)
			found = found || err == nil && (text == "webui" || strings.Contains(text, "webui/") ||
				strings.HasPrefix(text, "/") && slices.Contains(servedAssetExtensions, path.Ext(text)))
		case *ast.CallExpr:
			identifier, ok := node.Fun.(*ast.Ident)
			if !ok || identifier.Name != "serveTestRequest" {
				return true
			}
			for _, argument := range node.Args {
				literal, ok := argument.(*ast.BasicLit)
				if !ok || literal.Kind != token.STRING {
					continue
				}
				served, err := strconv.Unquote(literal.Value)
				found = found || err == nil && (served == "/" || slices.Contains(servedAssetExtensions, path.Ext(served)))
			}
		}
		return !found
	})
	return found
}
