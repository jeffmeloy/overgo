package workflowruntime

import (
	"errors"
	"go/ast"
	"go/parser"
	"go/token"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestResumePathRecordIsWhole holds the record of the paths that resume or reuse finished work to the tree.
//   - path = one of the three the test names, each with its record, identity, retry rule, counted declarations, readers and tests
//   - counted declaration = a production function, method or package variable its file declares, counted in one path only
//   - size = the syntax nodes of a path's counted declarations, counted as a commit's checks count production nodes
//   - whole file = a file the record says holds its path alone, with the nodes of the file
//   - reader = a production declaration that calls one of the names its path is entered by
//   - place = a file and a name that file declares, wherever the record cites one
//   - repair = one of the five the plan row names, on a mapped path or on files the tree no longer holds
//   - repair on a mapped path names declarations of that path, one of them counted
//   - site = a place that reuses work outside the four paths, with places in the tree
//   - connection = two of the record's paths or sites, with places in the tree
//   - figure = what the entries give, for every figure the record states and no figure besides
//   - closing = the largest path as the sizes give it, the closest path as a mapped path, a yes or a no on a merge row with its reason, and when the record retires
//   - a key stated twice in one entry is refused, because the later line would replace the earlier one unseen
func TestResumePathRecordIsWhole(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "verification", "resume-paths.txt"))
	if err != nil {
		t.Fatal(err)
	}

	// Four parts of the record hold entries, each opened by its own key, and two parts hold single fields.
	const pathsPart, repairsPart, connectionsPart, sitesPart, figuresPart, closingPart = "== paths ==", "== repairs ==", "== connections ==", "== sites ==", "== figures ==", "== closing =="
	openers := map[string]string{pathsPart: "path", repairsPart: "repair", connectionsPart: "connection", sitesPart: "site"}
	entries := map[string][]map[string]string{}
	figures, closing := map[string]string{}, map[string]string{}
	var part string
	for line := range strings.Lines(string(raw)) {
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, "== ") {
			part = line
			continue
		}
		key, value, found := strings.Cut(line, ": ")
		if !found || strings.Contains(key, " ") {
			continue
		}
		var held map[string]string
		switch opener, listed := openers[part]; {
		case part == figuresPart:
			held = figures
		case part == closingPart:
			held = closing
		case !listed:
			continue
		case key == opener:
			held = map[string]string{}
			entries[part] = append(entries[part], held)
		case len(entries[part]) == 0:
			t.Fatalf("the record states %s in %s before any %s", key, part, opener)
		default:
			held = entries[part][len(entries[part])-1]
		}
		if _, twice := held[key]; twice {
			t.Fatalf("the record states %s twice in one entry of %s", key, part)
		}
		held[key] = value
	}
	count := func(what, text string) int {
		t.Helper()
		value, err := strconv.Atoi(text)
		if err != nil || value < 0 {
			t.Fatalf("the record's %s is %q (%v), want a count", what, text, err)
		}
		return value
	}

	// Each file the record cites is parsed once, with its comments, as a commit's checks parse it.
	parsed := map[string]*ast.File{}
	syntax := func(what, file string) *ast.File {
		t.Helper()
		if held, read := parsed[file]; read {
			return held
		}
		held, err := parser.ParseFile(token.NewFileSet(), filepath.Join(root, filepath.FromSlash(file)), nil, parser.ParseComments)
		if err != nil {
			t.Fatalf("the record's %s cites %s, which the tree does not hold as Go source: %v", what, file, err)
		}
		parsed[file] = held
		return held
	}
	// size counts the syntax nodes beneath a node, the node among them.
	size := func(node ast.Node) (nodes int) {
		ast.Inspect(node, func(each ast.Node) bool {
			if each != nil {
				nodes++
			}
			return true
		})
		return nodes
	}
	// receiver names the type a method is declared on, without its pointer and its type parameters.
	receiver := func(function *ast.FuncDecl) string {
		if function.Recv == nil || len(function.Recv.List) == 0 {
			return ""
		}
		for expression := function.Recv.List[0].Type; ; {
			switch typed := expression.(type) {
			case *ast.StarExpr:
				expression = typed.X
			case *ast.IndexExpr:
				expression = typed.X
			case *ast.IndexListExpr:
				expression = typed.X
			case *ast.Ident:
				return typed.Name
			default:
				return ""
			}
		}
	}
	// declared finds what a place names, and says whether it is a function, a method or a package variable.
	//   - a declaration written alone carries its doc comment, so the whole declaration is the node
	//   - a declaration written in a group is its own line of the group
	place := regexp.MustCompile(`^([A-Za-z0-9_./-]+\.go)#(?:([A-Za-z_][A-Za-z0-9_]*)\.)?([A-Za-z_][A-Za-z0-9_]*)$`)
	declared := func(what, cited string) (ast.Node, bool) {
		t.Helper()
		match := place.FindStringSubmatch(cited)
		if match == nil {
			t.Fatalf("the record's %s cites %q, want file.go#Name or file.go#Type.Name", what, cited)
		}
		file, owner, name := match[1], match[2], match[3]
		for _, each := range syntax(what, file).Decls {
			switch typed := each.(type) {
			case *ast.FuncDecl:
				if typed.Name.Name == name && receiver(typed) == owner {
					return typed, true
				}
			case *ast.GenDecl:
				for _, spec := range typed.Specs {
					var names []*ast.Ident
					switch spec := spec.(type) {
					case *ast.TypeSpec:
						names = []*ast.Ident{spec.Name}
					case *ast.ValueSpec:
						names = spec.Names
					}
					if owner != "" || !slices.ContainsFunc(names, func(ident *ast.Ident) bool { return ident.Name == name }) {
						continue
					}
					if typed.Lparen.IsValid() {
						return spec, typed.Tok == token.VAR
					}
					return typed, typed.Tok == token.VAR
				}
			}
		}
		t.Fatalf("the record's %s cites %s, which %s does not declare", what, cited, file)
		return nil, false
	}
	// places reads a list of places and holds each to the tree.
	const absent = "none"
	places := func(what, list string) []string {
		t.Helper()
		if list == absent {
			return nil
		}
		cited := strings.Fields(list)
		if len(cited) == 0 {
			t.Fatalf("the record's %s names no place, want places or %q", what, absent)
		}
		for _, each := range cited {
			declared(what, each)
		}
		return cited
	}
	// stated holds a statement to the places after its bar.
	//   - needsPlaces refuses a statement with no bar
	const bar = " | "
	stated := func(what, value string, needsPlaces bool) {
		t.Helper()
		statement, cited, barred := strings.Cut(value, bar)
		if strings.TrimSpace(statement) == "" || needsPlaces && !barred {
			t.Fatalf("the record's %s is %q, want a statement, a bar and places in the tree", what, value)
		}
		if barred && len(places(what, cited)) == 0 {
			t.Fatalf("the record's %s cites no place after its bar", what)
		}
	}
	// calls says whether a declaration calls one of the names, by the name alone, through a selector or with type arguments.
	calls := func(node ast.Node, names []string) (found bool) {
		ast.Inspect(node, func(each ast.Node) bool {
			call, isCall := each.(*ast.CallExpr)
			if !isCall {
				return !found
			}
			target := call.Fun
			if index, generic := target.(*ast.IndexExpr); generic {
				target = index.X
			} else if index, generic := target.(*ast.IndexListExpr); generic {
				target = index.X
			}
			switch typed := target.(type) {
			case *ast.Ident:
				found = found || slices.Contains(names, typed.Name)
			case *ast.SelectorExpr:
				found = found || slices.Contains(names, typed.Sel.Name)
			}
			return !found
		})
		return found
	}
	const testSuffix = "_test.go"
	fileOf := func(cited string) string { file, _, _ := strings.Cut(cited, "#"); return file }

	// anchors names, for each path the plan row resume-census asks for, the declarations that define it.
	anchors := map[string][]string{
		"workflow-replay":     {"internal/workflowruntime/runtime.go#Runtime.Replay", "internal/workflowruntime/runtime.go#Runtime.recoverStage"},
		"measurement-cells":   {"cmd/validate/cell_credit.go#cellKey"},
		"trainer-checkpoints": {"internal/trainingprogram/checkpoint.go#ValidateResume", "internal/trainingworkflow/checkpoint_steps.go#restoreRun"},
	}
	pathKeys := []string{"path", "record", "identity", "retry", "code", "nodes", "whole-files", "whole-nodes", "beside", "entered-by", "readers", "reached-from", "held-by"}
	required := []string{"record", "identity", "retry", "code", "nodes", "whole-files", "whole-nodes", "beside", "entered-by", "readers", "held-by"}
	computed := map[string]int{"readers-shared": 0}
	counted, reads := map[string]string{}, map[string]string{}
	code, beside, sizes := map[string][]string{}, map[string][]string{}, map[string]int{}
	for _, entry := range entries[pathsPart] {
		name := entry["path"]
		if anchors[name] == nil || code[name] != nil {
			t.Fatalf("the record maps the path %q, want each of the three the test names, once", name)
		}
		for key, value := range entry {
			if !slices.Contains(pathKeys, key) && !strings.HasPrefix(key, "identity-") && !strings.HasPrefix(key, "retry-") {
				t.Fatalf("the record's path %s states %s, want one of %s, or a further identity or retry line", name, key, strings.Join(pathKeys, ", "))
			}
			if key == "record" || key == "reached-from" || strings.HasPrefix(key, "identity") || strings.HasPrefix(key, "retry") {
				stated(name+" "+key, value, true)
			}
		}
		for _, key := range required {
			if strings.TrimSpace(entry[key]) == "" {
				t.Fatalf("the record's path %s states no %s", name, key)
			}
		}
		// A path's nodes are the sum of its counted declarations, each counted once and in one path.
		code[name] = places(name+" code", entry["code"])
		nodes := 0
		for _, cited := range code[name] {
			node, countable := declared(name+" code", cited)
			if !countable || strings.HasSuffix(fileOf(cited), testSuffix) {
				t.Fatalf("the record's path %s counts %s, want a production function, method or package variable", name, cited)
			}
			if earlier := counted[cited]; earlier != "" {
				t.Fatalf("the record's path %s counts %s, which the path %s counts already", name, cited, earlier)
			}
			counted[cited] = name
			nodes += size(node)
		}
		for _, anchor := range anchors[name] {
			if !slices.Contains(code[name], anchor) {
				t.Fatalf("the record's path %s does not count %s, which defines it", name, anchor)
			}
		}
		if count(name+" nodes", entry["nodes"]) != nodes {
			t.Errorf("the record's path %s states %s nodes, and its %d declarations hold %d in the tree", name, entry["nodes"], len(code[name]), nodes)
		}
		whole := 0
		if entry["whole-files"] != absent {
			for file := range strings.FieldsSeq(entry["whole-files"]) {
				if !slices.ContainsFunc(code[name], func(cited string) bool { return fileOf(cited) == file }) {
					t.Fatalf("the record's path %s calls %s its own whole, and counts no declaration of it", name, file)
				}
				whole += size(syntax(name+" whole-files", file))
			}
		}
		if count(name+" whole-nodes", entry["whole-nodes"]) != whole {
			t.Errorf("the record's path %s states %s nodes in its whole files, and the tree holds %d there", name, entry["whole-nodes"], whole)
		}
		sizes[name] = nodes
		computed["nodes"] += nodes
		computed["declarations-"+name] = len(code[name])
		beside[name] = places(name+" beside", entry["beside"])

		// A reader is production code that enters the path by one of its names.
		entered := strings.Fields(entry["entered-by"])
		readers := places(name+" readers", entry["readers"])
		for _, cited := range readers {
			if node, _ := declared(name+" readers", cited); strings.HasSuffix(fileOf(cited), testSuffix) || !calls(node, entered) {
				t.Fatalf("the record's path %s names the reader %s, which is test code or calls none of %s", name, cited, strings.Join(entered, ", "))
			}
			if earlier := reads[cited]; earlier != "" {
				computed["readers-shared"]++
			}
			reads[cited] = name
		}
		if len(readers) == 0 {
			t.Fatalf("the record's path %s names no reader", name)
		}
		computed["readers-"+name] = len(readers)
		// A path is held by tests their files declare.
		tests := places(name+" held-by", entry["held-by"])
		for _, cited := range tests {
			if _, function := declared(name+" held-by", cited); !function || !strings.HasSuffix(fileOf(cited), testSuffix) || !strings.HasPrefix(strings.TrimPrefix(cited, fileOf(cited)+"#"), "Test") {
				t.Fatalf("the record's path %s is held by %s, want a test a test file declares", name, cited)
			}
		}
		if len(tests) == 0 {
			t.Fatalf("the record's path %s names no test that holds it", name)
		}
	}
	if t.Failed() {
		t.FailNow()
	}
	if len(code) != len(anchors) {
		t.Fatalf("the record maps %d paths, want the %d the test names", len(code), len(anchors))
	}
	computed["paths"] = len(code)

	// repairs names the five commits the plan row resume-census names.
	repairs := []string{"8cfa15418", "8988728ae", "2542eb677", "78534ee41", "2ae868219"}
	const gone = "gone"
	seen := map[string]bool{}
	for name := range anchors {
		computed["repairs-"+name] = 0
	}
	computed["repairs-"+gone] = 0
	for _, entry := range entries[repairsPart] {
		name, path := entry["repair"], entry["path"]
		if !slices.Contains(repairs, name) || seen[name] || strings.TrimSpace(entry["added"]) == "" {
			t.Fatalf("the record's repair %q is not one of %s named once with what it added", name, strings.Join(repairs, ", "))
		}
		seen[name] = true
		computed["repairs"]++
		computed["repairs-"+path]++
		if outside, given := entry["outside"]; given {
			places("repair "+name+" outside", outside)
		}
		if path == gone {
			// A file the record calls gone is one the tree does not hold.
			directory, err := os.Stat(filepath.Join(root, filepath.FromSlash(entry["from"])))
			files := strings.Fields(entry["files"])
			if err != nil || !directory.IsDir() || len(files) == 0 || strings.TrimSpace(entry["was"]) == "" || entry["in"] != "" {
				t.Fatalf("the record's repair %s is on a path that is gone, from %q (%v) with files %q: want a directory of the tree, the files it no longer holds, what the path was, and no place in a mapped path", name, entry["from"], err, entry["files"])
			}
			for _, file := range files {
				if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(entry["from"]), file)); !errors.Is(err, fs.ErrNotExist) {
					t.Fatalf("the record says repair %s changed %s, which is gone, and the tree holds it in %s (%v): map its path", name, file, entry["from"], err)
				}
			}
			continue
		}
		if code[path] == nil {
			t.Fatalf("the record's repair %s names the path %q, want a mapped path or %q", name, path, gone)
		}
		inCode := false
		for _, cited := range places("repair "+name+" in", entry["in"]) {
			if !slices.Contains(code[path], cited) && !slices.Contains(beside[path], cited) {
				t.Fatalf("the record's repair %s names %s, which the path %s neither counts nor names beside it", name, cited, path)
			}
			inCode = inCode || slices.Contains(code[path], cited)
		}
		if !inCode {
			t.Fatalf("the record's repair %s names no declaration the path %s counts", name, path)
		}
	}
	if len(seen) != len(repairs) {
		t.Fatalf("the record holds %d repairs, want the %d the plan row names", len(seen), len(repairs))
	}

	// A site is named once, apart from the mapped paths, with a place in the tree.
	sites := map[string]bool{}
	for _, entry := range entries[sitesPart] {
		name := entry["site"]
		if name == "" || sites[name] || anchors[name] != nil || strings.TrimSpace(entry["read"]) == "" {
			t.Fatalf("the record's site %q is not a site named once, apart from the mapped paths, with how far it was read", name)
		}
		sites[name] = true
		stated("site "+name+" keeps", entry["keeps"], true)
	}
	computed["sites"] = len(sites)

	// A connection lies between two of the record's paths or sites.
	for _, entry := range entries[connectionsPart] {
		left, right, two := strings.Cut(entry["connection"], " ")
		if !two || left == right || code[left] == nil && !sites[left] || code[right] == nil && !sites[right] || strings.TrimSpace(entry["basis"]) == "" {
			t.Fatalf("the record's connection %q is not between two of its paths or sites, with what it rests on", entry["connection"])
		}
		stated("connection "+entry["connection"]+" states", entry["states"], true)
	}
	computed["connections"] = len(entries[connectionsPart])

	// Each figure the record states is what the entries give, and the record states every one.
	for name, value := range figures {
		if want, given := computed[name]; !given || count(name, value) != want {
			t.Fatalf("the record states %s as %s, and its entries give %d (counted from them %t)", name, value, want, given)
		}
	}
	for name, want := range computed {
		if _, given := figures[name]; !given {
			t.Fatalf("the record leaves out %s, which its entries give as %d", name, want)
		}
	}

	// The largest path is the sizes' own, and each judgement of the closing is given.
	largest := ""
	for name, nodes := range sizes {
		if largest == "" || nodes > sizes[largest] || nodes == sizes[largest] && name < largest {
			largest = name
		}
	}
	if closing["closing-largest-path"] != largest || code[closing["closing-closest-path"]] == nil {
		t.Fatalf("the record closes with the largest path %q and the closest path %q, and its sizes give %q and want a mapped path", closing["closing-largest-path"], closing["closing-closest-path"], largest)
	}
	if answer, _, reasoned := strings.Cut(closing["closing-merge-row"], ", because "); !reasoned || answer != "yes" && answer != "no" {
		t.Fatalf("the record closes with closing-merge-row %q, want yes or no and the reason", closing["closing-merge-row"])
	}
	for _, key := range []string{"judgement-by", "closing-shared-identity", "closing-single-rule", "closing-retires"} {
		if strings.TrimSpace(closing[key]) == "" {
			t.Fatalf("the record gives no %s", key)
		}
	}
	for key, value := range closing {
		stated(key, value, false)
	}
}
