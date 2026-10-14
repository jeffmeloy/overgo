package workflowruntime

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"testing"

	"overgo/internal/testutil"
)

// TestInputBytesReadingIsWhole holds the record's part on how the bytes of an input are named to the tree.
//   - way = one of the four the test names, each added by one of the four commits the plan's step names
//   - kind = a content hash, a stat triple or a structural digest, and the four ways hold all three
//   - place = a file and a name that file declares, wherever the part cites one, in a field or in prose
//   - code = the production functions and methods that compute a way's name, each in one way only
//   - caller = a production declaration that names a declaration of its way's code
//   - lines = a way's code counted from each declaration's doc comment to its end, again from the tree
//   - deleted = the lines of the code deleted whole, counted from the tree, and the estimated further lines
//   - added = the estimated lines added at the ways and the owner's own estimated lines
//   - net = deleted minus added, inside the estimated range
//   - row = a yes or a no with its reason
//   - a key the part does not define, or a key stated twice in one entry, is refused
func TestInputBytesReadingIsWhole(t *testing.T) {
	t.Parallel()
	root := testutil.RepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "verification", "resume-paths.txt"))
	if err != nil {
		t.Fatal(err)
	}

	const part = "== input bytes =="
	wayKeys := []string{"way", "commit", "kind", "names", "why", "notices", "ignores", "misses", "unreadable", "callers", "keyed-in", "code", "lines", "offer", "deletes-whole", "estimated-deletes", "estimated-adds"}
	partKeys := []string{"owner", "estimated-owner-lines", "stat-cache", "ways-lines", "estimated-deleted", "estimated-added", "estimated-net", "estimated-net-range", "next-key", "input-bytes-row"}
	var ways []map[string]string
	var mentioned []string
	fields := map[string]string{}
	inPart, held := false, false
	for line := range strings.Lines(string(raw)) {
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, "== ") {
			inPart = line == part
			held = held || inPart
			continue
		}
		if !inPart {
			continue
		}
		for word := range strings.FieldsSeq(line) {
			if word = strings.Trim(word, "(),."); recordPlace.MatchString(word) {
				mentioned = append(mentioned, word)
			}
		}
		key, value, keyed := strings.Cut(line, ": ")
		if !keyed || strings.Contains(key, " ") {
			continue
		}
		var entry map[string]string
		switch {
		case slices.Contains(partKeys, key):
			entry = fields
		case key == "way":
			entry = map[string]string{}
			ways = append(ways, entry)
		case !slices.Contains(wayKeys, key):
			t.Fatalf("the record's part on input bytes states %s, want one of %s, or of %s", key, strings.Join(wayKeys, ", "), strings.Join(partKeys, ", "))
		case len(ways) == 0:
			t.Fatalf("the record's part on input bytes states %s before any way", key)
		default:
			entry = ways[len(ways)-1]
		}
		if _, twice := entry[key]; twice {
			t.Fatalf("the record's part on input bytes states %s twice in one entry", key)
		}
		entry[key] = value
	}
	if !held {
		t.Fatalf("the record holds no part %q", part)
	}
	// number reads a count, or a signed number where signed is set.
	number := func(what, text string, signed bool) int {
		t.Helper()
		value, err := strconv.Atoi(text)
		if err != nil || !signed && value < 0 {
			t.Fatalf("the record's %s is %q (%v), want a whole number", what, text, err)
		}
		return value
	}

	// Each cited file is parsed once into one file set, so a declaration's lines are read from it.
	files := token.NewFileSet()
	parsed := map[string]*ast.File{}
	declared := func(what, cited string) ast.Node {
		t.Helper()
		match := recordPlace.FindStringSubmatch(cited)
		if match == nil {
			t.Fatalf("the record's %s cites %q, want file.go#Name or file.go#Type.Name", what, cited)
		}
		syntax := parsed[match[1]]
		if syntax == nil {
			read, err := parser.ParseFile(files, filepath.Join(root, filepath.FromSlash(match[1])), nil, parser.ParseComments)
			if err != nil {
				t.Fatalf("the record's %s cites %s, which the tree does not hold as Go source: %v", what, match[1], err)
			}
			parsed[match[1]], syntax = read, read
		}
		node, _, found := declarationOf(syntax, match[2], match[3])
		if !found {
			t.Fatalf("the record's %s cites %s, which %s does not declare", what, cited, match[1])
		}
		return node
	}
	// lines counts a declaration from its doc comment, where it has one, to its end.
	lines := func(node ast.Node) int {
		start := node.Pos()
		var doc *ast.CommentGroup
		switch typed := node.(type) {
		case *ast.FuncDecl:
			doc = typed.Doc
		case *ast.GenDecl:
			doc = typed.Doc
		case *ast.TypeSpec:
			doc = typed.Doc
		case *ast.ValueSpec:
			doc = typed.Doc
		}
		if doc != nil {
			start = doc.Pos()
		}
		return files.Position(node.End()).Line - files.Position(start).Line + 1
	}
	// places reads a list of places, or none where absent is admitted, and holds each to the tree.
	const absent = "none"
	places := func(what, list string, admitAbsent bool) []string {
		t.Helper()
		if admitAbsent && list == absent {
			return nil
		}
		cited := strings.Fields(list)
		if len(cited) == 0 {
			t.Fatalf("the record's %s names no place", what)
		}
		for _, each := range cited {
			declared(what, each)
		}
		return cited
	}
	// stated holds a statement to the places after its bar, and needsPlaces refuses a statement with none.
	stated := func(what, value string, needsPlaces bool) {
		t.Helper()
		statement, cited, barred := strings.Cut(value, " | ")
		if strings.TrimSpace(statement) == "" || needsPlaces && !barred {
			t.Fatalf("the record's %s is %q, want a statement, a bar and places in the tree", what, value)
		}
		if barred {
			places(what, cited, false)
		}
	}
	// refers says whether a declaration names one of the names, called, passed or selected.
	refers := func(node ast.Node, names []string) (found bool) {
		if function, isFunction := node.(*ast.FuncDecl); isFunction {
			node = function.Body
		}
		ast.Inspect(node, func(each ast.Node) bool {
			ident, isIdent := each.(*ast.Ident)
			found = found || isIdent && slices.Contains(names, ident.Name)
			return !found
		})
		return found
	}
	production := func(cited string) bool {
		file, _, _ := strings.Cut(cited, "#")
		return !strings.HasSuffix(file, "_test.go")
	}
	for _, cited := range mentioned {
		declared("part on input bytes", cited)
	}

	// The four ways the record's closing-input-bytes names, and the four commits the plan's step input-bytes names.
	wanted := []string{"beside-by-content", "cell-files-by-stat", "adapter-by-values", "shards-by-content"}
	commits := []string{"26d332f81", "78534ee41", "2ae868219", "8988728ae"}
	kinds := []string{"content hash", "stat triple", "structural digest"}
	seen, committed, counted := map[string]bool{}, map[string]string{}, map[string]string{}
	kindsHeld := map[string]bool{}
	waysLines, deleted, added := 0, 0, 0
	for _, way := range ways {
		name := way["way"]
		if !slices.Contains(wanted, name) || seen[name] {
			t.Fatalf("the record reads the way %q, want each of %s once", name, strings.Join(wanted, ", "))
		}
		seen[name] = true
		for _, key := range wayKeys {
			if strings.TrimSpace(way[key]) == "" {
				t.Fatalf("the record's way %s states no %s", name, key)
			}
		}
		if commit := way["commit"]; !slices.Contains(commits, commit) || committed[commit] != "" {
			t.Fatalf("the record's way %s was added by %q, want one of %s, each for one way", name, commit, strings.Join(commits, ", "))
		}
		committed[way["commit"]] = name
		if !slices.ContainsFunc(kinds, func(kind string) bool { return strings.HasPrefix(way["kind"], kind) }) {
			t.Fatalf("the record's way %s is of the kind %q, want one that opens with one of %s", name, way["kind"], strings.Join(kinds, ", "))
		}
		for _, kind := range kinds {
			kindsHeld[kind] = kindsHeld[kind] || strings.Contains(way["kind"], kind)
		}
		for _, key := range []string{"names", "why", "notices", "ignores", "misses", "unreadable", "offer"} {
			stated(name+" "+key, way[key], key == "names")
		}
		places(name+" keyed-in", way["keyed-in"], false)

		// A way's lines are the lines of its code, each declaration counted in one way only.
		code := places(name+" code", way["code"], false)
		var codeNames []string
		wayLines := 0
		for _, cited := range code {
			node := declared(name+" code", cited)
			if _, function := node.(*ast.FuncDecl); !function || !production(cited) {
				t.Fatalf("the record's way %s counts %s, want a production function or method", name, cited)
			}
			if earlier := counted[cited]; earlier != "" {
				t.Fatalf("the record's way %s counts %s, which the way %s counts already", name, cited, earlier)
			}
			counted[cited] = name
			codeNames = append(codeNames, cited[strings.LastIndexAny(cited, "#.")+1:])
			wayLines += lines(node)
		}
		if given := number(name+" lines", way["lines"], false); given != wayLines {
			t.Errorf("the record's way %s states %d lines, and its %d declarations hold %d in the tree", name, given, len(code), wayLines)
		}
		waysLines += wayLines

		// A caller is production code that names a declaration of its way's code.
		for _, cited := range places(name+" callers", way["callers"], false) {
			if !production(cited) || !refers(declared(name+" callers", cited), codeNames) {
				t.Fatalf("the record's way %s names the caller %s, which is test code or names none of %s", name, cited, strings.Join(codeNames, ", "))
			}
		}

		// What the owner deletes whole is code of the way, counted from the tree.
		for _, cited := range places(name+" deletes-whole", way["deletes-whole"], true) {
			if !slices.Contains(code, cited) {
				t.Fatalf("the record's way %s deletes %s whole, which is not its code", name, cited)
			}
			deleted += lines(declared(name+" deletes-whole", cited))
		}
		deleted += number(name+" estimated-deletes", way["estimated-deletes"], false)
		added += number(name+" estimated-adds", way["estimated-adds"], false)
	}
	if t.Failed() {
		t.FailNow()
	}
	if len(seen) != len(wanted) {
		t.Fatalf("the record reads %d ways, want the %d of %s", len(seen), len(wanted), strings.Join(wanted, ", "))
	}
	for _, kind := range kinds {
		if !kindsHeld[kind] {
			t.Fatalf("the record's ways hold no %s, want all of %s", kind, strings.Join(kinds, ", "))
		}
	}

	// The part's fields are all given, and its sums are its entries' own.
	for _, key := range partKeys {
		if strings.TrimSpace(fields[key]) == "" {
			t.Fatalf("the record's part on input bytes gives no %s", key)
		}
	}
	for _, key := range []string{"owner", "stat-cache", "next-key"} {
		stated(key, fields[key], true)
	}
	added += number("estimated-owner-lines", fields["estimated-owner-lines"], false)
	if given := number("ways-lines", fields["ways-lines"], false); given != waysLines {
		t.Errorf("the record states ways-lines as %d, and its ways hold %d lines in the tree", given, waysLines)
	}
	if given := number("estimated-deleted", fields["estimated-deleted"], false); given != deleted {
		t.Errorf("the record states estimated-deleted as %d, and its ways give %d", given, deleted)
	}
	if given := number("estimated-added", fields["estimated-added"], false); given != added {
		t.Errorf("the record states estimated-added as %d, and its ways and the owner give %d", given, added)
	}
	net := number("estimated-net", fields["estimated-net"], true)
	if net != deleted-added {
		t.Errorf("the record states estimated-net as %d, want deleted minus added, %d", net, deleted-added)
	}
	bounds := strings.Fields(fields["estimated-net-range"])
	if len(bounds) != 2 {
		t.Fatalf("the record states estimated-net-range as %q, want a low and a high", fields["estimated-net-range"])
	}
	if low, high := number("estimated-net-range low", bounds[0], true), number("estimated-net-range high", bounds[1], true); low > net || net > high {
		t.Errorf("the record states estimated-net-range as %d to %d, which does not hold the net %d", low, high, net)
	}
	if answer, reason, reasoned := strings.Cut(fields["input-bytes-row"], ", because "); !reasoned || answer != "yes" && answer != "no" || strings.TrimSpace(reason) == "" {
		t.Fatalf("the record closes with input-bytes-row %q, want yes or no and the reason", fields["input-bytes-row"])
	}
}
