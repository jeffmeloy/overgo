package repoanalysis

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/gosyntax"
	"overgo/internal/processcontrol"
)

// The tracked document census answers, for every document the repository
// tracks, why it is tracked: which packages name it, embed it or write it, and
// which documents name it in their own text. Both namings are measured -- from
// syntax, and from the text of the documents -- and neither alone is enough: a
// specification that lists its evidence files is read by a tool that then
// opens each one, and no source literal ever names them. A document is kept
// only by what is measured, never by a classification of it: one that nothing
// reads cannot pass as kept, so no ledger of documents is needed.

// TrackedDocument is one document's measured naming.
type TrackedDocument struct {
	Path    string   `json:"path"`
	Writers []string `json:"writers,omitempty"`
	Readers []string `json:"readers,omitempty"`
	// ReferencedBy are the documents whose text names this one.
	ReferencedBy []string `json:"referenced_by,omitempty"`
}

// entryDocuments are the documents a person opens first, which no tool reads:
// the repository's own description, its agent instructions, its security
// policy, and the license notices beneath licensesRoot that its terms keep.
var entryDocuments = []string{"README.md", "skill.md", "SECURITY.md"}

// licensesRoot holds the license notices the repository carries.
const licensesRoot = "licenses/"

// WorkRecords name documents they are about -- ones a row will delete among
// them -- without reading any, so their text keeps nothing.
var WorkRecords = []string{"docs/plan.json", "docs/sqa_findings.json", "docs/api_manifest.json"}

// embedDirective opens a go:embed line; the patterns after it are read.
const embedDirective = "//go:embed "

// fileWriterCalls are the lower-case prefixes of a function that puts a file
// on disk, whichever package owns it: a call counts only when its arguments
// name a document.
var fileWriterCalls = []string{"write", "create", "openfile", "replace", "compareandswap", "outputgenerated"}

// documentsRoot and fixturesRoot are the two directories whose bare name names
// nothing beneath them: every document is beneath the one and every fixture
// beneath the other, so a walk of either reads no one of them.
const documentsRoot, fixturesRoot = "docs", "fixtures"

// modulePrefix turns an import path of this module into a package directory.
const modulePrefix = "overgo/"

// fileNameRunes are the characters a file name continues through.
const fileNameRunes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."

// TrackedDocumentCensus measures which packages name, embed or write each
// document.
func TrackedDocumentCensus(snapshot gosyntax.Snapshot, documents []string) ([]TrackedDocument, error) {
	census := make([]TrackedDocument, len(documents))
	for index, document := range documents {
		census[index].Path = document
	}
	// A test that opens a document reads it as surely as a tool does, so its
	// literals name documents; it is never the tool that generates one, so
	// what it writes -- a golden file it refreshes -- makes no writer. A test
	// names a file or a pattern, never a directory: it joins a directory to a
	// file name from its own table, the file name is what it reads, and the
	// directory alone would pass off every stray beneath it as read. A test's
	// bare word names the document whose file name it is without the
	// extension, the stem a fixture table joins to one. A go:embed pattern
	// reads what it matches, relative to its package.
	packages, literals, stems := map[string][]*ast.File{}, map[string][]string{}, map[string]map[string]bool{}
	for _, file := range snapshot.Files {
		syntax, err := file.Syntax()
		if err != nil {
			return nil, err
		}
		pkg, names := path.Dir(file.Path), stringLiterals(syntax)
		for _, group := range syntax.Comments {
			for _, comment := range group.List {
				if patterns, found := strings.CutPrefix(comment.Text, embedDirective); found {
					for pattern := range strings.FieldsSeq(patterns) {
						names = append(names, path.Join(pkg, pattern))
					}
				}
			}
		}
		if file.Test {
			names = slices.DeleteFunc(names, func(name string) bool {
				return path.Ext(name) == "" && !strings.ContainsRune(name, '*')
			})
			if stems[pkg] == nil {
				stems[pkg] = map[string]bool{}
			}
			for _, word := range bareWords(syntax) {
				stems[pkg][word] = true
			}
		} else {
			packages[pkg] = append(packages[pkg], syntax)
		}
		literals[pkg] = append(literals[pkg], names...)
	}
	constants := packageStrings(packages)
	for pkg, named := range literals {
		var written []string
		for _, syntax := range packages[pkg] {
			written = append(written, writtenNames(pkg, syntax, constants)...)
		}
		for index := range census {
			document := census[index].Path
			names := func(name string) bool { return namesDocument(name, document) }
			switch {
			case slices.ContainsFunc(written, names):
				census[index].Writers = append(census[index].Writers, pkg)
			case slices.ContainsFunc(named, names) || stems[pkg][strings.TrimSuffix(path.Base(document), path.Ext(document))]:
				census[index].Readers = append(census[index].Readers, pkg)
			}
		}
	}
	for index := range census {
		slices.Sort(census[index].Writers)
		slices.Sort(census[index].Readers)
	}
	return census, nil
}

// bareWords returns the string literals beneath node that carry no path
// separator or extension: in a test, the file stems a fixture table joins to
// a directory and an extension.
func bareWords(node ast.Node) (words []string) {
	ast.Inspect(node, func(node ast.Node) bool {
		if literal, isLiteral := node.(*ast.BasicLit); isLiteral && literal.Kind == token.STRING {
			if value, err := strconv.Unquote(literal.Value); err == nil && value != "" && !strings.ContainsAny(value, "./ ") {
				words = append(words, value)
			}
		}
		return true
	})
	return words
}

// stringLiterals returns the string literals beneath node that could name a
// document: a path, a pattern or a file name. Import paths name packages.
func stringLiterals(node ast.Node) (names []string) {
	ast.Inspect(node, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.ImportSpec:
			return false
		case *ast.BasicLit:
			if value, err := strconv.Unquote(typed.Value); typed.Kind == token.STRING && err == nil && strings.ContainsAny(value, "./") {
				names = append(names, value)
			}
		}
		return true
	})
	return names
}

// packageStrings maps each package-level constant or variable, as
// "<package directory>.<name>", to the string literals of its value: a
// document's path is declared once and written somewhere else by that name.
func packageStrings(packages map[string][]*ast.File) map[string][]string {
	constants := map[string][]string{}
	for pkg, files := range packages {
		for _, syntax := range files {
			for _, declaration := range syntax.Decls {
				general, isGeneral := declaration.(*ast.GenDecl)
				if !isGeneral {
					continue
				}
				for _, spec := range general.Specs {
					value, isValue := spec.(*ast.ValueSpec)
					for index := 0; isValue && index < min(len(value.Names), len(value.Values)); index++ {
						constants[pkg+"."+value.Names[index].Name] = stringLiterals(value.Values[index])
					}
				}
			}
		}
	}
	return constants
}

// writtenNames returns the names one file hands to a file writer. A name
// reaches the call as a literal, as a package-level name of this package or an
// imported one, or as a local variable assigned from either; each top-level
// declaration starts with no locals, so one function's variable is not
// another's.
func writtenNames(pkg string, syntax *ast.File, constants map[string][]string) (written []string) {
	imports := map[string]string{}
	for _, spec := range syntax.Imports {
		imported, _ := strconv.Unquote(spec.Path.Value)
		imported = strings.TrimPrefix(imported, modulePrefix)
		local := path.Base(imported)
		if spec.Name != nil {
			local = spec.Name.Name
		}
		imports[local] = imported
	}
	for _, declaration := range syntax.Decls {
		locals := map[string][]string{}
		resolve := func(node ast.Node) (names []string) {
			ast.Inspect(node, func(node ast.Node) bool {
				switch typed := node.(type) {
				case *ast.SelectorExpr:
					if owner, isIdent := typed.X.(*ast.Ident); isIdent {
						names = append(names, constants[imports[owner.Name]+"."+typed.Sel.Name]...)
					}
				case *ast.Ident:
					names = append(append(names, locals[typed.Name]...), constants[pkg+"."+typed.Name]...)
				}
				return true
			})
			return append(names, stringLiterals(node)...)
		}
		ast.Inspect(declaration, func(node ast.Node) bool {
			switch typed := node.(type) {
			case *ast.AssignStmt:
				for index := 0; index < min(len(typed.Lhs), len(typed.Rhs)); index++ {
					target, isIdent := typed.Lhs[index].(*ast.Ident)
					if !isIdent {
						continue
					}
					// Each name once: a variable assigned from itself in a
					// loop would otherwise double what it carries every time.
					carried := append(locals[target.Name], resolve(typed.Rhs[index])...)
					slices.Sort(carried)
					locals[target.Name] = slices.Compact(carried)
				}
			case *ast.CallExpr:
				if !callsFileWriter(typed.Fun, imports) {
					return true
				}
				for _, argument := range typed.Args {
					written = append(written, resolve(argument)...)
				}
			}
			return true
		})
	}
	return written
}

// callsFileWriter reports whether the callee is named as a file writer is: a
// function of this package or of an imported one. A method is not one -- a
// stream that is written to takes bytes that may mention a document, never
// the path of one.
func callsFileWriter(callee ast.Expr, imports map[string]string) bool {
	name := ""
	switch typed := callee.(type) {
	case *ast.Ident:
		name = typed.Name
	case *ast.SelectorExpr:
		if owner, isIdent := typed.X.(*ast.Ident); isIdent && imports[owner.Name] != "" {
			name = typed.Sel.Name
		}
	}
	name = strings.ToLower(name)
	return slices.ContainsFunc(fileWriterCalls, func(prefix string) bool { return strings.HasPrefix(name, prefix) })
}

// namesDocument reports whether one literal names the document: by its path,
// by its file name alone, by a pattern it matches, or by a directory above it.
func namesDocument(name, document string) bool {
	if name == path.Base(document) {
		return true
	}
	if carriesPath(name, document) {
		return true
	}
	if strings.ContainsRune(name, '*') {
		matched, err := path.Match(name, document)
		return err == nil && matched
	}
	directory := strings.TrimSuffix(name, "/")
	return directory != documentsRoot && directory != fixturesRoot && strings.HasPrefix(document, directory+"/") && contentDigest(document) == ""
}

// hexDigits are the characters a digest written in a text continues through.
const hexDigits = "0123456789abcdefABCDEF"

// contentDigest returns the SHA-256 a document is named by, its file name
// without the extension, or nothing when it is named otherwise. Such a file
// is opened by a name computed from the store or written in a record, never
// by a walk of its directory, so the directory reads none: one that its last
// record stops naming is left unread, and refused.
func contentDigest(document string) string {
	name := strings.TrimSuffix(path.Base(document), path.Ext(document))
	if _, err := hex.DecodeString(name); err != nil || len(name) != hex.EncodedLen(sha256.Size) {
		return ""
	}
	return name
}

// carriesDigest reports whether text carries the digest whole, as an output
// identity, a path or a file name does: one of its runs of hexadecimal digits
// is the digest, and a longer run that holds it is another number. Only a
// text that holds the digest at all is split into its runs.
func carriesDigest(text, digest string) bool {
	return strings.Contains(text, digest) &&
		slices.Contains(strings.FieldsFunc(text, func(r rune) bool { return !strings.ContainsRune(hexDigits, r) }), digest)
}

// carriesPath reports whether text carries the path starting at a boundary: a
// sentence may carry it, the tail of another file's name may not.
func carriesPath(text, document string) bool {
	for from := 0; ; {
		at := strings.Index(text[from:], document)
		if at < 0 {
			return false
		}
		if at += from; at == 0 || !strings.ContainsRune(fileNameRunes, rune(text[at-1])) {
			return true
		}
		from = at + len(document)
	}
}

// textDocuments are the extensions of a document whose text can name another.
var textDocuments = []string{".json", ".jsonl", ".md", ".txt", ".svg"}

// DocumentReferences records, on each document, the documents whose text names
// it: by its repository path, or by its path from the naming document's own
// directory, as a link in a report is written. Source syntax cannot see this
// channel -- a specification that lists its evidence files is read by a tool
// that then opens every file it lists -- so a document is read when code names
// it or when a document that is read names it. A document named by the digest
// of its bytes is named by a text that carries the digest, however it writes
// the rest: a record binds an output by its identity and never by a path.
func DocumentReferences(census []TrackedDocument, read func(document string) ([]byte, error)) error {
	for _, referrer := range census {
		if !slices.Contains(textDocuments, path.Ext(referrer.Path)) {
			continue
		}
		content, err := read(referrer.Path)
		if err != nil {
			return fmt.Errorf("document references: %w", err)
		}
		text, directory := string(content), path.Dir(referrer.Path)+"/"
		for index := range census {
			target := census[index].Path
			if target == referrer.Path {
				continue
			}
			relative, beneath := strings.CutPrefix(target, directory)
			digest := contentDigest(target)
			if digest != "" && carriesDigest(text, digest) ||
				digest == "" && (carriesPath(text, target) || beneath && directory != "./" && carriesPath(text, relative)) {
				census[index].ReferencedBy = append(census[index].ReferencedBy, referrer.Path)
			}
		}
	}
	return nil
}

// LiveDocuments returns the documents something reads: an entry document or a
// license notice a person opens, one a package names, embeds or writes, and --
// to a fixed point -- one that a live document other than a work record names.
func LiveDocuments(census []TrackedDocument) map[string]bool {
	live := map[string]bool{}
	for grew := true; grew; {
		grew = false
		for _, document := range census {
			if live[document.Path] {
				continue
			}
			if slices.Contains(entryDocuments, document.Path) || strings.HasPrefix(document.Path, licensesRoot) ||
				len(document.Writers)+len(document.Readers) > 0 ||
				slices.ContainsFunc(document.ReferencedBy, func(referrer string) bool {
					return live[referrer] && !slices.Contains(WorkRecords, referrer)
				}) {
				live[document.Path], grew = true, true
			}
		}
	}
	return live
}

// CheckoutDocuments measures the census of the checkout at root: every
// document it tracks or would track if added, with the packages of its
// source and the documents that name each. A commit's checks and a listing of
// what the checkout holds read the same measure through it.
func CheckoutDocuments(ctx context.Context, root string) ([]TrackedDocument, error) {
	snapshot, err := gosyntax.DiscoverGo(root, "internal", "cmd")
	documents, listErr := TrackedDocuments(ctx, root)
	if err = errors.Join(err, listErr); err != nil {
		return nil, err
	}
	census, err := TrackedDocumentCensus(snapshot, documents)
	if err != nil {
		return nil, err
	}
	return census, DocumentReferences(census, func(document string) ([]byte, error) {
		return os.ReadFile(filepath.Join(root, filepath.FromSlash(document)))
	})
}

// documentPathspecs select what the census covers: documents by their
// extension wherever they are, and everything beneath docs and beneath
// fixtures whatever it is: a golden tensor is held to a reader as a page is.
var documentPathspecs = []string{"*.json", "*.md", "*.txt", "*.jsonl", documentsRoot, fixturesRoot}

// TrackedDocuments lists the documents of the checkout at root: what git
// tracks and what it would track if added, so a stray document is measured
// before it is committed.
func TrackedDocuments(ctx context.Context, root string) ([]string, error) {
	return checkoutFiles(ctx, root, documentPathspecs)
}

// checkoutFiles lists what the checkout at root holds of the pathspecs: what
// git tracks and what it would track if added.
func checkoutFiles(ctx context.Context, root string, pathspecs []string) ([]string, error) {
	var output, failure bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: append([]string{"-C", root, "ls-files", "-co", "--exclude-standard", "--"}, pathspecs...),
		Stdout: &output, Stderr: &failure,
	})
	if err != nil {
		return nil, fmt.Errorf("tracked documents: git ls-files: %w", err)
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("tracked documents: git ls-files: exit=%d: %s", receipt.ExitCode, strings.TrimSpace(failure.String()))
	}
	// The index still lists a document the checkout has deleted and not yet
	// committed; the census measures what the checkout holds.
	var documents []string
	for line := range strings.Lines(output.String()) {
		if document := strings.TrimSpace(line); document != "" {
			if _, err := os.Stat(filepath.Join(root, filepath.FromSlash(document))); err == nil {
				documents = append(documents, document)
			}
		}
	}
	slices.Sort(documents)
	return slices.Compact(documents), nil
}
