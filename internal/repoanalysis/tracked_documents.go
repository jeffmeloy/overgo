package repoanalysis

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"path"
	"slices"
	"strconv"
	"strings"

	"overgo/internal/processcontrol"
)

// The tracked document census answers, for every document the repository
// tracks beside its source, why it is tracked: which production packages name
// it, which of them write it, and which family of documents it belongs to. The
// naming is measured from syntax; the family's kind is a reviewed judgement,
// and the census holds the two to each other, so a document nothing reads
// cannot pass as an input and a new document cannot arrive unclassified.

// DocumentKind says why a family of documents is tracked.
type DocumentKind string

const (
	// DocumentAuthored is written by a person and reviewed in diffs.
	DocumentAuthored DocumentKind = "authored"
	// DocumentBaseline is a ratchet's reviewed threshold.
	DocumentBaseline DocumentKind = "baseline"
	// DocumentGenerated is rewritten by a tool from state held elsewhere.
	DocumentGenerated DocumentKind = "generated"
	// DocumentReceipt is the evidence of one run, written once.
	DocumentReceipt DocumentKind = "receipt"
	// DocumentFixture is an input a tool or a test consumes.
	DocumentFixture DocumentKind = "fixture"
	// DocumentUnread is named by no production package.
	DocumentUnread DocumentKind = "unread"
)

// DocumentFamily classifies the documents its pattern matches. A pattern that
// ends in a slash matches everything beneath that directory; any other is a
// path.Match pattern over the whole repository-relative path.
type DocumentFamily struct {
	Pattern string       `json:"pattern"`
	Kind    DocumentKind `json:"kind"`
}

// TrackedDocument is one document's measured naming and declared family.
type TrackedDocument struct {
	Path    string         `json:"path"`
	Family  DocumentFamily `json:"family,omitzero"`
	Writers []string       `json:"writers,omitempty"`
	Readers []string       `json:"readers,omitempty"`
}

// fileWriterCalls are the lower-case prefixes of a function that puts a file
// on disk, whichever package owns it: a call counts only when its arguments
// name a document.
var fileWriterCalls = []string{"write", "create", "openfile", "replace", "compareandswap", "outputgenerated"}

// documentsRoot is the one directory whose bare name does not name a document:
// every document is beneath it, so a walk of it reads no one of them.
const documentsRoot = "docs"

// familyTypeName is the element type of a classification of documents.
const familyTypeName = "DocumentFamily"

// modulePrefix turns an import path of this module into a package directory.
const modulePrefix = "overgo/"

// fileNameRunes are the characters a file name continues through.
const fileNameRunes = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-."

// TrackedDocumentCensus measures which production packages name each document
// and classifies it by the first family that matches. The first family wins,
// so a specific pattern is listed before the general one that contains it.
func TrackedDocumentCensus(snapshot SourceSnapshot, documents []string, families []DocumentFamily) ([]TrackedDocument, error) {
	census := make([]TrackedDocument, len(documents))
	for index, document := range documents {
		census[index].Path = document
		for _, family := range families {
			if family.matches(document) {
				census[index].Family = family
				break
			}
		}
	}
	packages := map[string][]*ast.File{}
	for _, file := range snapshot.Files {
		if file.Test {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			return nil, err
		}
		packages[path.Dir(file.Path)] = append(packages[path.Dir(file.Path)], syntax)
	}
	constants := packageStrings(packages)
	for pkg, files := range packages {
		var named, written []string
		for _, syntax := range files {
			named = append(named, stringLiterals(syntax)...)
			written = append(written, writtenNames(pkg, syntax, constants)...)
		}
		for index := range census {
			names := func(name string) bool { return namesDocument(name, census[index].Path) }
			switch {
			case slices.ContainsFunc(written, names):
				census[index].Writers = append(census[index].Writers, pkg)
			case slices.ContainsFunc(named, names):
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

func (f DocumentFamily) matches(document string) bool {
	if strings.HasSuffix(f.Pattern, "/") {
		return strings.HasPrefix(document, f.Pattern)
	}
	matched, err := path.Match(f.Pattern, document)
	return err == nil && matched
}

// stringLiterals returns the string literals beneath node that could name a
// document: a path, a pattern or a file name. Import paths name packages.
func stringLiterals(node ast.Node) (names []string) {
	ast.Inspect(node, func(node ast.Node) bool {
		switch typed := node.(type) {
		case *ast.ImportSpec:
			return false
		case *ast.CompositeLit:
			// A classification names documents without reading them.
			if list, isList := typed.Type.(*ast.ArrayType); isList {
				if element, isIdent := list.Elt.(*ast.Ident); isIdent && element.Name == familyTypeName {
					return false
				}
			}
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
	// A path inside a longer literal starts at a boundary: a sentence of help
	// may carry it, the tail of another file's name may not.
	if at := strings.Index(name, document); at == 0 || at > 0 && !strings.ContainsRune(fileNameRunes, rune(name[at-1])) {
		return true
	}
	if strings.ContainsRune(name, '*') {
		matched, err := path.Match(name, document)
		return err == nil && matched
	}
	directory := strings.TrimSuffix(name, "/")
	return directory != documentsRoot && strings.HasPrefix(document, directory+"/")
}

// ValidateTrackedDocuments holds each document's declared family to what was
// measured, and the families to the documents: none unclassified, none stale.
func ValidateTrackedDocuments(census []TrackedDocument, families []DocumentFamily) error {
	var findings []error
	matched := map[DocumentFamily]bool{}
	for _, document := range census {
		named := len(document.Writers)+len(document.Readers) > 0
		matched[document.Family] = true
		switch kind := document.Family.Kind; {
		case kind == "":
			findings = append(findings, fmt.Errorf("tracked document %s belongs to no family: a document enters through a plan row that classifies it", document.Path))
		case kind == DocumentUnread && named:
			findings = append(findings, fmt.Errorf("tracked document %s is declared unread and is named by %v %v", document.Path, document.Writers, document.Readers))
		case kind == DocumentGenerated && len(document.Writers) == 0:
			findings = append(findings, fmt.Errorf("tracked document %s is declared generated and no package that names it writes a file", document.Path))
		case kind != DocumentUnread && kind != DocumentAuthored && !named:
			findings = append(findings, fmt.Errorf("tracked document %s is declared %s and no production package names it", document.Path, kind))
		}
	}
	for _, family := range families {
		if !matched[family] {
			findings = append(findings, fmt.Errorf("document family %s matches no tracked document", family.Pattern))
		}
	}
	return errors.Join(findings...)
}

// TrackedDocumentFamilies is the reviewed classification of every document
// tracked at the repository root and beneath docs. The first match wins, so a
// document named exactly stands before the pattern that would also take it.
var TrackedDocumentFamilies = []DocumentFamily{
	{"README.md", DocumentAuthored},
	{"SECURITY.md", DocumentAuthored},
	{"skill.md", DocumentAuthored},
	{"compatibility.json", DocumentAuthored},
	{"library_validation.json", DocumentAuthored},
	{"media_policy.json", DocumentAuthored},
	{"resource_policy.json", DocumentAuthored},
	{"SBOM.cdx.json", DocumentGenerated},
	{"docs/plan.json", DocumentAuthored},
	{"docs/training_routes.json", DocumentAuthored},
	{"docs/OVERGODB_IMPORT.md", DocumentAuthored},
	{"docs/gui/", DocumentAuthored},
	{"docs/assets/", DocumentAuthored},
	{"docs/image_video_protocol.json", DocumentFixture},
	{"docs/image_video_validation.json", DocumentFixture},
	{"docs/image_video_assertions.json", DocumentFixture},
	{"docs/image_video_*.json", DocumentUnread},
	{"docs/*_baseline.json", DocumentBaseline},
	{"docs/structure_budgets.json", DocumentBaseline},
	{"docs/staged_surface.json", DocumentBaseline},
	{"docs/published_debt.json", DocumentBaseline},
	{"docs/plan_proof_horizon.json", DocumentBaseline},
	{"docs/*.md", DocumentGenerated},
	{"docs/*.json", DocumentGenerated},
	{"docs/media_samples/", DocumentReceipt},
	{"docs/verification/smoke-oracles.json", DocumentFixture},
	{"docs/verification/wan-dit-training-stimulus.txt", DocumentFixture},
	{"docs/verification/rxbrain-mot-training-stimulus.txt", DocumentFixture},
	{"docs/verification/*.json", DocumentReceipt},
	{"docs/verification/", DocumentUnread},
}

// documentPathspecs select what the census covers: documents by their
// extension wherever they are, and everything beneath docs whatever it is.
var documentPathspecs = []string{"*.json", "*.md", "*.txt", "*.jsonl", documentsRoot}

// TrackedDocuments lists the documents of the checkout at root, at the
// repository root and beneath docs: what git tracks and what it would track if
// added, so a stray document is measured before it is committed.
func TrackedDocuments(ctx context.Context, root string) ([]string, error) {
	var output, failure bytes.Buffer
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "git", Args: append([]string{"-C", root, "ls-files", "-co", "--exclude-standard", "--"}, documentPathspecs...),
		Stdout: &output, Stderr: &failure,
	})
	if err != nil {
		return nil, fmt.Errorf("tracked documents: git ls-files: %w", err)
	}
	if receipt.ExitCode != 0 {
		return nil, fmt.Errorf("tracked documents: git ls-files: exit=%d: %s", receipt.ExitCode, strings.TrimSpace(failure.String()))
	}
	var documents []string
	for line := range strings.Lines(output.String()) {
		document := strings.TrimSpace(line)
		if directory := path.Dir(document); directory == "." || directory == documentsRoot || strings.HasPrefix(directory, documentsRoot+"/") {
			documents = append(documents, document)
		}
	}
	slices.Sort(documents)
	return slices.Compact(documents), nil
}
