// closure-scan: the native magic baseline tool (floor component 5, Automation
// Doctrine Layer 6). Report mode ranks overgo's own numeric constants as
// closure candidates; -triage mode emits selected constants as closure-ledger
// documents in the store, owned and pinned by their declaring source file.
//
// A MAGICS.json side-file is deliberately NOT the shape here: the store is the
// ledger's one owner. The scan reports; a human or agent triages (tier +
// closure path per row); the tool authors canonical documents. The full
// catalog is ongoing triage -- this tool is the mechanism.
//
// Skips: test files, testdata, iota enumerations (enumerations are not
// magics), string constants, and generated kernel bindings. Everything else
// numeric is REPORTED; filtering by judgment happens at triage, not in the
// scanner, so nothing is silently exempt.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/closureledger"
	"overgo/internal/repodb"
	"overgo/internal/strictjson"
)

type candidate struct {
	Name    string `json:"name"`
	File    string `json:"file"`
	Value   string `json:"value"`
	Doc     string `json:"doc,omitempty"`
	Score   int    `json:"score"`
	Package string `json:"package"`
}

type triageRow struct {
	Name          string `json:"name"`
	File          string `json:"file"`
	Tier          string `json:"tier"`
	Status        string `json:"status"`
	ClosurePath   string `json:"closure_path"`
	RerankTrigger string `json:"rerank_trigger"`
}

type triageFile struct {
	Rows []triageRow `json:"rows"`
}

func main() {
	triagePath := flag.String("triage", "", "triage JSON ({rows:[{name,file,tier,status,closure_path,rerank_trigger}]}); emits closure documents to the store")
	storePath := flag.String("store", "repodb-store", "RepoDB store directory (emit mode)")
	limit := flag.Int("limit", 40, "report mode: top-N candidates to print")
	flag.Parse()
	root, err := os.Getwd()
	if err != nil {
		fatal(err)
	}
	candidates, err := scan(root)
	if err != nil {
		fatal(err)
	}
	if *triagePath == "" {
		report(candidates, *limit)
		return
	}
	if err := emit(root, *storePath, *triagePath, candidates); err != nil {
		fatal(err)
	}
}

func fatal(err error) {
	fmt.Fprintf(os.Stderr, "closure-scan: %v\n", err)
	os.Exit(1)
}

func scan(root string) ([]candidate, error) {
	var out []candidate
	fileSet := token.NewFileSet()
	for _, top := range []string{"internal", "cmd"} {
		err := filepath.WalkDir(filepath.Join(root, top), func(path string, entry os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if entry.IsDir() {
				if entry.Name() == "testdata" || entry.Name() == "generated" {
					return filepath.SkipDir
				}
				return nil
			}
			if !strings.HasSuffix(path, ".go") || strings.HasSuffix(path, "_test.go") {
				return nil
			}
			parsed, err := parser.ParseFile(fileSet, path, nil, parser.ParseComments)
			if err != nil {
				return err
			}
			relative, err := filepath.Rel(root, path)
			if err != nil {
				return err
			}
			collect(parsed, filepath.ToSlash(relative), &out)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		if out[i].Score != out[j].Score {
			return out[i].Score > out[j].Score
		}
		return out[i].Name < out[j].Name
	})
	return out, nil
}

func collect(file *ast.File, relative string, out *[]candidate) {
	for _, declaration := range file.Decls {
		generic, ok := declaration.(*ast.GenDecl)
		if !ok || generic.Tok != token.CONST {
			continue
		}
		blockDoc := ""
		if generic.Doc != nil {
			blockDoc = strings.TrimSpace(generic.Doc.Text())
		}
		for _, spec := range generic.Specs {
			value, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			if usesIota(value) {
				continue // enumerations are not magics
			}
			doc := blockDoc
			if value.Doc != nil {
				doc = strings.TrimSpace(value.Doc.Text())
			}
			for index, name := range value.Names {
				if name.Name == "_" || index >= len(value.Values) {
					continue
				}
				literal := numericLiteral(value.Values[index])
				if literal == "" {
					continue
				}
				*out = append(*out, candidate{
					Name: name.Name, File: relative, Value: literal,
					Doc: doc, Score: score(name.Name, doc, literal),
					Package: filepath.ToSlash(filepath.Dir(relative)),
				})
			}
		}
	}
}

func usesIota(spec *ast.ValueSpec) bool {
	for _, value := range spec.Values {
		found := false
		ast.Inspect(value, func(node ast.Node) bool {
			if identifier, ok := node.(*ast.Ident); ok && identifier.Name == "iota" {
				found = true
			}
			return !found
		})
		if found {
			return true
		}
	}
	return len(spec.Values) == 0 // implicit iota continuation
}

// numericLiteral renders the constant's expression when every leaf is a
// numeric literal; composite bounds like 512<<20 stay reportable.
func numericLiteral(expression ast.Expr) string {
	numeric := true
	ast.Inspect(expression, func(node ast.Node) bool {
		switch leaf := node.(type) {
		case *ast.BasicLit:
			if leaf.Kind != token.INT && leaf.Kind != token.FLOAT {
				numeric = false
			}
		case *ast.Ident, *ast.CallExpr, *ast.SelectorExpr:
			numeric = false
		}
		return numeric
	})
	if !numeric {
		return ""
	}
	return render(expression)
}

func render(expression ast.Expr) string {
	switch typed := expression.(type) {
	case *ast.BasicLit:
		return typed.Value
	case *ast.BinaryExpr:
		return render(typed.X) + typed.Op.String() + render(typed.Y)
	case *ast.UnaryExpr:
		return typed.Op.String() + render(typed.X)
	case *ast.ParenExpr:
		return "(" + render(typed.X) + ")"
	default:
		return ""
	}
}

// score is a triage-ordering heuristic ONLY -- it decides report order, never
// admission. Decision-shaped vocabulary ranks up; codec/layout vocabulary
// ranks down (format facts are usually implementation constraints).
func score(name, doc, value string) int {
	text := strings.ToLower(name + " " + doc)
	total := 0
	for _, hot := range []string{"max", "min", "limit", "bound", "budget", "threshold", "depth", "width", "iter", "retry", "timeout", "window", "sample", "multiplier", "seq", "batch"} {
		if strings.Contains(text, hot) {
			total += 2
		}
	}
	for _, cold := range []string{"offset", "version", "magic-number", "header", "byte", "kind", "schema"} {
		if strings.Contains(text, cold) {
			total--
		}
	}
	if value != "0" && value != "1" {
		total++
	}
	return total
}

func report(candidates []candidate, limit int) {
	fmt.Printf("closure-scan: %d numeric constants (iota enums, tests, generated excluded)\n", len(candidates))
	fmt.Printf("%-6s %-44s %-16s %s\n", "score", "const", "value", "file")
	for index, row := range candidates {
		if index >= limit {
			fmt.Printf("... %d more (raise -limit)\n", len(candidates)-limit)
			break
		}
		fmt.Printf("%-6d %-44s %-16s %s\n", row.Score, row.Name, row.Value, row.File)
	}
}

func emit(root, storePath, triagePath string, candidates []candidate) error {
	raw, err := os.ReadFile(triagePath)
	if err != nil {
		return err
	}
	var triage triageFile
	if err := strictjson.DecodeBytes(raw, &triage); err != nil {
		return fmt.Errorf("triage file: %w", err)
	}
	if len(triage.Rows) == 0 {
		return fmt.Errorf("triage file has no rows")
	}
	byKey := map[string]candidate{}
	for _, row := range candidates {
		byKey[row.File+"#"+row.Name] = row
	}
	batch := artifact.Batch{Key: "closure-scan/native-baseline"}
	fileIDs := map[string]artifact.ID{}
	for _, row := range triage.Rows {
		found, ok := byKey[row.File+"#"+row.Name]
		if !ok {
			return fmt.Errorf("triage row %s not found by scan in %s (stale triage?)", row.Name, row.File)
		}
		fileID, ok := fileIDs[row.File]
		if !ok {
			id, descriptor, location, err := fileArtifact(root, row.File)
			if err != nil {
				return err
			}
			batch.Artifacts = append(batch.Artifacts, descriptor)
			batch.Locations = append(batch.Locations, location)
			fileIDs[row.File] = id
			fileID = id
		}
		valueJSON, err := json.Marshal(json.Number(found.Value))
		if err != nil || !json.Valid(valueJSON) {
			valueJSON, _ = json.Marshal(found.Value) // composite expressions pin as strings
		}
		document, err := closureledger.New(
			row.Name, valueJSON,
			closureledger.Tier(row.Tier), closureledger.Status(row.Status),
			[]artifact.ID{fileID}, row.ClosurePath, row.RerankTrigger, fileID,
		)
		if err != nil {
			return fmt.Errorf("row %s: %w", row.Name, err)
		}
		content, err := document.Content()
		if err != nil {
			return err
		}
		batch.Contents = append(batch.Contents, content)
		batch.Lineage = append(batch.Lineage, document.Lineage()...)
	}
	store, err := repodb.Open(filepath.Join(root, storePath))
	if err != nil {
		return err
	}
	defer store.Close()
	commit, err := store.Commit(context.Background(), batch)
	if err != nil {
		return err
	}
	fmt.Printf("emitted %d closure documents (%d owner files) commit %x\n",
		len(triage.Rows), len(fileIDs), commit[:8])
	return nil
}

func fileArtifact(root, relative string) (artifact.ID, artifact.Descriptor, artifact.LocationEvent, error) {
	resolved := filepath.Join(root, filepath.FromSlash(relative))
	file, err := os.Open(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	defer file.Close()
	id, size, err := artifact.Identify(artifact.KindFile, file)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	absolute, err := filepath.Abs(resolved)
	if err != nil {
		return artifact.ID{}, artifact.Descriptor{}, artifact.LocationEvent{}, err
	}
	return id, artifact.Descriptor{ID: id, Size: size}, artifact.LocationEvent{
		Location: artifact.Location{Artifact: id, Kind: artifact.LocationFile, Value: absolute},
		Action:   artifact.LocationAdd,
	}, nil
}
