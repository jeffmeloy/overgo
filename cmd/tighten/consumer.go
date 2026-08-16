package main

import (
	"go/ast"
	"maps"
	"path/filepath"
	"slices"
	"strings"
	"unicode"

	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
)

type reducibleDeclaration struct {
	file *sourceFile
	node ast.Node
	name *ast.Ident
}

func discoverConsumerReductions(root string, snapshot repoanalysis.SourceSnapshot, groups map[string][]*sourceFile) ([]candidate, error) {
	selection, err := repoanalysis.HostBuildSelection(root, "./...")
	if err != nil {
		return nil, err
	}
	declarations, _, err := codeprofile.ProductionConsumerCensus(snapshot, selection, nil)
	if err != nil {
		return nil, err
	}
	locations, names := reducibleDeclarations(groups)
	batches := map[string]*candidate{}
	for _, declaration := range declarations {
		location := locations[consumerLocationKey(declaration)]
		if location.name == nil || declaration.Exported || hasAssembly(root, filepath.ToSlash(filepath.Dir(location.file.Path))) ||
			declaration.ProductionReferences != 0 || declaration.TestReferences != 0 || declaration.Boundary != "" {
			continue
		}
		batch := consumerBatch(batches, location)
		batch.edits[location.file.Path] = append(batch.edits[location.file.Path], declarationEdit(location))
		batch.RemovedNodes += codeprofile.NodeCount(location.node)
		batch.Symbols = append(batch.Symbols, "delete:"+declaration.Name)
	}
	for _, declaration := range declarations {
		location := locations[consumerLocationKey(declaration)]
		batch := batches[packageKey(location)]
		if batch == nil || location.name == nil || len(batch.edits[location.file.Path]) == 0 || declaration.Kind != functionKind ||
			!declaration.Exported || declaration.ProductionReferences == 0 ||
			declaration.ExternalReferences != 0 || declaration.Boundary != "" {
			continue
		}
		replacement := lowerFirst(declaration.Name)
		if replacement == declaration.Name || names[packageKey(location)][replacement] {
			continue
		}
		for _, file := range groups[packageKey(location)] {
			for _, identifier := range file.references[declaration.Name] {
				if selector, ok := file.parents[identifier].(*ast.SelectorExpr); ok && selector.Sel == identifier {
					continue
				}
				if identifier != location.name && identifier.Obj != nil && identifier.Obj != location.name.Obj {
					continue
				}
				batch.edits[file.Path] = append(batch.edits[file.Path], edit{
					start: int(identifier.Pos()) - 1, end: int(identifier.End()) - 1, text: replacement,
				})
				if identifier != location.name {
					batch.Calls++
				}
			}
		}
		batch.Symbols = append(batch.Symbols, "privatize:"+declaration.Name+"->"+replacement)
	}
	result := make([]candidate, 0, len(batches))
	for _, batch := range batches {
		batch.Files = slices.Sorted(maps.Keys(batch.edits))
		slices.Sort(batch.Symbols)
		result = append(result, *batch)
	}
	return result, nil
}

func reducibleDeclarations(groups map[string][]*sourceFile) (map[string]reducibleDeclaration, map[string]map[string]bool) {
	locations := map[string]reducibleDeclaration{}
	names := map[string]map[string]bool{}
	for key, files := range groups {
		names[key] = map[string]bool{}
		for _, file := range files {
			if file.Test || file.generated {
				continue
			}
			for _, declaration := range file.syntax.Decls {
				switch value := declaration.(type) {
				case *ast.FuncDecl:
					if value.Recv == nil {
						recordReducible(locations, names[key], file, value, value.Name, "function")
					}
				case *ast.GenDecl:
					if len(value.Specs) != 1 {
						continue
					}
					switch spec := value.Specs[0].(type) {
					case *ast.TypeSpec:
						recordReducible(locations, names[key], file, value, spec.Name, "type")
					case *ast.ValueSpec:
						if len(spec.Names) == 1 {
							recordReducible(locations, names[key], file, value, spec.Names[0], strings.ToLower(value.Tok.String()))
						}
					}
				}
			}
		}
	}
	return locations, names
}

func recordReducible(locations map[string]reducibleDeclaration, names map[string]bool, file *sourceFile, node ast.Node, name *ast.Ident, kind string) {
	key := strings.Join([]string{file.Path, kind, name.Name}, "\x00")
	locations[key] = reducibleDeclaration{file: file, node: node, name: name}
	names[name.Name] = true
}

func consumerLocationKey(declaration codeprofile.ConsumerDeclaration) string {
	return strings.Join([]string{declaration.File, declaration.Kind, declaration.Name}, "\x00")
}

func packageKey(declaration reducibleDeclaration) string {
	if declaration.file == nil {
		return ""
	}
	return filepath.ToSlash(filepath.Dir(declaration.file.Path)) + "\x00" + declaration.file.syntax.Name.Name
}

func consumerBatch(batches map[string]*candidate, declaration reducibleDeclaration) *candidate {
	key := packageKey(declaration)
	if batches[key] == nil {
		packagePath, _, _ := strings.Cut(key, "\x00")
		batches[key] = &candidate{proposal: proposal{
			ID: packagePath + ":" + consumerSurfaceAction, Action: consumerSurfaceAction, Package: packagePath,
		}, edits: map[string][]edit{}}
	}
	return batches[key]
}

const functionKind = "function"

func declarationEdit(declaration reducibleDeclaration) edit {
	start := declaration.node.Pos()
	if function, ok := declaration.node.(*ast.FuncDecl); ok && function.Doc != nil {
		start = function.Doc.Pos()
	} else if generated, ok := declaration.node.(*ast.GenDecl); ok && generated.Doc != nil {
		start = generated.Doc.Pos()
	}
	return edit{start: int(start) - 1, end: int(declaration.node.End()) - 1}
}

func lowerFirst(value string) string {
	runes := []rune(value)
	if len(runes) > 0 && (len(runes) == 1 || !unicode.IsUpper(runes[1])) {
		runes[0] = unicode.ToLower(runes[0])
	}
	return string(runes)
}
