// Package codeprofile derives advisory structural facts from parsed Go source.
package codeprofile

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"go/ast"
	"go/printer"
	"go/scanner"
	"go/token"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/repoanalysis"
)

type Partition struct {
	Files int `json:"files"`
	Nodes int `json:"nodes"`
}

type Function struct {
	File          string `json:"file"`
	Name          string `json:"name"`
	Nodes         int    `json:"nodes"`
	Branches      int    `json:"branches"`
	AdvisoryClass string `json:"advisory_class,omitempty"`
}

type Clone struct {
	Fingerprint   string   `json:"fingerprint"`
	Nodes         int      `json:"nodes"`
	Functions     []string `json:"functions"`
	AdvisoryClass string   `json:"advisory_class,omitempty"`
}

type Profile struct {
	Production           Partition       `json:"production"`
	Test                 Partition       `json:"test"`
	Functions            []Function      `json:"functions"`
	Clones               []Clone         `json:"clones"`
	DuplicateExcessNodes int             `json:"duplicate_excess_nodes"`
	ExportedDeclarations int             `json:"exported_declarations"`
	PackageImportEdges   int             `json:"package_import_edges"`
	Consumers            ConsumerSummary `json:"consumers"`
}

func Build(snapshot repoanalysis.SourceSnapshot) (Profile, error) {
	var profile Profile
	imports := map[string]bool{}
	type body struct {
		nodes int
		refs  []string
	}
	bodies := map[string]*body{}
	for _, source := range snapshot.Files {
		generated, err := source.Generated()
		if err != nil {
			return Profile{}, err
		}
		if generated {
			continue
		}
		file, err := source.Syntax()
		if err != nil {
			return Profile{}, err
		}
		nodes := NodeCount(file)
		partition := &profile.Production
		if source.Test {
			partition = &profile.Test
		}
		partition.Files++
		partition.Nodes += nodes
		packagePath := filepath.ToSlash(filepath.Dir(source.Path))
		for _, imported := range file.Imports {
			imports[packagePath+"\x00"+imported.Path.Value] = true
		}
		for _, declaration := range file.Decls {
			switch value := declaration.(type) {
			case *ast.FuncDecl:
				if !source.Test && ast.IsExported(value.Name.Name) {
					profile.ExportedDeclarations++
				}
				if value.Body == nil {
					continue
				}
				size, branches := NodeCount(value.Body), branchCount(value.Body)
				ref := source.Path + ":" + value.Name.Name
				class := advisoryClass(source.Test, value)
				profile.Functions = append(profile.Functions, Function{
					File: source.Path, Name: value.Name.Name, Nodes: size, Branches: branches, AdvisoryClass: class,
				})
				fingerprint, err := bodyFingerprint(value.Body)
				if err != nil {
					return Profile{}, err
				}
				key := class + "\x00" + fingerprint
				if bodies[key] == nil {
					bodies[key] = &body{nodes: size}
				}
				bodies[key].refs = append(bodies[key].refs, ref)
			case *ast.GenDecl:
				for _, spec := range value.Specs {
					switch named := spec.(type) {
					case *ast.TypeSpec:
						if !source.Test && ast.IsExported(named.Name.Name) {
							profile.ExportedDeclarations++
						}
					case *ast.ValueSpec:
						for _, name := range named.Names {
							if !source.Test && ast.IsExported(name.Name) {
								profile.ExportedDeclarations++
							}
						}
					}
				}
			}
		}
	}
	profile.PackageImportEdges = len(imports)
	for key, group := range bodies {
		if len(group.refs) < 2 {
			continue
		}
		class, fingerprint, _ := strings.Cut(key, "\x00")
		sort.Strings(group.refs)
		profile.Clones = append(profile.Clones, Clone{
			Fingerprint: hex.EncodeToString([]byte(fingerprint)), Nodes: group.nodes, Functions: group.refs, AdvisoryClass: class,
		})
		profile.DuplicateExcessNodes += group.nodes * (len(group.refs) - 1)
	}
	sort.Slice(profile.Functions, func(i, j int) bool {
		if profile.Functions[i].Nodes != profile.Functions[j].Nodes {
			return profile.Functions[i].Nodes > profile.Functions[j].Nodes
		}
		return profile.Functions[i].File+profile.Functions[i].Name < profile.Functions[j].File+profile.Functions[j].Name
	})
	sort.Slice(profile.Clones, func(i, j int) bool {
		if profile.Clones[i].Nodes != profile.Clones[j].Nodes {
			return profile.Clones[i].Nodes > profile.Clones[j].Nodes
		}
		return profile.Clones[i].Fingerprint < profile.Clones[j].Fingerprint
	})
	return profile, nil
}

// NodeCount measures the AST surface rooted at node.
func NodeCount(root ast.Node) int {
	count := 0
	ast.Inspect(root, func(node ast.Node) bool {
		if node != nil {
			count++
		}
		return true
	})
	return count
}

func branchCount(root ast.Node) int {
	count := 0
	ast.Inspect(root, func(node ast.Node) bool {
		switch node.(type) {
		case *ast.IfStmt, *ast.ForStmt, *ast.RangeStmt, *ast.CaseClause, *ast.CommClause:
			count++
		}
		return true
	})
	return count
}

func bodyFingerprint(body *ast.BlockStmt) (string, error) {
	var rendered bytes.Buffer
	if err := printer.Fprint(&rendered, token.NewFileSet(), body); err != nil {
		return "", err
	}
	var lexer scanner.Scanner
	lexer.Init(token.NewFileSet().AddFile("", -1, rendered.Len()), rendered.Bytes(), nil, 0)
	hash := sha256.New()
	identifiers := map[string]int{}
	previous := token.ILLEGAL
	for {
		_, tok, literal := lexer.Scan()
		if tok == token.EOF {
			break
		}
		value := tok.String()
		if tok == token.IDENT && previous != token.PERIOD {
			index, ok := identifiers[literal]
			if !ok {
				index = len(identifiers)
				identifiers[literal] = index
			}
			value = string(rune(index + 1))
		} else if tok.IsLiteral() && tok != token.IDENT {
			value = tok.String()
		} else if literal != "" {
			value = literal
		}
		hash.Write([]byte{byte(tok)})
		hash.Write([]byte(value))
		previous = tok
	}
	return string(hash.Sum(nil)), nil
}

func advisoryClass(test bool, function *ast.FuncDecl) string {
	if test {
		return "test"
	}
	if !strings.HasPrefix(strings.ToLower(function.Name.Name), "validate") || function.Type.Results == nil {
		return ""
	}
	for _, result := range function.Type.Results.List {
		if name, ok := result.Type.(*ast.Ident); ok && name.Name == "error" {
			return "validator"
		}
	}
	return ""
}
