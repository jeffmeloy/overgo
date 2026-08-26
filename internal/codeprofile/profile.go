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
	File                 string `json:"file"`
	Name                 string `json:"name"`
	Nodes                int    `json:"nodes"`
	Branches             int    `json:"branches"`
	AdvisoryClass        string `json:"advisory_class,omitempty"`
	packagePath          string
	receiver             string
	fingerprint          string
	signatureFingerprint string
	bodyFingerprint      string
}

// StructuralFingerprints returns the package-directory identity, receiver,
// normalized signature digest, and exact body digest already derived while
// building the profile. Digests are lowercase SHA-256 hex.
func (f Function) StructuralFingerprints() (packagePath, receiver, signatureSHA256, bodySHA256 string) {
	return f.packagePath, f.receiver, hex.EncodeToString([]byte(f.signatureFingerprint)), hex.EncodeToString([]byte(f.bodyFingerprint))
}

type Clone struct {
	Fingerprint   string   `json:"fingerprint"`
	Nodes         int      `json:"nodes"`
	Functions     []string `json:"functions"`
	AdvisoryClass string   `json:"advisory_class,omitempty"`
}

// ImpactSelection records how structural ownership affected expensive checks.
// Excluded/Owned is the exact exclusion rate; unresolved checks ran.
type ImpactSelection struct {
	Identity   string `json:"identity,omitempty"`
	Owned      int    `json:"owned"`
	Triggered  int    `json:"triggered"`
	Excluded   int    `json:"excluded"`
	Unresolved int    `json:"unresolved"`
}

type Profile struct {
	Runtime              Partition       `json:"runtime"`
	Automation           Partition       `json:"automation"`
	Generated            Partition       `json:"generated"`
	Test                 Partition       `json:"test"`
	Functions            []Function      `json:"functions"`
	Clones               []Clone         `json:"clones"`
	DuplicateExcessNodes int             `json:"duplicate_excess_nodes"`
	ExportedDeclarations int             `json:"exported_declarations"`
	PackageImportEdges   int             `json:"package_import_edges"`
	Consumers            ConsumerSummary `json:"consumers"`
	Impact               ImpactSelection `json:"impact_selection"`
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
			profile.Generated.Files++
			file, syntaxErr := source.Syntax()
			if syntaxErr != nil {
				return Profile{}, syntaxErr
			}
			profile.Generated.Nodes += NodeCount(file)
			continue
		}
		file, err := source.Syntax()
		if err != nil {
			return Profile{}, err
		}
		nodes := NodeCount(file)
		partition := &profile.Runtime
		switch {
		case source.Test:
			partition = &profile.Test
		case strings.HasPrefix(filepath.ToSlash(source.Path), "cmd/"):
			partition = &profile.Automation
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
				fingerprint, err := functionFingerprint(value)
				if err != nil {
					return Profile{}, err
				}
				signature, err := functionSignatureFingerprint(value)
				if err != nil {
					return Profile{}, err
				}
				bodyFingerprint, err := exactNodeFingerprint(value.Body)
				if err != nil {
					return Profile{}, err
				}
				clone, err := cloneFingerprint(value.Body)
				if err != nil {
					return Profile{}, err
				}
				profile.Functions = append(profile.Functions, Function{
					File: source.Path, Name: value.Name.Name, Nodes: size, Branches: branches, AdvisoryClass: class,
					packagePath: packagePath, receiver: receiverName(value), fingerprint: fingerprint,
					signatureFingerprint: signature, bodyFingerprint: bodyFingerprint,
				})
				key := class + "\x00" + clone
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
		if !hasFunctionPair(group.refs) {
			continue
		}
		class, fingerprint, _ := strings.Cut(key, "\x00")
		sort.Strings(group.refs)
		profile.Clones = append(profile.Clones, Clone{
			Fingerprint: hex.EncodeToString([]byte(fingerprint)), Nodes: group.nodes, Functions: group.refs, AdvisoryClass: class,
		})
		excessCopies := len(group.refs)
		excessCopies--
		profile.DuplicateExcessNodes += group.nodes * excessCopies
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

func receiverName(function *ast.FuncDecl) string {
	if function == nil || function.Recv == nil || len(function.Recv.List) == 0 {
		return ""
	}
	return receiverTypeName(function.Recv.List[0].Type)
}

func receiverTypeName(expression ast.Expr) string {
	switch value := expression.(type) {
	case *ast.Ident:
		return value.Name
	case *ast.StarExpr:
		return receiverTypeName(value.X)
	case *ast.ParenExpr:
		return receiverTypeName(value.X)
	case *ast.IndexExpr:
		return receiverTypeName(value.X)
	case *ast.IndexListExpr:
		return receiverTypeName(value.X)
	default:
		return ""
	}
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

func functionFingerprint(function *ast.FuncDecl) (string, error) {
	return exactNodeFingerprint(function)
}

func exactNodeFingerprint(node ast.Node) (string, error) {
	var rendered bytes.Buffer
	if err := printer.Fprint(&rendered, token.NewFileSet(), node); err != nil {
		return "", err
	}
	digest := sha256.Sum256(rendered.Bytes())
	return string(digest[:]), nil
}

func functionSignatureFingerprint(function *ast.FuncDecl) (string, error) {
	signature := *function
	signature.Doc = nil
	signature.Body = nil
	return exactNodeFingerprint(&signature)
}

func cloneFingerprint(body *ast.BlockStmt) (string, error) {
	var rendered bytes.Buffer
	if err := printer.Fprint(&rendered, token.NewFileSet(), body); err != nil {
		return "", err
	}
	var lexer scanner.Scanner
	files := token.NewFileSet()
	var mode scanner.Mode
	lexer.Init(files.AddFile("", files.Base(), rendered.Len()), rendered.Bytes(), nil, mode)
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
			normalized := index
			normalized++
			value = string(rune(normalized))
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
