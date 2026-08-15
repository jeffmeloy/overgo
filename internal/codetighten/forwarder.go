// Package codetighten performs bounded, mechanically proven source reductions.
package codetighten

import (
	"bytes"
	"errors"
	"fmt"
	"go/ast"
	"go/format"
	"go/printer"
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"

	"overgo/internal/codeprofile"
	"overgo/internal/repoanalysis"
)

// Proposal identifies one exact forwarding wrapper reduction.
type Proposal struct {
	ID           string   `json:"id"`
	Package      string   `json:"package"`
	Wrapper      string   `json:"wrapper"`
	Target       string   `json:"target"`
	Calls        int      `json:"calls"`
	Files        []string `json:"files"`
	RemovedNodes int      `json:"removed_nodes"`
}

type sourceFile struct {
	repoanalysis.GoFile
	syntax     *ast.File
	generated  bool
	parents    map[ast.Node]ast.Node
	references map[string][]*ast.Ident
	shadowed   map[string]bool
}

type declaration struct {
	file *sourceFile
	decl *ast.FuncDecl
}

type edit struct {
	start, end int
	text       string
}

type candidate struct {
	Proposal
	wrapper declaration
	edits   map[string][]edit
}

// Discover returns only reductions whose signatures, forwarding arguments,
// declarations, and direct-call-only usage can be proven from repository ASTs.
func Discover(root string) ([]Proposal, error) {
	candidates, err := discover(root)
	if err != nil {
		return nil, err
	}
	proposals := make([]Proposal, len(candidates))
	for index := range candidates {
		proposals[index] = candidates[index].Proposal
	}
	return proposals, nil
}

// Apply migrates every direct caller, deletes the selected wrapper, and keeps
// the edit only when the affected package tests pass and its AST surface falls.
func Apply(root, id string) (Proposal, error) {
	candidates, err := discover(root)
	if err != nil {
		return Proposal{}, err
	}
	var selected *candidate
	for index := range candidates {
		if candidates[index].ID == id {
			selected = &candidates[index]
			break
		}
	}
	if selected == nil {
		return Proposal{}, fmt.Errorf("exact forwarder %q is not available", id)
	}
	before, err := packageProfile(root, selected.Package)
	if err != nil {
		return Proposal{}, err
	}
	originals := map[string][]byte{}
	updates := map[string][]byte{}
	for name, edits := range selected.edits {
		path := filepath.Join(root, filepath.FromSlash(name))
		data, err := os.ReadFile(path)
		if err != nil {
			return Proposal{}, err
		}
		originals[name] = data
		updated, err := applyEdits(data, edits)
		if err != nil {
			return Proposal{}, fmt.Errorf("tighten %s: %w", name, err)
		}
		updates[name] = updated
	}
	for name, updated := range updates {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.WriteFile(path, updated, fileMode(path)); err != nil {
			return Proposal{}, errors.Join(err, restore(root, originals))
		}
	}
	rollback := func(err error) (Proposal, error) {
		return Proposal{}, errors.Join(err, restore(root, originals))
	}
	command := exec.Command("go", "test", packagePattern(selected.Package), "-count=1")
	command.Dir = root
	if output, err := command.CombinedOutput(); err != nil {
		return rollback(fmt.Errorf("verify reduction: %w\n%s", err, bytes.TrimSpace(output)))
	}
	after, err := packageProfile(root, selected.Package)
	if err != nil {
		return rollback(err)
	}
	beforeNodes := before.Production.Nodes + before.Test.Nodes
	afterNodes := after.Production.Nodes + after.Test.Nodes
	if afterNodes >= beforeNodes || len(after.Functions) >= len(before.Functions) {
		return rollback(fmt.Errorf("reduction did not lower package AST surface: nodes %d -> %d, functions %d -> %d", beforeNodes, afterNodes, len(before.Functions), len(after.Functions)))
	}
	return selected.Proposal, nil
}

func discover(root string) ([]candidate, error) {
	snapshot, err := repoanalysis.DiscoverGo(root, ".")
	if err != nil {
		return nil, err
	}
	groups := map[string][]*sourceFile{}
	for index := range snapshot.Files {
		file := &snapshot.Files[index]
		if (file.Test && !strings.HasSuffix(file.Path, "_test.go")) || strings.Contains("/"+filepath.ToSlash(file.Path)+"/", "/vendor/") {
			continue
		}
		syntax, err := file.Syntax()
		if err != nil {
			return nil, fmt.Errorf("parse %s: %w", file.Path, err)
		}
		generated, err := file.Generated()
		if err != nil {
			return nil, err
		}
		parents, references, shadowed := indexSyntax(syntax)
		source := &sourceFile{GoFile: *file, syntax: syntax, generated: generated, parents: parents, references: references, shadowed: shadowed}
		key := filepath.ToSlash(filepath.Dir(file.Path)) + "\x00" + syntax.Name.Name
		groups[key] = append(groups[key], source)
	}
	var found []candidate
	for key, files := range groups {
		packagePath, _, _ := strings.Cut(key, "\x00")
		if hasAssembly(root, packagePath) {
			continue
		}
		declarations := collectDeclarations(files)
		for _, named := range declarations {
			if len(named) != 1 || ast.IsExported(named[0].decl.Name.Name) || named[0].file.generated || hasDirective(named[0].decl.Doc) {
				continue
			}
			wrapper := named[0]
			targetName, arguments, ellipsis, ok := forwardedCall(wrapper.decl)
			if !ok || targetName == wrapper.decl.Name.Name || len(declarations[targetName]) != 1 {
				continue
			}
			target := declarations[targetName][0]
			if target.file != wrapper.file || ast.IsExported(targetName) || !sameSignature(wrapper.decl.Type, target.decl.Type) || !forwardsParameters(wrapper.decl.Type, arguments, ellipsis) {
				continue
			}
			value, ok := buildCandidate(packagePath, files, wrapper, target)
			if ok {
				found = append(found, value)
			}
		}
	}
	sort.Slice(found, func(i, j int) bool { return found[i].ID < found[j].ID })
	return found, nil
}

func collectDeclarations(files []*sourceFile) map[string][]declaration {
	result := map[string][]declaration{}
	for _, file := range files {
		for _, node := range file.syntax.Decls {
			function, ok := node.(*ast.FuncDecl)
			if ok && function.Recv == nil && function.Body != nil && function.Type.TypeParams == nil {
				result[function.Name.Name] = append(result[function.Name.Name], declaration{file: file, decl: function})
			}
		}
	}
	return result
}

func forwardedCall(function *ast.FuncDecl) (string, []ast.Expr, bool, bool) {
	if len(function.Body.List) != 1 {
		return "", nil, false, false
	}
	var call *ast.CallExpr
	switch statement := function.Body.List[0].(type) {
	case *ast.ReturnStmt:
		if len(statement.Results) != 1 {
			return "", nil, false, false
		}
		call, _ = statement.Results[0].(*ast.CallExpr)
	case *ast.ExprStmt:
		call, _ = statement.X.(*ast.CallExpr)
	}
	if call == nil {
		return "", nil, false, false
	}
	target, ok := call.Fun.(*ast.Ident)
	if !ok {
		return "", nil, false, false
	}
	return target.Name, call.Args, call.Ellipsis.IsValid(), true
}

func hasDirective(comments *ast.CommentGroup) bool {
	if comments == nil {
		return false
	}
	for _, comment := range comments.List {
		text := strings.TrimSpace(strings.TrimPrefix(comment.Text, "//"))
		if strings.HasPrefix(text, "go:") || strings.HasPrefix(text, "export ") {
			return true
		}
	}
	return false
}

func hasAssembly(root, packagePath string) bool {
	entries, err := os.ReadDir(filepath.Join(root, filepath.FromSlash(packagePath)))
	if err != nil {
		return true
	}
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasSuffix(entry.Name(), ".s") {
			return true
		}
	}
	return false
}

func sameSignature(left, right *ast.FuncType) bool {
	return fieldTypes(left.Params) == fieldTypes(right.Params) && fieldTypes(left.Results) == fieldTypes(right.Results)
}

func fieldTypes(fields *ast.FieldList) string {
	if fields == nil {
		return ""
	}
	var result strings.Builder
	for _, field := range fields.List {
		count := len(field.Names)
		if count == 0 {
			count = 1
		}
		var rendered strings.Builder
		_ = printer.Fprint(&rendered, token.NewFileSet(), field.Type)
		for range count {
			result.WriteString(rendered.String())
			result.WriteByte(0)
		}
	}
	return result.String()
}

func forwardsParameters(function *ast.FuncType, arguments []ast.Expr, ellipsis bool) bool {
	var names []string
	if function.Params == nil {
		return len(arguments) == 0 && !ellipsis
	}
	for _, field := range function.Params.List {
		if len(field.Names) == 0 {
			return false
		}
		for _, name := range field.Names {
			if name.Name == "_" {
				return false
			}
			names = append(names, name.Name)
		}
	}
	if len(names) != len(arguments) {
		return false
	}
	for index, argument := range arguments {
		name, ok := argument.(*ast.Ident)
		if !ok || name.Name != names[index] {
			return false
		}
	}
	variadic := false
	if len(function.Params.List) > 0 {
		_, variadic = function.Params.List[len(function.Params.List)-1].Type.(*ast.Ellipsis)
	}
	return ellipsis == variadic
}

func buildCandidate(packagePath string, files []*sourceFile, wrapper, target declaration) (candidate, bool) {
	value := candidate{edits: map[string][]edit{}}
	changed := map[string]bool{wrapper.file.Path: true}
	for _, file := range files {
		valid := true
		for _, identifier := range file.references[wrapper.decl.Name.Name] {
			if identifier == wrapper.decl.Name {
				continue
			}
			if identifier.Obj != nil && identifier.Obj != wrapper.decl.Name.Obj {
				continue
			}
			call, direct := file.parents[identifier].(*ast.CallExpr)
			if !direct || call.Fun != identifier || file.generated || file.shadowed[target.decl.Name.Name] {
				valid = false
				break
			}
			value.edits[file.Path] = append(value.edits[file.Path], edit{start: int(identifier.Pos()) - 1, end: int(identifier.End()) - 1, text: target.decl.Name.Name})
			changed[file.Path] = true
			value.Calls++
		}
		if !valid {
			return candidate{}, false
		}
	}
	if value.Calls == 0 {
		return candidate{}, false
	}
	start := wrapper.decl.Pos()
	if wrapper.decl.Doc != nil {
		start = wrapper.decl.Doc.Pos()
	}
	value.edits[wrapper.file.Path] = append(value.edits[wrapper.file.Path], edit{start: int(start) - 1, end: int(wrapper.decl.End()) - 1})
	value.ID = packagePath + ":" + wrapper.decl.Name.Name + "->" + target.decl.Name.Name
	value.Package = packagePath
	value.Wrapper = wrapper.decl.Name.Name
	value.Target = target.decl.Name.Name
	value.RemovedNodes = codeprofile.NodeCount(wrapper.decl)
	for name := range changed {
		value.Files = append(value.Files, name)
	}
	sort.Strings(value.Files)
	return value, true
}

func indexSyntax(root ast.Node) (map[ast.Node]ast.Node, map[string][]*ast.Ident, map[string]bool) {
	parents := map[ast.Node]ast.Node{}
	references := map[string][]*ast.Ident{}
	shadowed := map[string]bool{}
	var stack []ast.Node
	ast.Inspect(root, func(node ast.Node) bool {
		if node == nil {
			stack = stack[:len(stack)-1]
			return true
		}
		if len(stack) > 0 {
			parents[node] = stack[len(stack)-1]
		}
		if identifier, ok := node.(*ast.Ident); ok {
			references[identifier.Name] = append(references[identifier.Name], identifier)
			if identifier.Obj != nil && identifier.Obj.Kind != ast.Fun {
				shadowed[identifier.Name] = true
			}
		}
		stack = append(stack, node)
		return true
	})
	return parents, references, shadowed
}

func applyEdits(data []byte, edits []edit) ([]byte, error) {
	sort.Slice(edits, func(i, j int) bool { return edits[i].start > edits[j].start })
	for _, change := range edits {
		if change.start < 0 || change.end < change.start || change.end > len(data) {
			return nil, fmt.Errorf("invalid source edit [%d:%d]", change.start, change.end)
		}
		data = append(append(append([]byte(nil), data[:change.start]...), change.text...), data[change.end:]...)
	}
	return format.Source(data)
}

func packageProfile(root, packagePath string) (codeprofile.Profile, error) {
	snapshot, err := repoanalysis.DiscoverGo(root, packagePath)
	if err != nil {
		return codeprofile.Profile{}, err
	}
	return codeprofile.Build(snapshot)
}

func packagePattern(path string) string {
	if path == "." {
		return "."
	}
	return "./" + filepath.ToSlash(path)
}

func fileMode(path string) os.FileMode {
	info, err := os.Stat(path)
	if err != nil {
		return 0o644
	}
	return info.Mode()
}

func restore(root string, originals map[string][]byte) error {
	var failures []error
	for name, data := range originals {
		path := filepath.Join(root, filepath.FromSlash(name))
		if err := os.WriteFile(path, data, fileMode(path)); err != nil {
			failures = append(failures, fmt.Errorf("restore %s: %w", name, err))
		}
	}
	return errors.Join(failures...)
}
