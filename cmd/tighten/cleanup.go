package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path"
	"strconv"
)

func cleanupSource(data []byte) ([]byte, error) {
	set := token.NewFileSet()
	file, err := parser.ParseFile(set, "", data, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	used := map[string]bool{}
	ast.Inspect(file, func(node ast.Node) bool {
		if selector, ok := node.(*ast.SelectorExpr); ok {
			if qualifier, ok := selector.X.(*ast.Ident); ok {
				used[qualifier.Name] = true
			}
		}
		return true
	})
	var removals []edit
	for _, declaration := range file.Decls {
		imports, ok := declaration.(*ast.GenDecl)
		if !ok || imports.Tok != token.IMPORT {
			continue
		}
		var unused []ast.Spec
		for _, raw := range imports.Specs {
			spec := raw.(*ast.ImportSpec)
			name := ""
			if spec.Name != nil {
				name = spec.Name.Name
			} else if imported, err := strconv.Unquote(spec.Path.Value); err == nil {
				name = path.Base(imported)
			}
			if name != "_" && name != "." && !used[name] {
				unused = append(unused, raw)
			}
		}
		if len(unused) == len(imports.Specs) {
			removals = append(removals, sourceEdit(imports))
		} else {
			for _, spec := range unused {
				removals = append(removals, sourceEdit(spec))
			}
		}
	}
	if len(removals) > 0 {
		data, err = applyEdits(data, removals)
		if err != nil {
			return nil, err
		}
		file, err = parser.ParseFile(token.NewFileSet(), "", data, parser.ParseComments)
		if err != nil {
			return nil, err
		}
	}
	if len(file.Decls) == 0 {
		return nil, nil
	}
	return data, nil
}

func sourceEdit(node ast.Node) edit {
	return edit{start: int(node.Pos()) - 1, end: int(node.End()) - 1}
}
