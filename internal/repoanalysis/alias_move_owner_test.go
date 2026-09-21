package repoanalysis

import (
	"go/ast"
	"path"
	"path/filepath"
	"testing"
)

// TestAliasMovesHaveOneOwner holds the alias move to its owner. Naming the
// target an alias held is the store's compare-and-swap; artifact.AliasMove,
// AliasMoveFrom, MoveAlias and AliasRemoval own it, so production code
// outside the artifact package does not set an artifact.AliasBinding's
// Previous, however the value is spelled. The first form of this test looked
// for one spelling of the value and missed half the sites; this one follows
// the binding's type instead: a literal declared as the binding or elided
// inside a slice of them, and an assignment through a variable built from
// one or through a batch's Aliases. An overlaid package that writes the move
// out four ways is held to being found four times.
func TestAliasMovesHaveOneOwner(t *testing.T) {
	snapshot, err := DiscoverGo(filepath.Join("..", ".."), "internal", "cmd")
	if err != nil {
		t.Fatal(err)
	}
	handwritten := func(snapshot SourceSnapshot) []string {
		var sites []string
		for _, file := range snapshot.Files {
			if file.Test || path.Dir(file.Path) == "internal/artifact" {
				continue
			}
			syntax, err := file.Syntax()
			if err != nil {
				t.Fatal(err)
			}
			bindings := map[string]bool{}
			report := func(ast.Node) { sites = append(sites, file.Path) }
			ast.Inspect(syntax, func(node ast.Node) bool {
				switch typed := node.(type) {
				case *ast.CompositeLit:
					literals := []*ast.CompositeLit{typed}
					if array, ok := typed.Type.(*ast.ArrayType); ok && isAliasBinding(array.Elt) {
						literals = literals[:0]
						for _, element := range typed.Elts {
							if inner, ok := element.(*ast.CompositeLit); ok && inner.Type == nil {
								literals = append(literals, inner)
							}
						}
					} else if !isAliasBinding(typed.Type) {
						return true
					}
					for _, literal := range literals {
						for _, element := range literal.Elts {
							if pair, ok := element.(*ast.KeyValueExpr); ok && identNamed(pair.Key, "Previous") {
								report(pair)
							}
						}
					}
				case *ast.AssignStmt:
					for index, left := range typed.Lhs {
						if name, ok := left.(*ast.Ident); ok && index < len(typed.Rhs) && buildsAliasBinding(typed.Rhs[index]) {
							bindings[name.Name] = true
						}
						selector, ok := left.(*ast.SelectorExpr)
						if !ok || selector.Sel.Name != "Previous" {
							continue
						}
						if name, ok := selector.X.(*ast.Ident); ok && bindings[name.Name] {
							report(typed)
						}
						if indexed, ok := selector.X.(*ast.IndexExpr); ok {
							if owner, ok := indexed.X.(*ast.SelectorExpr); ok && owner.Sel.Name == "Aliases" {
								report(typed)
							}
						}
					}
				}
				return true
			})
		}
		return sites
	}
	if sites := handwritten(snapshot); len(sites) != 0 {
		t.Fatalf("alias moves written out by hand, use artifact.AliasMove, AliasMoveFrom, MoveAlias or AliasRemoval: %v", sites)
	}

	const rogue = `package rogue

import "overgo/internal/artifact"

func move(name string, target, held artifact.ID, pointer *artifact.ID) artifact.Batch {
	first := artifact.AliasBinding{Name: name, Target: target, Previous: artifact.IDPointer(held)}
	second := artifact.AliasBinding{Name: name, Target: target}
	second.Previous = &held
	batch := artifact.Batch{Aliases: []artifact.AliasBinding{{Name: name, Target: target, Previous: pointer}, first, second}}
	batch.Aliases[0].Previous = artifact.CloneID(pointer)
	return batch
}
`
	overlaid, err := snapshot.Overlay(map[string][]byte{"internal/rogue/rogue.go": []byte(rogue)})
	if err != nil {
		t.Fatal(err)
	}
	if sites := handwritten(overlaid); len(sites) != 4 {
		t.Fatalf("the walk found %v in a package that writes the move out four ways", sites)
	}
}

// isAliasBinding reports the type expression artifact.AliasBinding.
func isAliasBinding(expression ast.Expr) bool {
	selector, ok := expression.(*ast.SelectorExpr)
	return ok && selector.Sel.Name == "AliasBinding" && identNamed(selector.X, "artifact")
}

// buildsAliasBinding reports a value that is a binding: the literal, or one
// of the owner's constructors.
func buildsAliasBinding(expression ast.Expr) bool {
	switch typed := expression.(type) {
	case *ast.CompositeLit:
		return isAliasBinding(typed.Type)
	case *ast.CallExpr:
		selector, ok := typed.Fun.(*ast.SelectorExpr)
		return ok && identNamed(selector.X, "artifact") && (selector.Sel.Name == "AliasMove" || selector.Sel.Name == "AliasMoveFrom" || selector.Sel.Name == "MoveAlias")
	}
	return false
}

func identNamed(expression ast.Expr, name string) bool {
	ident, ok := expression.(*ast.Ident)
	return ok && ident.Name == name
}
