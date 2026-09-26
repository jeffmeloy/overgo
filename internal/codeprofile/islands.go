package codeprofile

import "slices"

// PrivateIslands returns the unexported production declarations no live code
// reaches. Roots are the declarations a boundary keeps -- a command entry,
// interface dispatch, serialization, reflection, cgo -- and package
// initializers; liveness follows production references from them, so a
// private helper reached only through an export only tests call is found. An
// island is a review question -- a capability to wire, test support to move,
// or code a named wired replacement supersedes -- never a deletion order.
func PrivateIslands(declarations []ConsumerDeclaration, references []ConsumerReference) []ConsumerDeclaration {
	next := map[string][]string{}
	for _, reference := range references {
		from := declarationIdentity(reference.From)
		next[from] = append(next[from], declarationIdentity(reference.To))
	}
	live := map[string]bool{}
	var visit func(string)
	visit = func(identity string) {
		if !live[identity] {
			live[identity] = true
			for _, target := range next[identity] {
				visit(target)
			}
		}
	}
	for _, declaration := range declarations {
		if declaration.Boundary != "" || declaration.Name == "init" && declaration.Receiver == "" {
			visit(declarationIdentity(declaration))
		}
	}
	return slices.DeleteFunc(slices.Clone(declarations), func(declaration ConsumerDeclaration) bool {
		return declaration.Exported || live[declarationIdentity(declaration)]
	})
}
