package apimanifest

import (
	"errors"
	"slices"

	"overgo/internal/codemanifest"
)

// CompileDocuments projects source-derived contract declarations.
func CompileDocuments(declarations []codemanifest.DocumentDeclaration) ([]Document, error) {
	documents := make([]Document, 0, len(declarations))
	keys := map[string]bool{}
	for _, declaration := range declarations {
		key := documentKey(declaration.Kind, declaration.MediaType, declaration.Schema)
		if keys[key] {
			return nil, errors.New("API manifest: duplicate source document contract")
		}
		keys[key] = true
		documents = append(documents, Document{
			Name: declaration.Name, Owner: declaration.Owner, VersionOwner: declaration.VersionOwner,
			Source: declaration.Source, SourceIdentity: declaration.SourceIdentity,
			Kind: declaration.Kind, MediaType: declaration.MediaType, Schema: declaration.Schema,
			BuildContexts: slices.Clone(declaration.BuildContexts),
		})
	}
	return documents, nil
}
