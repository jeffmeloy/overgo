package gate

import (
	"go/ast"
	"go/parser"
	"go/token"
	"path/filepath"
	"testing"
)

// TestProjectorSessionOwnsNoPromptWrappers keeps the legacy prompt wrapper
// methods out of internal/projector. The rule lives here, in a package that
// already reads Go source, so the projector's own tests parse no Go and the
// device lane that owns them binds no root beyond what it compiles.
func TestProjectorSessionOwnsNoPromptWrappers(t *testing.T) {
	t.Parallel()
	directory := filepath.Join(liveRepositoryFixture(t).worktree, "internal", "projector")
	packages, err := parser.ParseDir(token.NewFileSet(), directory, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	forbidden := map[string]struct{}{
		"imagesPrompt": {}, "videoPrompt": {}, "audioPrompt": {},
		"audioSampleRate": {}, "mediaHistoryPrompt": {},
		"imagePromptProgram": {}, "mediaPromptProgram": {},
	}
	projector, found := packages["projector"]
	if !found {
		t.Fatal("internal/projector has no package projector")
	}
	for _, file := range projector.Files {
		for _, declaration := range file.Decls {
			function, ok := declaration.(*ast.FuncDecl)
			if !ok || function.Recv == nil {
				continue
			}
			if _, legacy := forbidden[function.Name.Name]; legacy {
				t.Errorf("legacy prompt wrapper remains: %s", function.Name.Name)
			}
		}
	}
}
