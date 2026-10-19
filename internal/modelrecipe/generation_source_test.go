package modelrecipe

import (
	"errors"
	"slices"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/hfrepo"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func TestResolveGenerationSource(t *testing.T) {
	tests := []struct {
		identity hfrepo.Identity
		task     recipe.Task
		prepare  recipe.ModuleID
	}{
		{hfrepo.Identity{ModelType: "neo_chat", Architectures: []string{"NEOChatModel"}}, recipe.TaskImageGen, ModuleRoutedImagePrepare},
		{hfrepo.Identity{Pipeline: "Krea2Pipeline"}, recipe.TaskImageGen, ModuleLatentImagePrepare},
		{hfrepo.Identity{Pipeline: "MOSSImagePipeline"}, recipe.TaskImageGen, ModuleLatentImagePrepare},
		{hfrepo.Identity{ModelType: "uvit"}, recipe.TaskImageGen, ModuleDiffusionImagePrepare},
		{hfrepo.Identity{ModelType: "un0"}, recipe.TaskImageGen, ModuleOscillatorImagePrepare},
		{hfrepo.Identity{ModelType: "un0"}, recipe.TaskVideoGen, ModuleOscillatorVideoPrepare},
	}
	for _, test := range tests {
		profile, err := ResolveGenerationSource(test.task, test.identity)
		if err != nil {
			t.Fatal(err)
		}
		if profile.Prepare != test.prepare {
			t.Fatalf("prepare=%q want %q", profile.Prepare, test.prepare)
		}
		modelID := testutil.ArtifactID(t, artifact.KindModel, string(test.prepare))
		profileID := testutil.ArtifactID(t, artifact.KindProfile, string(test.prepare))
		if profile.Profile == "" {
			profileID = artifact.ID{}
		}
		definition, err := GenerationDefinition(profile.Prepare, modelID, profileID)
		if err != nil {
			t.Fatal(err)
		}
		if definition.Task != test.task || !slices.ContainsFunc(definition.Nodes, func(node recipe.Node) bool {
			return node.Module == test.prepare
		}) {
			t.Fatalf("definition task=%q nodes=%v", definition.Task, definition.Nodes)
		}
	}
}

func TestResolveGenerationSourceRejectsUnknownIdentity(t *testing.T) {
	if _, err := ResolveGenerationSource(recipe.TaskImageGen, hfrepo.Identity{ModelType: "unknown"}); err == nil {
		t.Fatal("expected unknown source rejection")
	}
}

func TestResolveGenerationSourceInfersTask(t *testing.T) {
	for _, pipeline := range []string{"MOSSImagePipeline", "Krea2Pipeline"} {
		profile, err := ResolveGenerationSource("", hfrepo.Identity{Pipeline: pipeline})
		if err != nil || profile.Task != recipe.TaskImageGen {
			t.Fatal(pipeline, profile, err)
		}
	}
	if _, err := ResolveGenerationSource("", hfrepo.Identity{ModelType: "un0"}); err == nil || errors.Is(err, ErrGenerationSourceAbsent) {
		t.Fatal("ambiguous image/video source was not refused", err)
	}
	if _, err := ResolveGenerationSource("", hfrepo.Identity{ModelType: "unknown"}); !errors.Is(err, ErrGenerationSourceAbsent) {
		t.Fatal("unknown source lacks typed absence", err)
	}
}
