package modelrecipe

import (
	"context"
	"errors"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

// TestProjectionSessionPlanBindsProjector pins the first projection
// activation contract: a projection definition's project node carries a
// request-scoped session, and the component plan accounts the projector
// artifact's bytes — not the language model's — so a GGUF-plus-mmproj
// pair compiles a non-empty session policy instead of refusing.
func TestProjectionSessionPlanBindsProjector(t *testing.T) {
	modelID, err := artifact.IdentifyBytes(artifact.KindModel, []byte("language-model"))
	if err != nil {
		t.Fatal(err)
	}
	projectorID, err := artifact.IdentifyBytes(artifact.KindProjector, []byte("vision-projector"))
	if err != nil {
		t.Fatal(err)
	}
	definition, err := ProjectionDefinition(modelID, projectorID, artifact.ID{}, recipe.DataImage, recipe.DataVideo)
	if err != nil {
		t.Fatal(err)
	}
	program, err := CompileCapability(definition)
	if err != nil {
		t.Fatal(err)
	}
	const projectorBytes = 931
	plan, err := CompileComponentSessionPlanWithExtents(
		t.Context(), program,
		func(_ context.Context, id artifact.ID) (uint64, error) {
			if id == projectorID {
				return projectorBytes, nil
			}
			return 0, errors.New("projection sessions must not load the language model")
		},
	)
	if err != nil {
		t.Fatalf("projection session plan: %v", err)
	}
	if len(plan.Components) != 2 {
		t.Fatalf("projection plan holds %d components, want one per media branch", len(plan.Components))
	}
	for _, component := range plan.Components {
		if component.Model != projectorID {
			t.Fatalf("component %s binds %s, want the projector artifact", component.Node, component.Model)
		}
		if component.Session != recipe.SessionRequest {
			t.Fatalf("component %s session = %s, want request scope", component.Node, component.Session)
		}
		if component.ArtifactBytes != projectorBytes {
			t.Fatalf("component %s accounts %d bytes, want the projector extent", component.Node, component.ArtifactBytes)
		}
	}
}
