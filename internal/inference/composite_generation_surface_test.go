package inference

import (
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/testutil"
)

func TestCompositeGenerationUnpromotedRefusal(t *testing.T) {
	store, authority := productionCompositionFixture(t, true, testutil.ArtifactID(t, artifact.KindModel, "bridge-target-model"), bridgeEncoderChannels)
	if _, err := ResolveCompositeGenerationSurface(
		t.Context(), store, authority.Recipe.SourceModel, authority.Recipe.TargetModel, authority.Recipe.Task,
	); err == nil {
		t.Fatal("active but generation-unpromoted composition reached a serving surface")
	}
}
