package evaluation

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/inference"
	"overgo/internal/overgodb"
	"overgo/internal/sequencescore"
	"overgo/internal/tokenizer"
)

// vectorEmbedder is an embedding session whose pictures have the vectors a table gives.
type vectorEmbedder map[string][]float32

func (e vectorEmbedder) Embed(_ context.Context, name string) ([]float32, error) {
	vector, known := e[name]
	if !known {
		return nil, errors.New("no vector for " + name)
	}
	return vector, nil
}

func (vectorEmbedder) Generate(context.Context, string, inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	return nil, "", errors.New("an embedding session generates no text")
}

func (vectorEmbedder) ScoreContinuations(context.Context, string, []string) ([]sequencescore.Score, error) {
	return nil, errors.New("an embedding session scores no continuation")
}

// TestNearestSameGroupCountsTheSiblingsThatAreNearest holds the retrieval score to
// cosine similarity among all the other pictures: a picture whose nearest is a sibling is
// right, one whose nearest is a stranger is wrong, a picture with no sibling is not asked,
// and a vector that is empty is refused.
func TestNearestSameGroupCountsTheSiblingsThatAreNearest(t *testing.T) {
	t.Parallel()
	groups := []string{"a", "a", "b", "b", "solo"}
	vectors := [][]float32{
		{1, 0.1}, {0.9, 0.2}, // a: each other's nearest, though the scales differ
		{0, 1}, {0.1, 0.2}, // b: both are nearer to the solo picture than to each other
		{0.05, 0.9}, // solo
	}
	queries, correct, err := nearestSameGroup(groups, vectors)
	if err != nil {
		t.Fatal(err)
	}
	// Asked: the four pictures with a sibling. The two a's find each other; the first b's
	// nearest is solo and the second b's nearest is an a.
	if queries != 4 || correct != 2 {
		t.Fatalf("asked %d, right %d, want 4 and 2", queries, correct)
	}
	if _, _, err := nearestSameGroup(groups, [][]float32{{1, 0}, {0, 0}, {1, 1}, {1, 1}, {1, 1}}); err == nil {
		t.Fatal("an empty embedding was scored")
	}
}

// TestImageRetrievalScoresASessionAndPicksItsClips runs a suite through a session whose
// vectors place each group together, and holds the selection to its seed and the compiler to
// the suites it refuses.
func TestImageRetrievalScoresASessionAndPicksItsClips(t *testing.T) {
	t.Parallel()
	suite := ImageRetrievalSuite{Kind: imageRetrievalKind, Schema: "fixture/v1", Source: "fixture", Cases: []ImageRetrievalCase{
		{Name: "a1", Group: "a", Image: "a1.png"}, {Name: "a2", Group: "a", Image: "a2.png"},
		{Name: "b1", Group: "b", Image: "b1.png"}, {Name: "b2", Group: "b", Image: "b2.png"},
		{Name: "solo", Group: "solo", Image: "solo.png"},
	}}
	compiled, err := compileImageRetrieval(suite)
	if err != nil {
		t.Fatal(err)
	}
	plan, err := bindImageRetrieval(compiled, ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "model"), RuntimeRecipe: planID(t, artifact.KindRecipe, "recipe"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "environment"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	})
	if err != nil {
		t.Fatal(err)
	}
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	publishPlanFixtureAuthorities(t, store, plan)
	embedder := vectorEmbedder{
		"a1.png": {1, 0, 0}, "a2.png": {0.9, 0.1, 0}, "b1.png": {0, 1, 0}, "b2.png": {0, 0.9, 0.1}, "solo.png": {0, 0, 1},
	}
	report, err := evaluateImageRetrieval(t.Context(), store, embedder, compiled, plan)
	if err != nil || report.Queries != 4 || report.Accuracy != 1 {
		t.Fatalf("report = %+v (%v), want 4 queries all right", report, err)
	}
	if _, err := evaluateImageRetrieval(t.Context(), store, textOnlyRuntime{}, compiled, plan); err == nil {
		t.Fatal("a runtime that embeds nothing was evaluated")
	}
	for name, change := range map[string]func(*ImageRetrievalSuite){
		"no pair":         func(s *ImageRetrievalSuite) { s.Cases = s.Cases[:1] },
		"no image":        func(s *ImageRetrievalSuite) { s.Cases[0].Image = "" },
		"a repeated case": func(s *ImageRetrievalSuite) { s.Cases[1].Name = s.Cases[0].Name },
	} {
		broken := ImageRetrievalSuite{Kind: suite.Kind, Schema: suite.Schema, Source: suite.Source, Cases: append([]ImageRetrievalCase(nil), suite.Cases...)}
		change(&broken)
		if _, err := compileImageRetrieval(broken); err == nil {
			t.Errorf("a suite with %s compiled", name)
		}
	}

	directory := t.TempDir()
	for source := range 3 {
		for scene := range 2 {
			write := func(name string) {
				if err := os.WriteFile(filepath.Join(directory, name+clipExtension), []byte("clip"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			write(fmt.Sprintf("pair%d%s%04d", source, sceneMarker, scene))
		}
	}
	for single := range 6 {
		if err := os.WriteFile(filepath.Join(directory, fmt.Sprintf("single%d%s0001%s", single, sceneMarker, clipExtension)), []byte("clip"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	picks, err := PickRetrievalClips(directory, 3, 17)
	if err != nil || len(picks) != 9 {
		t.Fatalf("picked %d clips (%v), want the six with a sibling and three distractors", len(picks), err)
	}
	again, _ := PickRetrievalClips(directory, 3, 17)
	other, _ := PickRetrievalClips(directory, 3, 18)
	if fmt.Sprint(picks) != fmt.Sprint(again) || fmt.Sprint(picks) == fmt.Sprint(other) {
		t.Fatal("the distractors are not a function of the seed")
	}
	images := map[string]string{}
	for _, pick := range picks {
		images[pick.Clip] = pick.Clip + ".png"
	}
	built, err := BuildSceneRetrieval(picks, images)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := compileImageRetrieval(built); err != nil || len(built.Cases) != 9 {
		t.Fatalf("the picked suite has %d cases and compiles with %v", len(built.Cases), err)
	}
	if _, err := BuildSceneRetrieval(picks, map[string]string{}); err == nil {
		t.Error("a suite was built for clips with no frame")
	}
	if _, err := PickRetrievalClips(directory, 99, 17); err == nil {
		t.Error("more distractors than clips were picked")
	}
}
