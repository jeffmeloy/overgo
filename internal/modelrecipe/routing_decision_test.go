package modelrecipe

import (
	"math"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
	"overgo/internal/testutil"
)

func routingCandidateFixture(t *testing.T, name string, resource uint64) RoutingCandidate {
	t.Helper()
	return RoutingCandidate{
		Recipe:        testutil.ArtifactID(t, artifact.KindRecipe, name),
		Model:         testutil.ArtifactID(t, artifact.KindModel, "routing-model"),
		Evidence:      []artifact.ID{testutil.ArtifactID(t, artifact.KindEvidence, name+"-evaluation")},
		Quality:       0.7,
		ResourceBytes: resource,
	}
}

// TestRecipeRoutingDecisionContract pins the routing decision record: the canonical
// fact binds the task signal with its evidence-derived threshold, every
// candidate with the exact evidence it was judged on, and a selection derived
// by the one registered rule. Candidate order and evidence order canonicalize
// so the identity is content-derived, the parsed document proves its
// identity, lineage names every cited artifact exactly once, and a record
// that omits any binding — or invents a derivation — refuses.
func TestRecipeRoutingDecisionContract(t *testing.T) {
	cheap := routingCandidateFixture(t, "routing-cheap", 1024)
	costly := routingCandidateFixture(t, "routing-costly", 4096)
	template := RecipeRoutingDecision{
		Signal: RoutingSignal{
			Task: recipe.TaskInference, Capability: "long-context retrieval",
			QualityThreshold:  0.6,
			ThresholdEvidence: testutil.ArtifactID(t, artifact.KindEvidence, "routing-threshold"),
		},
		Candidates: []RoutingCandidate{costly, cheap},
		Selected:   cheap.Recipe,
		Derivation: RoutingDerivationCheapestEligible,
	}
	decision, err := NewRecipeRoutingDecision(template)
	if err != nil {
		t.Fatal(err)
	}
	if !decision.ID.Valid() || decision.ID.Kind() != artifact.KindEvidence {
		t.Fatalf("decision identity = %s", decision.ID)
	}
	if strings.Compare(
		decision.Candidates[0].Recipe.String(), decision.Candidates[1].Recipe.String(),
	) >= 0 {
		t.Fatalf("candidates are not canonical: %+v", decision.Candidates)
	}
	reordered := template
	reordered.Candidates = []RoutingCandidate{cheap, costly}
	if identical, err := NewRecipeRoutingDecision(reordered); err != nil || identical.ID != decision.ID {
		t.Fatalf("candidate order changed the identity: (%s, %v)", identical.ID, err)
	}

	batch, err := decision.Batch("routing/decision/" + decision.ID.String())
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseRecipeRoutingDecision(batch.Contents[0].Data)
	if err != nil || parsed.ID != decision.ID || parsed.Selected != decision.Selected {
		t.Fatalf("parsed decision = (%s, %v)", parsed.ID, err)
	}

	lineage := decision.Lineage()
	cited := map[artifact.ID]int{}
	for _, edge := range lineage {
		if edge.Child != decision.ID {
			t.Fatalf("lineage child = %s", edge.Child)
		}
		cited[edge.Parent]++
	}
	for _, parent := range []artifact.ID{
		template.Signal.ThresholdEvidence, cheap.Recipe, costly.Recipe,
		cheap.Model, cheap.Evidence[0], costly.Evidence[0],
	} {
		if cited[parent] != 1 {
			t.Fatalf("lineage cites %s %d times", parent, cited[parent])
		}
	}

	refusals := []struct {
		name   string
		mutate func(*RecipeRoutingDecision)
		want   string
	}{
		{"invented derivation", func(d *RecipeRoutingDecision) { d.Derivation = "hand-tuned/v0" }, "unregistered routing derivation"},
		{"foreign selection", func(d *RecipeRoutingDecision) {
			d.Selected = testutil.ArtifactID(t, artifact.KindRecipe, "never-considered")
		}, "must name a considered candidate"},
		{"empty candidate set", func(d *RecipeRoutingDecision) { d.Candidates = nil }, "at least one candidate"},
		{"duplicate candidate", func(d *RecipeRoutingDecision) {
			d.Candidates = []RoutingCandidate{cheap, cheap}
		}, "duplicate routing candidate"},
		{"unjudged candidate", func(d *RecipeRoutingDecision) {
			unjudged := cheap
			unjudged.Evidence = nil
			d.Candidates = []RoutingCandidate{unjudged, costly}
		}, "judged on no evidence"},
		{"unmeasured resource", func(d *RecipeRoutingDecision) {
			free := cheap
			free.ResourceBytes = 0
			d.Candidates = []RoutingCandidate{free, costly}
		}, "finite measured quality and resource"},
		{"unbounded threshold", func(d *RecipeRoutingDecision) {
			d.Signal.QualityThreshold = math.Inf(1)
		}, "finite bound derived from exact evidence"},
		{"threshold without evidence", func(d *RecipeRoutingDecision) {
			d.Signal.ThresholdEvidence = artifact.ID{}
		}, "finite bound derived from exact evidence"},
		{"invalid task", func(d *RecipeRoutingDecision) { d.Signal.Task = "weight-surgery" }, "invalid routing task"},
		{"unnamed capability", func(d *RecipeRoutingDecision) { d.Signal.Capability = "" }, "bounded capability"},
	}
	for _, refusal := range refusals {
		mutated := template
		mutated.Candidates = []RoutingCandidate{costly, cheap}
		refusal.mutate(&mutated)
		if _, err := NewRecipeRoutingDecision(mutated); err == nil || !strings.Contains(err.Error(), refusal.want) {
			t.Fatalf("%s admitted: %v", refusal.name, err)
		}
	}

	weak := routingCandidateFixture(t, "routing-weak", 512)
	weak.Quality = 0.4
	derived, err := DeriveRecipeRoutingDecision(template.Signal, []RoutingCandidate{costly, weak, cheap})
	if err != nil {
		t.Fatal(err)
	}
	if derived.Selected != cheap.Recipe || derived.Derivation != RoutingDerivationCheapestEligible {
		t.Fatalf("derivation selected %s: cheapest below threshold must not win", derived.Selected)
	}
	strict := template.Signal
	strict.QualityThreshold = 0.9
	if _, err := DeriveRecipeRoutingDecision(strict, []RoutingCandidate{costly, cheap}); err == nil ||
		!strings.Contains(err.Error(), "meets the routing threshold") {
		t.Fatalf("unmet threshold selected a fallback: %v", err)
	}
}
