package composition

import (
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/organ"
	"overgo/internal/recipe"
	"overgo/internal/tensorstats"
	"overgo/internal/testutil"
)

// TestProposalRankingImprovesOnRefusalHistory pins the learned prior: plain
// similarity retrieval prefers the donor family whose contract exactly
// matches the query, but that family's bridges have been refused repeatedly
// while a near-miss family shipped. Training on the decision ledger must flip
// the ranking toward the family that composes successfully, stay identical
// under observation reorder (content-addressed determinism), refuse histories
// without contrast, and enforce proposer blinding by rejecting any outcome
// that is not an accepted or refused composition decision.
func TestTermProposalRankingImprovesOnRefusalHistory(t *testing.T) {
	statistics := &tensorstats.Characterization{
		Median: 0.001, InterquartileRange: 0.02,
		LMoments: tensorstats.LMoments{L2: 0.01, Tau3: 0.1, Tau4: 0.2},
	}
	gate := func(model artifact.ID, name, dtype string) CatalogComponent {
		return CatalogComponent{
			Model: model, Name: name,
			Contract: organ.Classify(name, dtype, "qwen2", "text", ""), Statistics: statistics,
		}
	}
	target := testutil.ArtifactID(t, artifact.KindModel, "ranking-target")
	query := gate(target, "blk.8.ffn_gate.weight", "f16")
	// The refused family matches the query contract exactly; the shipped
	// family differs only in dtype, so similarity alone ranks it second.
	refusedFamily := []CatalogComponent{
		gate(testutil.ArtifactID(t, artifact.KindModel, "ranking-refused-1"), "blk.10.ffn_gate.weight", "f16"),
		gate(testutil.ArtifactID(t, artifact.KindModel, "ranking-refused-2"), "blk.12.ffn_gate.weight", "f16"),
	}
	shippedFamily := []CatalogComponent{
		gate(testutil.ArtifactID(t, artifact.KindModel, "ranking-shipped-1"), "blk.11.ffn_gate.weight", "bf16"),
		gate(testutil.ArtifactID(t, artifact.KindModel, "ranking-shipped-2"), "blk.13.ffn_gate.weight", "bf16"),
	}
	index, err := NewHypervectorIndex(append(append([]CatalogComponent(nil), refusedFamily...), shippedFamily...))
	if err != nil {
		t.Fatal(err)
	}
	hits, err := index.Search(query, 4)
	if err != nil {
		t.Fatal(err)
	}
	refusedDType, shippedDType := refusedFamily[0].Contract.DType, shippedFamily[0].Contract.DType
	if refusedDType == shippedDType || query.Contract.DType != refusedDType {
		t.Fatalf("protocol broken: query dtype %s, families %s vs %s must differ", query.Contract.DType, refusedDType, shippedDType)
	}
	if hits[0].Component.Contract.DType != refusedDType {
		t.Fatalf("similarity baseline top hit dtype = %s, want the exact-contract %s family", hits[0].Component.Contract.DType, refusedDType)
	}

	decider := recipe.Decider{
		CodeCommit: strings.Repeat("ab", 20),
		Derivation: testutil.ArtifactID(t, artifact.KindEvidence, "ranking-decider"),
	}
	evidence := testutil.ArtifactID(t, artifact.KindEvidence, "ranking-measurement")
	decide := func(seed string, outcome recipe.DecisionOutcome, reason string, evidence []artifact.ID) recipe.Decision {
		decision, err := recipe.NewDecision(
			testutil.ArtifactID(t, artifact.KindRecipe, seed), outcome,
			recipe.EvidenceExperimental, reason, decider, evidence,
		)
		if err != nil {
			t.Fatal(err)
		}
		return decision
	}
	observations := make([]RankingObservation, 0, 8)
	for round := 0; round < 2; round++ {
		for i, component := range refusedFamily {
			observations = append(observations, RankingObservation{
				Decision: decide(
					"refuse-"+string(rune('a'+round))+string(rune('0'+i)), recipe.DecisionRefused,
					"bridge refused: held-out regression on every seed", []artifact.ID{evidence},
				),
				Component: component,
			})
		}
		for i, component := range shippedFamily {
			observations = append(observations, RankingObservation{
				Decision:  decide("ship-"+string(rune('a'+round))+string(rune('0'+i)), recipe.DecisionAccepted, "", nil),
				Component: component,
			})
		}
	}
	ranking, err := TrainProposalRanking(observations)
	if err != nil {
		t.Fatal(err)
	}
	reversed := make([]RankingObservation, len(observations))
	for i, observation := range observations {
		reversed[len(observations)-1-i] = observation
	}
	replay, err := TrainProposalRanking(reversed)
	if err != nil || replay.ID != ranking.ID {
		t.Fatalf("ranking identity not order-canonical: (%v, %v)", replay.ID, err)
	}

	ranked := ranking.Rerank(hits)
	if ranked[0].Component.Contract.DType != shippedDType {
		t.Fatalf("learned ranking top hit dtype = %s, want the historically shipped %s family", ranked[0].Component.Contract.DType, shippedDType)
	}
	for _, hit := range ranked {
		refused := hit.Component.Contract.DType == refusedDType
		if refused && hit.Prior >= 0 || !refused && hit.Prior <= 0 {
			t.Fatalf("prior %f has the wrong sign for %s (dtype %s)", hit.Prior, hit.Component.Name, hit.Component.Contract.DType)
		}
		if hit.Score != hit.Relevance+hit.Prior {
			t.Fatalf("score %f is not relevance %f plus prior %f", hit.Score, hit.Relevance, hit.Prior)
		}
	}

	content, err := ranking.Content()
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := ParseProposalRanking(content.Data)
	if err != nil || parsed.ID != ranking.ID || len(parsed.Sources) != len(observations) {
		t.Fatalf("roundtrip = (%v, %v), want %d sources", parsed.ID, err, len(observations))
	}
	if replayed := parsed.Rerank(hits); replayed[0].Component.Name != ranked[0].Component.Name {
		t.Fatalf("parsed ranking reranks differently: %s", replayed[0].Component.Name)
	}

	if _, err := TrainProposalRanking(nil); err == nil {
		t.Fatal("empty history accepted")
	}
	if _, err := TrainProposalRanking(observations[:2]); err == nil {
		t.Fatal("history without contrast accepted (all refusals)")
	}
	blinded := RankingObservation{
		Decision:  decide("audit-observation", recipe.DecisionObserved, "audit-split observation", nil),
		Component: refusedFamily[0],
	}
	if _, err := TrainProposalRanking(append(append([]RankingObservation(nil), observations...), blinded)); err == nil {
		t.Fatal("proposer blinding violated: an observed (audit) outcome trained the ranking")
	}
}
