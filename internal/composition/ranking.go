// Learned retrieval prior. The decision ledger is the training set.
package composition

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	ProposalRankingVersion   uint16 = 1
	ProposalRankingMediaType        = "application/vnd.overgo.proposal-ranking+json"
	ProposalRankingSchema           = "overgo/proposal-ranking/v1"
)

// RankingObservation pairs one committed composition decision with the
// catalog component it judged. The decision supplies the label; the
// component supplies the signature terms the label attaches to.
type RankingObservation struct {
	Decision  recipe.Decision
	Component CatalogComponent
}

// TermOutcome is the per-term outcome tally: how many accepted and refused
// composition decisions carried this signature term. Counts are the stored
// form; log-odds are derived at query time so the artifact stays integral.
type TermOutcome struct {
	Term     string `json:"term"`
	Accepted uint32 `json:"accepted"`
	Refused  uint32 `json:"refused"`
}

// ProposalRanking is the learned prior over composition success: outcome
// tallies in the same term space the hypervector index encodes, trained
// solely on accepted and refused composition decisions. Proposer blinding is
// structural -- the trainer takes decision documents, never promotion or
// audit-split results, so the ranking can only ever learn from the refusal
// ledger it is allowed to see.
type ProposalRanking struct {
	Version uint16        `json:"version"`
	Terms   []TermOutcome `json:"terms"`
	// Sources are the decision artifacts the tallies derive from.
	Sources []artifact.ID `json:"sources"`
	ID      artifact.ID   `json:"-"`
}

var proposalRankingCodec = artifact.JSONDocumentCodec(
	"proposal ranking", artifact.KindEvidence, ProposalRankingMediaType, ProposalRankingSchema,
	canonicalizeProposalRanking,
	func(value ProposalRanking) artifact.ID { return value.ID },
	func(value *ProposalRanking, id artifact.ID) { value.ID = id },
	func(value ProposalRanking) ProposalRanking {
		value.Terms = slices.Clone(value.Terms)
		value.Sources = slices.Clone(value.Sources)
		return value
	},
)

// TrainProposalRanking tallies signature-term outcomes from composition
// decisions. Only accepted and refused outcomes are admissible: any other
// document is rejected, which is what keeps the proposer blind to promotion
// and audit-split machinery. Learning requires contrast -- a history that is
// all refusals or all ships carries no ranking signal and is refused.
func TrainProposalRanking(observations []RankingObservation) (ProposalRanking, error) {
	if len(observations) == 0 {
		return ProposalRanking{}, errors.New("composition: ranking requires observations")
	}
	tallies := map[string]*TermOutcome{}
	sources := map[artifact.ID]bool{}
	var accepted, refused int
	for _, observation := range observations {
		decision := observation.Decision
		if !decision.ID.Valid() {
			return ProposalRanking{}, errors.New("composition: ranking observation requires an identified decision")
		}
		if decision.Outcome != recipe.DecisionAccepted && decision.Outcome != recipe.DecisionRefused {
			return ProposalRanking{}, fmt.Errorf(
				"composition: proposer blinding admits only accepted and refused composition decisions, got %q", decision.Outcome)
		}
		component := observation.Component
		if component.Name == "" || !component.Model.Valid() {
			return ProposalRanking{}, errors.New("composition: ranking observation requires a model and a component name")
		}
		if decision.Outcome == recipe.DecisionAccepted {
			accepted++
		} else {
			refused++
		}
		sources[decision.ID] = true
		seen := map[string]bool{}
		for _, term := range componentTerms(component) {
			if seen[term] {
				continue
			}
			seen[term] = true
			tally := tallies[term]
			if tally == nil {
				tally = &TermOutcome{Term: term}
				tallies[term] = tally
			}
			if decision.Outcome == recipe.DecisionAccepted {
				tally.Accepted++
			} else {
				tally.Refused++
			}
		}
	}
	if accepted == 0 || refused == 0 {
		return ProposalRanking{}, errors.New("composition: ranking requires both accepted and refused history to learn contrast")
	}
	ranking := ProposalRanking{Version: ProposalRankingVersion}
	for _, tally := range tallies {
		ranking.Terms = append(ranking.Terms, *tally)
	}
	for source := range sources {
		ranking.Sources = append(ranking.Sources, source)
	}
	return proposalRankingCodec.New(ranking)
}

func ParseProposalRanking(content []byte) (ProposalRanking, error) {
	return proposalRankingCodec.Parse(content)
}

func (r ProposalRanking) Content() (artifact.Content, error) {
	return proposalRankingCodec.Content(r)
}

// Batch commits the ranking with lineage to every decision it learned from.
func (r ProposalRanking) Batch(key string) (artifact.Batch, error) {
	return proposalRankingCodec.Batch(key, r, artifact.DependencyLineage(r.ID, r.Sources...), nil)
}

// RankedHit is one retrieval hit rescored by the learned prior. Relevance is
// the similarity signal, Prior the outcome-history signal, Score their sum.
type RankedHit struct {
	HypervectorHit
	Prior float64
	Score float64
}

// Rerank rescores retrieval hits with the learned prior: each hit's score is
// its similarity relevance plus the mean smoothed log-odds of acceptance over
// the component's signature terms. Terms without history contribute zero, so
// where the ledger is silent the ranking degrades to plain similarity.
func (r ProposalRanking) Rerank(hits []HypervectorHit) []RankedHit {
	odds := make(map[string]float64, len(r.Terms))
	for _, term := range r.Terms {
		odds[term.Term] = math.Log(float64(term.Accepted+1) / float64(term.Refused+1))
	}
	ranked := make([]RankedHit, len(hits))
	for i, hit := range hits {
		terms := componentTerms(hit.Component)
		prior := 0.0
		for _, term := range terms {
			prior += odds[term]
		}
		if len(terms) > 0 {
			prior /= float64(len(terms))
		}
		ranked[i] = RankedHit{HypervectorHit: hit, Prior: prior, Score: hit.Relevance + prior}
	}
	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].Score != ranked[j].Score {
			return ranked[i].Score > ranked[j].Score
		}
		if ranked[i].Component.Name != ranked[j].Component.Name {
			return ranked[i].Component.Name < ranked[j].Component.Name
		}
		return ranked[i].Component.Model.String() < ranked[j].Component.Model.String()
	})
	return ranked
}

func canonicalizeProposalRanking(value *ProposalRanking) error {
	if value == nil || value.Version != ProposalRankingVersion {
		return errors.New("composition: invalid proposal ranking version")
	}
	if len(value.Terms) == 0 || len(value.Sources) == 0 {
		return errors.New("composition: proposal ranking requires terms and source decisions")
	}
	var accepted, refused uint64
	for _, term := range value.Terms {
		if term.Term == "" || term.Accepted == 0 && term.Refused == 0 {
			return errors.New("composition: proposal ranking term requires a name and at least one outcome")
		}
		accepted += uint64(term.Accepted)
		refused += uint64(term.Refused)
	}
	if accepted == 0 || refused == 0 {
		return errors.New("composition: proposal ranking requires both accepted and refused history")
	}
	sort.Slice(value.Terms, func(i, j int) bool { return value.Terms[i].Term < value.Terms[j].Term })
	for i := 1; i < len(value.Terms); i++ {
		if value.Terms[i].Term == value.Terms[i-1].Term {
			return errors.New("composition: proposal ranking terms must be unique")
		}
	}
	for _, source := range value.Sources {
		if !source.Valid() {
			return errors.New("composition: proposal ranking source must be a valid artifact")
		}
	}
	sort.Slice(value.Sources, func(i, j int) bool {
		return value.Sources[i].String() < value.Sources[j].String()
	})
	value.Sources = slices.Compact(value.Sources)
	return nil
}
