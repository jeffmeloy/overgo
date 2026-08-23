package composition

import (
	"context"
	"errors"
	"fmt"
	"math/bits"
	"slices"
	"sort"
	"strings"
	"unicode"

	"overgo/internal/artifact"
	"overgo/internal/organ"
	"overgo/internal/recipe"
	"overgo/internal/repodb"
)

const (
	ProposalRankerVersion   uint16 = 1
	ProposalRankerMediaType        = "application/vnd.overgo.proposal-ranker+json"
	ProposalRankerSchema           = "overgo/proposal-ranker/v1"
)

type ProposalRankingRow struct {
	Signature string        `json:"signature"`
	Observed  uint64        `json:"observed"`
	Refused   uint64        `json:"refused"`
	Reasons   []artifact.ID `json:"reasons,omitempty"`
}

// ProposalRanker: refusal-ledger prior; no promotion evidence.
type ProposalRanker struct {
	Version   uint16               `json:"version"`
	Decisions []artifact.ID        `json:"decisions,omitempty"`
	Rows      []ProposalRankingRow `json:"rows,omitempty"`
	ID        artifact.ID          `json:"-"`
}

var proposalRankerCodec = artifact.JSONDocumentCodec(
	"proposal ranker", artifact.KindProfile, ProposalRankerMediaType, ProposalRankerSchema,
	canonicalizeProposalRanker,
	func(value ProposalRanker) artifact.ID { return value.ID },
	func(value *ProposalRanker, id artifact.ID) { value.ID = id },
	func(value ProposalRanker) ProposalRanker {
		value.Decisions = slices.Clone(value.Decisions)
		value.Rows = slices.Clone(value.Rows)
		for index := range value.Rows {
			value.Rows[index].Reasons = slices.Clone(value.Rows[index].Reasons)
		}
		return value
	},
)

func TrainProposalRanker(ctx context.Context, reader artifact.Reader, decisions []artifact.ID) (ProposalRanker, error) {
	ids := slices.Clone(decisions)
	sort.Slice(ids, func(left, right int) bool { return ids[left].String() < ids[right].String() })
	ids = slices.Compact(ids)
	if len(ids) != 0 && (ctx == nil || reader == nil) {
		return ProposalRanker{}, errors.New("composition: proposal ranker requires its refusal ledger")
	}
	rows := map[string]ProposalRankingRow{}
	for _, id := range ids {
		decision, proposal, err := loadProposalDecision(ctx, reader, id)
		if err != nil {
			return ProposalRanker{}, err
		}
		for _, candidate := range proposal.Candidates {
			signature := candidateSignature(candidate)
			row := rows[signature]
			row.Signature = signature
			switch decision.Outcome {
			case recipe.DecisionObserved:
				row.Observed++
			case recipe.DecisionRefused:
				row.Refused++
				reason, err := artifact.JSONID(artifact.KindProfile, decision.Reason)
				if err != nil {
					return ProposalRanker{}, err
				}
				row.Reasons = append(row.Reasons, reason)
			default:
				return ProposalRanker{}, fmt.Errorf("composition: decision %s is not blinded trial evidence", id)
			}
			rows[signature] = row
		}
	}
	compiled := make([]ProposalRankingRow, 0, len(rows))
	for _, row := range rows {
		sort.Slice(row.Reasons, func(left, right int) bool { return row.Reasons[left].String() < row.Reasons[right].String() })
		row.Reasons = slices.Compact(row.Reasons)
		compiled = append(compiled, row)
	}
	sort.Slice(compiled, func(left, right int) bool { return compiled[left].Signature < compiled[right].Signature })
	return proposalRankerCodec.New(ProposalRanker{Version: ProposalRankerVersion, Decisions: ids, Rows: compiled})
}

func TrainProposalRankerFromStore(ctx context.Context, store *repodb.Store) (ProposalRanker, error) {
	if ctx == nil || store == nil {
		return ProposalRanker{}, errors.New("composition: proposal ranker store absent")
	}
	result, err := store.Query(ctx, repodb.Query{
		Kind: artifact.KindEvidence, MaxResults: store.QueryExtent(),
		Projection: repodb.ProjectArtifacts | repodb.ProjectContentData,
	})
	if err != nil {
		return ProposalRanker{}, err
	}
	history := make([]artifact.ID, 0)
	for _, descriptor := range result.Artifacts {
		if descriptor.MediaType != recipe.DecisionMediaType {
			continue
		}
		content, ok := result.Content(descriptor.ID)
		if !ok {
			continue
		}
		decision, err := recipe.ParseDecision(content)
		if err != nil || decision.Outcome != recipe.DecisionObserved && decision.Outcome != recipe.DecisionRefused {
			continue
		}
		subject, ok, err := artifact.ReadContent(ctx, store, decision.Subject)
		if err != nil {
			return ProposalRanker{}, err
		}
		if ok && subject.Descriptor.MediaType == BridgeProposalMediaType {
			history = append(history, descriptor.ID)
		}
	}
	return TrainProposalRanker(ctx, store, history)
}

func (r ProposalRanker) Content() (artifact.Content, error) { return proposalRankerCodec.Content(r) }

func (r ProposalRanker) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(r.ID, r.Decisions...)
}

func (r ProposalRanker) Rank(candidates []BridgeCandidate) []BridgeCandidate {
	result := slices.Clone(candidates)
	sort.SliceStable(result, func(left, right int) bool {
		leftRow := r.row(candidateSignature(result[left]))
		rightRow := r.row(candidateSignature(result[right]))
		if order := compareOutcomeEvidence(leftRow, rightRow); order != 0 {
			return order > 0
		}
		if result[left].Distance != result[right].Distance {
			return result[left].Distance < result[right].Distance
		}
		if result[left].Component != result[right].Component {
			return result[left].Component < result[right].Component
		}
		return result[left].Donor.String() < result[right].Donor.String()
	})
	return result
}

func (r ProposalRanker) row(signature string) ProposalRankingRow {
	index, found := slices.BinarySearchFunc(r.Rows, signature, func(row ProposalRankingRow, target string) int {
		return strings.Compare(row.Signature, target)
	})
	if !found {
		return ProposalRankingRow{}
	}
	return r.Rows[index]
}

func canonicalizeProposalRanker(value *ProposalRanker) error {
	if value.Version != ProposalRankerVersion {
		return errors.New("composition: invalid proposal ranker version")
	}
	for index, id := range value.Decisions {
		if id.Kind() != artifact.KindEvidence || index > 0 && value.Decisions[index-1].String() >= id.String() {
			return errors.New("composition: invalid proposal ranker decision order")
		}
	}
	for index := range value.Rows {
		row := &value.Rows[index]
		if row.Signature == "" || row.Observed+row.Refused == 0 || index > 0 && value.Rows[index-1].Signature >= row.Signature {
			return errors.New("composition: invalid proposal ranker row")
		}
		for reasonIndex, reason := range row.Reasons {
			if reason.Kind() != artifact.KindProfile || reasonIndex > 0 && row.Reasons[reasonIndex-1].String() >= reason.String() {
				return errors.New("composition: invalid proposal refusal reason")
			}
		}
	}
	return nil
}

func loadProposalDecision(ctx context.Context, reader artifact.Reader, id artifact.ID) (recipe.Decision, BridgeProposal, error) {
	content, ok, err := artifact.ReadContent(ctx, reader, id)
	if err != nil || !ok || content.Descriptor.MediaType != recipe.DecisionMediaType {
		return recipe.Decision{}, BridgeProposal{}, errors.Join(err, errors.New("composition: proposal decision absent"))
	}
	decision, err := recipe.ParseDecision(content.Data)
	if err != nil || len(decision.Evidence) == 0 {
		return recipe.Decision{}, BridgeProposal{}, errors.Join(err, errors.New("composition: proposal decision lacks measured evidence"))
	}
	content, ok, err = artifact.ReadContent(ctx, reader, decision.Subject)
	if err != nil || !ok || content.Descriptor.MediaType != BridgeProposalMediaType {
		return recipe.Decision{}, BridgeProposal{}, errors.Join(err, errors.New("composition: decision subject is not a bridge proposal"))
	}
	proposal, err := ParseBridgeProposal(content.Data)
	return decision, proposal, err
}

func candidateSignature(candidate BridgeCandidate) string {
	contract := organ.Classify(candidate.Component, "", "", "", "")
	terms := []string{"role:" + string(contract.Role), "space:" + string(contract.Space), "layout:" + string(contract.Layout)}
	for _, token := range strings.FieldsFunc(strings.ToLower(candidate.Component), func(value rune) bool {
		return !unicode.IsLetter(value) && !unicode.IsDigit(value)
	}) {
		if strings.Trim(token, "0123456789") != "" {
			terms = append(terms, "name:"+token)
		}
	}
	return strings.Join(terms, "|")
}

func compareOutcomeEvidence(left, right ProposalRankingRow) int {
	leftTier := compare(left.Observed, left.Refused)
	rightTier := compare(right.Observed, right.Refused)
	if leftTier != rightTier {
		return leftTier - rightTier
	}
	leftTotal, rightTotal := left.Observed+left.Refused, right.Observed+right.Refused
	if leftTotal == 0 || rightTotal == 0 {
		return 0
	}
	leftHigh, leftLow := bits.Mul64(left.Observed, rightTotal)
	rightHigh, rightLow := bits.Mul64(right.Observed, leftTotal)
	if leftHigh != rightHigh {
		return compare(leftHigh, rightHigh)
	}
	return compare(leftLow, rightLow)
}

func compare(left, right uint64) int {
	if left < right {
		return -1
	}
	if left > right {
		return 1
	}
	return 0
}
