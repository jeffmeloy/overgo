package composition

import (
	"errors"
	"math"
	"slices"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/testevidence"
)

const (
	BridgeProposalVersion   uint16 = 1
	BridgeProposalMediaType        = "application/vnd.overgo.bridge-proposal+json"
	BridgeProposalSchema           = "overgo/bridge-proposal/v1"
	// ProposalPromotionBlocked is the ONLY representable proposal state:
	// a bridge proposal is advisory by construction and can never authorize
	// work. Promotion happens only through the experiment plane's own
	// admission, never through this document.
	ProposalPromotionBlocked = "promotion-blocked"
)

// BridgeCandidate: retrieved cross-model component pair.
type BridgeCandidate struct {
	Donor     artifact.ID `json:"donor"`
	Component string      `json:"component"`
	Distance  float64     `json:"distance"`
}

// BridgeProposal: ranked, blocked composition candidate.
type BridgeProposal struct {
	Version    uint16            `json:"version"`
	Target     artifact.ID       `json:"target"`
	Candidates []BridgeCandidate `json:"candidates"`
	// RequiredVerifier: failable experiment check.
	RequiredVerifier string      `json:"required_verifier"`
	State            string      `json:"state"`
	Blocker          string      `json:"blocker"`
	Ranker           artifact.ID `json:"ranker"`
	ID               artifact.ID `json:"-"`
}

var bridgeProposalCodec = artifact.JSONDocumentCodec(
	"bridge proposal", artifact.KindEvidence, BridgeProposalMediaType, BridgeProposalSchema,
	canonicalizeBridgeProposal,
	func(value BridgeProposal) artifact.ID { return value.ID },
	func(value *BridgeProposal, id artifact.ID) { value.ID = id },
	func(value BridgeProposal) BridgeProposal {
		value.Candidates = slices.Clone(value.Candidates)
		return value
	},
)

func NewBridgeProposal(
	target artifact.ID,
	candidates []BridgeCandidate,
	ranker ProposalRanker,
	requiredVerifier, blocker string,
) (BridgeProposal, error) {
	if err := proposalRankerCodec.ValidateIdentity(ranker); err != nil {
		return BridgeProposal{}, err
	}
	return bridgeProposalCodec.New(BridgeProposal{
		Version: BridgeProposalVersion, Target: target,
		Candidates: ranker.Rank(candidates), RequiredVerifier: requiredVerifier,
		State: ProposalPromotionBlocked, Blocker: blocker, Ranker: ranker.ID,
	})
}

func ParseBridgeProposal(content []byte) (BridgeProposal, error) {
	return bridgeProposalCodec.Parse(content)
}

func (p BridgeProposal) Content() (artifact.Content, error) {
	return bridgeProposalCodec.Content(p)
}

func (p BridgeProposal) Lineage() []artifact.Lineage {
	lineage := artifact.DependencyLineage(p.ID, p.Target, p.Ranker)
	for _, candidate := range p.Candidates {
		lineage = append(lineage, artifact.Lineage{
			Child: p.ID, Parent: candidate.Donor, Relation: artifact.RelationDependsOn,
		})
	}
	return lineage
}

func (p BridgeProposal) Batch(key string) (artifact.Batch, error) {
	return bridgeProposalCodec.Batch(key, p, p.Lineage(), nil)
}

func canonicalizeBridgeProposal(value *BridgeProposal) error {
	if value == nil || value.Version != BridgeProposalVersion {
		return errors.New("composition: invalid bridge proposal version")
	}
	if value.Target.Kind() != artifact.KindModel {
		return errors.New("composition: proposal target must be a model")
	}
	if value.Ranker.Kind() != artifact.KindProfile {
		return errors.New("composition: proposal requires a ranking authority")
	}
	if len(value.Candidates) == 0 {
		return errors.New("composition: proposal carries no candidates")
	}
	for _, candidate := range value.Candidates {
		if candidate.Donor.Kind() != artifact.KindModel || strings.TrimSpace(candidate.Component) == "" ||
			candidate.Distance < 0 || math.IsNaN(candidate.Distance) || math.IsInf(candidate.Distance, 0) {
			return errors.New("composition: invalid proposal candidate")
		}
	}
	if err := testevidence.ValidateGoTestCommand(value.RequiredVerifier); err != nil {
		return errors.New("composition: proposal requires a failable verifier: " + err.Error())
	}
	if value.State != ProposalPromotionBlocked {
		return errors.New("composition: a bridge proposal is promotion-blocked by construction")
	}
	if strings.TrimSpace(value.Blocker) == "" {
		return errors.New("composition: proposal requires its blocker reason")
	}
	return nil
}
