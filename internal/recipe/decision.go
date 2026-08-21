package recipe

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"slices"
	"sort"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	DecisionVersion        uint16 = 1
	DecisionMediaType             = "application/vnd.overgo.decision+json"
	DecisionSchema                = "overgo/decision/v1"
	maxDecisionReasonBytes        = 4 << 10
)

var decisionContract = artifact.DocumentContract{
	Kind: artifact.KindEvidence, MediaType: DecisionMediaType, Schema: DecisionSchema,
}

var decisionCodec = artifact.DocumentCodec[Decision]{
	Name: "recipe decision", Contract: decisionContract,
	Decode: func(data []byte, value *Decision) error { return strictjson.DecodeBytes(data, value) },
	Encode: decisionBytes, Canonicalize: canonicalizeDecision,
	Clone: func(value Decision) Decision {
		value.Evidence = slices.Clone(value.Evidence)
		return value
	},
	Identity:    func(value Decision) artifact.ID { return value.ID },
	SetIdentity: func(value *Decision, id artifact.ID) { value.ID = id },
}

type DecisionOutcome string

const (
	DecisionUnknown      DecisionOutcome = "unknown"
	DecisionObserved     DecisionOutcome = "observed"
	DecisionAccepted     DecisionOutcome = "accepted"
	DecisionFailed       DecisionOutcome = "failed"
	DecisionInapplicable DecisionOutcome = "inapplicable"
	DecisionRefused      DecisionOutcome = "refused"
)

type EvidenceTier string

const (
	EvidenceExperimental EvidenceTier = "experimental"
	EvidenceParity       EvidenceTier = "parity"
	EvidenceProduction   EvidenceTier = "production"
)

func (t EvidenceTier) Valid() bool { return validEvidenceTier(t) }

// StrongerThan reports strict evidence precedence.
func (t EvidenceTier) StrongerThan(other EvidenceTier) bool {
	switch t {
	case EvidenceProduction:
		return other == EvidenceParity || other == EvidenceExperimental
	case EvidenceParity:
		return other == EvidenceExperimental
	default:
		return false
	}
}

type Decider struct {
	CodeCommit string      `json:"code_commit"`
	Derivation artifact.ID `json:"derivation"`
}

type Decision struct {
	Version  uint16          `json:"version"`
	ID       artifact.ID     `json:"-"`
	Subject  artifact.ID     `json:"subject"`
	Outcome  DecisionOutcome `json:"outcome"`
	Tier     EvidenceTier    `json:"tier"`
	Reason   string          `json:"reason,omitempty"`
	Decider  Decider         `json:"decider"`
	Evidence []artifact.ID   `json:"evidence,omitempty"`
}

func NewDecision(
	subject artifact.ID,
	outcome DecisionOutcome,
	tier EvidenceTier,
	reason string,
	decider Decider,
	evidence []artifact.ID,
) (Decision, error) {
	return decisionCodec.New(Decision{
		Version: DecisionVersion, Subject: subject, Outcome: outcome, Tier: tier,
		Reason: reason, Decider: decider, Evidence: slices.Clone(evidence),
	})
}

func ParseDecision(content []byte) (Decision, error) {
	return decisionCodec.Parse(content)
}

func (d Decision) ValidateIdentity() error {
	return decisionCodec.ValidateIdentity(d)
}

func (d Decision) Content() (artifact.Content, error) {
	return decisionCodec.Content(d)
}

func (d Decision) Lineage() []artifact.Lineage {
	parents := make([]artifact.ID, 0, len(d.Evidence)+2)
	if d.Subject.Valid() {
		parents = append(parents, d.Subject)
	}
	if d.Decider.Derivation.Valid() && d.Decider.Derivation != d.Subject {
		parents = append(parents, d.Decider.Derivation)
	}
	for _, parent := range d.Evidence {
		if parent.Valid() && parent != d.Subject && parent != d.Decider.Derivation {
			parents = append(parents, parent)
		}
	}
	return artifact.DependencyLineage(d.ID, parents...)
}

func (d Decision) Batch(key string) (artifact.Batch, error) {
	return decisionCodec.Batch(key, d, d.Lineage(), nil)
}

func canonicalizeDecision(decision *Decision) error {
	if decision == nil || decision.Version != DecisionVersion || !decision.Subject.Valid() {
		return errors.New("recipe: invalid decision envelope")
	}
	if !validDecisionOutcome(decision.Outcome) || !validEvidenceTier(decision.Tier) {
		return errors.New("recipe: invalid decision outcome or tier")
	}
	if strings.TrimSpace(decision.Reason) != decision.Reason || len(decision.Reason) > maxDecisionReasonBytes ||
		(decision.Outcome == DecisionFailed || decision.Outcome == DecisionInapplicable || decision.Outcome == DecisionRefused) && decision.Reason == "" {
		return errors.New("recipe: invalid decision reason")
	}
	// Refusals require measured evidence.
	if decision.Outcome == DecisionRefused && len(decision.Evidence) == 0 {
		return errors.New("recipe: refusal decision carries no measurement evidence")
	}
	if !validGitCommit(decision.Decider.CodeCommit) || !decision.Decider.Derivation.Valid() {
		return errors.New("recipe: invalid decision decider")
	}
	for _, evidence := range decision.Evidence {
		if !evidence.Valid() || evidence == decision.Subject {
			return errors.New("recipe: invalid decision evidence")
		}
	}
	sort.Slice(decision.Evidence, func(i, j int) bool {
		return decision.Evidence[i].String() < decision.Evidence[j].String()
	})
	decision.Evidence = slices.Compact(decision.Evidence)
	return nil
}

func validDecisionOutcome(outcome DecisionOutcome) bool {
	switch outcome {
	case DecisionUnknown, DecisionObserved, DecisionAccepted, DecisionFailed, DecisionInapplicable, DecisionRefused:
		return true
	default:
		return false
	}
}

func validEvidenceTier(tier EvidenceTier) bool {
	return tier == EvidenceExperimental || tier == EvidenceParity || tier == EvidenceProduction
}

func validGitCommit(value string) bool {
	if len(value) != 40 && len(value) != 64 || strings.ToLower(value) != value {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func decisionBytes(decision Decision) ([]byte, error) {
	return json.Marshal(decision)
}
