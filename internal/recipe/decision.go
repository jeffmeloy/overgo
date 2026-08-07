package recipe

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
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
	decision := Decision{
		Version: DecisionVersion, Subject: subject, Outcome: outcome, Tier: tier,
		Reason: reason, Decider: decider, Evidence: slices.Clone(evidence),
	}
	if err := canonicalizeDecision(&decision); err != nil {
		return Decision{}, err
	}
	content, err := decisionBytes(decision)
	if err != nil {
		return Decision{}, err
	}
	decision.ID, err = decisionContract.Identify(content)
	return decision, err
}

func ParseDecision(content []byte) (Decision, error) {
	var decision Decision
	if err := strictjson.DecodeBytes(content, &decision); err != nil {
		return Decision{}, fmt.Errorf("recipe: decode decision: %w", err)
	}
	parsed, err := NewDecision(
		decision.Subject, decision.Outcome, decision.Tier, decision.Reason,
		decision.Decider, decision.Evidence,
	)
	if err != nil {
		return Decision{}, err
	}
	canonical, err := parsed.ContentBytes()
	if err != nil {
		return Decision{}, err
	}
	if !bytes.Equal(canonical, content) {
		return Decision{}, errors.New("recipe: non-canonical decision content")
	}
	return parsed, nil
}

func (d Decision) ValidateIdentity() error {
	if d.ID.Kind() != artifact.KindEvidence {
		return errors.New("recipe: invalid decision identity")
	}
	canonical := d
	canonical.ID = artifact.ID{}
	if err := canonicalizeDecision(&canonical); err != nil {
		return err
	}
	if !sameDecision(d, canonical) {
		return errors.New("recipe: decision is not canonical")
	}
	content, err := decisionBytes(canonical)
	if err != nil {
		return err
	}
	if err := decisionContract.ValidateIdentity(d.ID, content); err != nil {
		return errors.New("recipe: decision identity mismatch")
	}
	return nil
}

func (d Decision) ContentBytes() ([]byte, error) {
	if err := d.ValidateIdentity(); err != nil {
		return nil, err
	}
	canonical := d
	canonical.ID = artifact.ID{}
	return decisionBytes(canonical)
}

func (d Decision) Content() (artifact.Content, error) {
	content, err := d.ContentBytes()
	if err != nil {
		return artifact.Content{}, err
	}
	return decisionContract.Content(d.ID, content)
}

func DecisionDocumentContract() artifact.DocumentContract { return decisionContract }

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

func sameDecision(left, right Decision) bool {
	right.ID = left.ID
	return left.Version == right.Version && left.ID == right.ID && left.Subject == right.Subject &&
		left.Outcome == right.Outcome && left.Tier == right.Tier && left.Reason == right.Reason &&
		left.Decider == right.Decider && slices.Equal(left.Evidence, right.Evidence)
}
