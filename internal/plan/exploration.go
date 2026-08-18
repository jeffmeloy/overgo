package plan

import (
	"errors"
	"fmt"

	"overgo/internal/artifact"
)

const (
	explorationGrantMediaType  = "application/vnd.overgo.exploration-grant+json"
	explorationGrantSchema     = "overgo/exploration-grant/v1"
	explorationChargeMediaType = "application/vnd.overgo.exploration-charge+json"
	explorationChargeSchema    = "overgo/exploration-charge/v1"
	explorationVersion         = uint16(1)
)

// ExplorationGrant is one GPU-time budget token for autonomous proposal work:
// an external authority issues a bounded number of GPU-minutes to one
// proposer identity. Consumption is derived from committed charges -- there
// is no mutable counter -- and exceeding the grant is never a local decision.
type ExplorationGrant struct {
	Version   uint16      `json:"version"`
	Proposer  artifact.ID `json:"proposer"`
	Authority artifact.ID `json:"authority"`
	// IssuedGPUMinutes is the whole grant; minutes keep arithmetic integral
	// and deterministic.
	IssuedGPUMinutes uint64      `json:"issued_gpu_minutes"`
	ID               artifact.ID `json:"-"`
}

// ExplorationCharge consumes part of a grant for one experiment.
type ExplorationCharge struct {
	Version    uint16      `json:"version"`
	Grant      artifact.ID `json:"grant"`
	Experiment artifact.ID `json:"experiment"`
	GPUMinutes uint64      `json:"gpu_minutes"`
	ID         artifact.ID `json:"-"`
}

var explorationGrantCodec = artifact.JSONDocumentCodec(
	"exploration grant", artifact.KindEvidence, explorationGrantMediaType, explorationGrantSchema,
	canonicalizeExplorationGrant,
	func(value ExplorationGrant) artifact.ID { return value.ID },
	func(value *ExplorationGrant, id artifact.ID) { value.ID = id }, nil,
)

var explorationChargeCodec = artifact.JSONDocumentCodec(
	"exploration charge", artifact.KindEvidence, explorationChargeMediaType, explorationChargeSchema,
	canonicalizeExplorationCharge,
	func(value ExplorationCharge) artifact.ID { return value.ID },
	func(value *ExplorationCharge, id artifact.ID) { value.ID = id }, nil,
)

func NewExplorationGrant(proposer, authority artifact.ID, issuedGPUMinutes uint64) (ExplorationGrant, error) {
	return explorationGrantCodec.New(ExplorationGrant{
		Version: explorationVersion, Proposer: proposer, Authority: authority,
		IssuedGPUMinutes: issuedGPUMinutes,
	})
}

func NewExplorationCharge(grant, experiment artifact.ID, gpuMinutes uint64) (ExplorationCharge, error) {
	return explorationChargeCodec.New(ExplorationCharge{
		Version: explorationVersion, Grant: grant, Experiment: experiment, GPUMinutes: gpuMinutes,
	})
}

func ParseExplorationGrant(content []byte) (ExplorationGrant, error) {
	return explorationGrantCodec.Parse(content)
}

func ParseExplorationCharge(content []byte) (ExplorationCharge, error) {
	return explorationChargeCodec.Parse(content)
}

func (g ExplorationGrant) Content() (artifact.Content, error) {
	return explorationGrantCodec.Content(g)
}

func (c ExplorationCharge) Content() (artifact.Content, error) {
	return explorationChargeCodec.Content(c)
}

func canonicalizeExplorationGrant(value *ExplorationGrant) error {
	if value == nil || value.Version != explorationVersion {
		return errors.New("plan: invalid exploration grant version")
	}
	if !value.Proposer.Valid() || !value.Authority.Valid() || value.Proposer == value.Authority {
		return errors.New("plan: exploration grant requires distinct proposer and issuing authority")
	}
	if value.IssuedGPUMinutes == 0 {
		return errors.New("plan: exploration grant issues no time")
	}
	return nil
}

func canonicalizeExplorationCharge(value *ExplorationCharge) error {
	if value == nil || value.Version != explorationVersion {
		return errors.New("plan: invalid exploration charge version")
	}
	if !value.Grant.Valid() || !value.Experiment.Valid() || value.GPUMinutes == 0 {
		return errors.New("plan: exploration charge requires a grant, an experiment and time")
	}
	return nil
}

// ExplorationAdmission is the deterministic verdict on one requested spend.
// Admitted and RequiresExternalDecision are never both true: exceeding the
// grant, or any request while contained, is refused locally and escalated --
// the local system cannot authorize its own overrun.
type ExplorationAdmission struct {
	Admitted                 bool
	RemainingGPUMinutes      uint64
	RequiresExternalDecision bool
	Reason                   string
}

// ExplorationBalance folds the committed charges against one grant. Charges
// for other grants are an error, and over-consumption is an error -- a
// corrupt record set is reported, never rounded away.
func ExplorationBalance(grant ExplorationGrant, charges []ExplorationCharge) (uint64, error) {
	if err := grant.ValidateIdentity(); err != nil {
		return 0, err
	}
	consumed := uint64(0)
	for _, charge := range charges {
		if charge.Grant != grant.ID {
			return 0, fmt.Errorf("plan: charge %s belongs to grant %s, not %s", charge.ID, charge.Grant, grant.ID)
		}
		if charge.GPUMinutes > grant.IssuedGPUMinutes-consumed {
			return 0, fmt.Errorf("plan: committed charges exceed grant %s: %d + %d > %d",
				grant.ID, consumed, charge.GPUMinutes, grant.IssuedGPUMinutes)
		}
		consumed += charge.GPUMinutes
	}
	return grant.IssuedGPUMinutes - consumed, nil
}

// AdmitExploration judges a requested spend against the grant's derived
// balance. Containment refuses all further spend regardless of balance.
func AdmitExploration(
	grant ExplorationGrant,
	charges []ExplorationCharge,
	requestedGPUMinutes uint64,
	contained bool,
) (ExplorationAdmission, error) {
	remaining, err := ExplorationBalance(grant, charges)
	if err != nil {
		return ExplorationAdmission{}, err
	}
	if contained {
		return ExplorationAdmission{
			RemainingGPUMinutes: remaining, RequiresExternalDecision: true,
			Reason: "proposer is contained; exploration spend requires an external decision",
		}, nil
	}
	if requestedGPUMinutes == 0 {
		return ExplorationAdmission{}, errors.New("plan: requested exploration spend must be positive")
	}
	if requestedGPUMinutes > remaining {
		return ExplorationAdmission{
			RemainingGPUMinutes: remaining, RequiresExternalDecision: true,
			Reason: fmt.Sprintf("requested %d GPU-minutes exceeds the remaining grant of %d; exceeding a budget requires an external decision",
				requestedGPUMinutes, remaining),
		}, nil
	}
	return ExplorationAdmission{
		Admitted:            true,
		RemainingGPUMinutes: remaining - requestedGPUMinutes,
		Reason:              fmt.Sprintf("admitted %d GPU-minutes; %d remain", requestedGPUMinutes, remaining-requestedGPUMinutes),
	}, nil
}

func (g ExplorationGrant) ValidateIdentity() error {
	return explorationGrantCodec.ValidateIdentity(g)
}
