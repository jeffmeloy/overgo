package runrecord

import (
	"errors"
	"fmt"
	"strings"

	"overgo/internal/artifact"
	"overgo/internal/recipe"
)

const (
	AdmissionBindingMediaType = "application/vnd.overgo.admission-binding+json"
	AdmissionBindingSchema    = "overgo/admission-binding/v1"
)

// AuthorityDomain names one independent authority: a domain name (the
// organizational boundary -- worker pool, evaluation plane, promotion
// authority) plus the sealed identity that exercises it. Independence is a
// property of the DOMAIN pair, not merely of distinct artifact IDs.
type AuthorityDomain struct {
	Name     string      `json:"name"`
	Identity artifact.ID `json:"identity"`
}

// AdmissionBinding binds one generation's proposer, evaluator and decider to
// three distinct authority domains, names the sealed input set evaluations
// must run against and the clean-worker environment that re-evaluates, and --
// for every generation after the bootstrap -- carries the prior generation's
// approval of this generation's evaluator and policy.
type AdmissionBinding struct {
	Version    uint16          `json:"version"`
	Generation uint32          `json:"generation"`
	Proposer   AuthorityDomain `json:"proposer"`
	Evaluator  AuthorityDomain `json:"evaluator"`
	Decider    AuthorityDomain `json:"decider"`
	// SealedInputs is the input set evaluations run against: sealed beyond
	// candidate-worker reach, named so re-evaluation is reproducible.
	SealedInputs artifact.ID `json:"sealed_inputs"`
	// CleanWorker is the evidence identity of the clean re-evaluation
	// environment -- a worker that never touched the proposal.
	CleanWorker artifact.ID `json:"clean_worker"`
	// PriorApproval is the generation N-1 decider's decision approving this
	// generation's evaluator; required for every generation after 0.
	PriorApproval artifact.ID `json:"prior_approval,omitzero"`
	ID            artifact.ID `json:"-"`
}

var admissionBindingCodec = artifact.JSONDocumentCodec(
	"admission binding", artifact.KindEvidence, AdmissionBindingMediaType, AdmissionBindingSchema,
	canonicalizeAdmissionBinding,
	func(value AdmissionBinding) artifact.ID { return value.ID },
	func(value *AdmissionBinding, id artifact.ID) { value.ID = id }, nil,
)

// NewAdmissionBinding identifies one generation's authority binding.
func NewAdmissionBinding(value AdmissionBinding) (AdmissionBinding, error) {
	return admissionBindingCodec.NewInitial(value)
}

func ParseAdmissionBinding(content []byte) (AdmissionBinding, error) {
	return admissionBindingCodec.Parse(content)
}

func (b AdmissionBinding) Content() (artifact.Content, error) {
	return admissionBindingCodec.Content(b)
}

func canonicalizeAdmissionBinding(value *AdmissionBinding) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion {
		return errors.New("run record: invalid admission binding version")
	}
	domains := []AuthorityDomain{value.Proposer, value.Evaluator, value.Decider}
	for _, domain := range domains {
		if strings.TrimSpace(domain.Name) == "" || domain.Identity.Kind() != artifact.KindEvidence {
			return errors.New("run record: authority domain requires a name and an evidence identity")
		}
	}
	for i := 0; i < len(domains); i++ {
		for j := i + 1; j < len(domains); j++ {
			if domains[i].Name == domains[j].Name {
				return fmt.Errorf("run record: authority domains %q are not independent", domains[i].Name)
			}
			if domains[i].Identity == domains[j].Identity {
				return errors.New("run record: authority domains share one identity; distinct artifact IDs are not distinct domains")
			}
		}
	}
	if !value.SealedInputs.Valid() || !value.CleanWorker.Valid() {
		return errors.New("run record: admission requires sealed inputs and a clean re-evaluation worker")
	}
	if value.Generation > 0 && !value.PriorApproval.Valid() {
		return errors.New("run record: generation succession requires the prior generation's approval")
	}
	if value.Generation == 0 && value.PriorApproval.Valid() {
		return errors.New("run record: the bootstrap generation has no prior authority to cite")
	}
	return nil
}

// ValidateAdmissionSuccession proves generation N's binding was approved by
// generation N-1's decider authority: the approval decision must be decided
// under the prior decider's identity, must promote the NEW evaluator identity
// as its subject, and must be the exact approval the new binding cites.
func ValidateAdmissionSuccession(current, prior AdmissionBinding, approval recipe.Decision) error {
	if err := current.ValidateIdentity(); err != nil {
		return err
	}
	if err := prior.ValidateIdentity(); err != nil {
		return err
	}
	if current.Generation != prior.Generation+1 {
		return fmt.Errorf("run record: generation %d does not succeed %d", current.Generation, prior.Generation)
	}
	if current.PriorApproval != approval.ID {
		return errors.New("run record: binding cites a different approval")
	}
	if approval.Decider.Derivation != prior.Decider.Identity {
		return errors.New("run record: approval was not decided by the prior generation's decider authority")
	}
	if approval.Subject != current.Evaluator.Identity {
		return errors.New("run record: approval subject is not the new evaluator identity")
	}
	if approval.Outcome != recipe.DecisionAccepted {
		return fmt.Errorf("run record: approval outcome %q does not admit the new generation", approval.Outcome)
	}
	return nil
}

func (b AdmissionBinding) ValidateIdentity() error {
	return admissionBindingCodec.ValidateIdentity(b)
}
