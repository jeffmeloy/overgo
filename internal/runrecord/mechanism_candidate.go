package runrecord

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
)

const (
	// MechanismCandidateAdmissionMediaType identifies a mechanism admission decision.
	MechanismCandidateAdmissionMediaType = "application/vnd.overgo.mechanism-candidate-admission+json"
	// MechanismCandidateAdmissionSchema identifies the mechanism admission contract version.
	MechanismCandidateAdmissionSchema = "overgo/mechanism-candidate-admission/v1"
)

// MechanismCandidateAdmission is an independent authority's admission of a research candidate.
type MechanismCandidateAdmission struct {
	Version   uint16      `json:"version"`
	Candidate artifact.ID `json:"candidate"`
	Authority artifact.ID `json:"authority"`
	ID        artifact.ID `json:"-"`
}

var mechanismCandidateAdmissionCodec = artifact.JSONDocumentCodec(
	"mechanism candidate admission", artifact.KindEvidence,
	MechanismCandidateAdmissionMediaType, MechanismCandidateAdmissionSchema,
	canonicalizeMechanismCandidateAdmission,
	func(value MechanismCandidateAdmission) artifact.ID { return value.ID },
	func(value *MechanismCandidateAdmission, id artifact.ID) { value.ID = id }, nil,
)

// AdmitMechanismCandidate records independent authority for an evidence-gated candidate.
func AdmitMechanismCandidate(
	ctx context.Context,
	reader artifact.Reader,
	candidate trainingprogram.MechanismCandidate,
	provenance ExternalMechanismProvenance,
	authority artifact.ID,
) (MechanismCandidateAdmission, error) {
	if ctx == nil || reader == nil || provenance.ID != candidate.Provenance() || provenance.Census != candidate.Census() {
		return MechanismCandidateAdmission{}, errors.New("run record: mechanism candidate provenance differs")
	}
	references := []struct {
		id   artifact.ID
		role trainingprogram.MechanismEvidenceRole
	}{
		{candidate.Gap(), trainingprogram.MechanismEvidenceGap},
		{candidate.Falsifier(), trainingprogram.MechanismEvidenceFalsifier},
		{candidate.CostBound(), trainingprogram.MechanismEvidenceCostBound},
		{candidate.Benefit(), trainingprogram.MechanismEvidenceBenefit},
	}
	sources := make(map[artifact.ID]struct{}, len(references))
	for _, reference := range references {
		evidence, err := trainingprogram.RequireMechanismEvidence(ctx, reader, reference.id, reference.role)
		if err != nil || evidence.Mechanism != candidate.Mechanism() {
			return MechanismCandidateAdmission{}, errors.Join(err, errors.New("run record: mechanism candidate evidence is not relevant to its subject"))
		}
		if _, duplicate := sources[evidence.Source]; duplicate {
			return MechanismCandidateAdmission{}, errors.New("run record: mechanism candidate evidence sources are not independent")
		}
		sources[evidence.Source] = struct{}{}
	}
	binding, err := admissionBindingCodec.Require(ctx, reader, authority)
	if err != nil {
		return MechanismCandidateAdmission{}, errors.Join(err, errors.New("run record: mechanism candidate authority is not an admission binding"))
	}
	for _, evidence := range append(candidate.Evidence(), binding.Proposer.Identity, binding.Evaluator.Identity) {
		if binding.Decider.Identity == evidence || authority == evidence {
			return MechanismCandidateAdmission{}, errors.New("run record: mechanism candidate authority is not independent")
		}
	}
	for source := range sources {
		if source == authority || source == binding.Decider.Identity {
			return MechanismCandidateAdmission{}, errors.New("run record: mechanism candidate authority authored its own evidence")
		}
	}
	return mechanismCandidateAdmissionCodec.New(MechanismCandidateAdmission{
		Version: artifact.InitialDocumentVersion, Candidate: candidate.ID(), Authority: authority,
	})
}

// Content returns the canonical candidate admission document.
func (value MechanismCandidateAdmission) Content() (artifact.Content, error) {
	return mechanismCandidateAdmissionCodec.Content(value)
}

// Lineage binds the admission to its candidate and independent authority.
func (value MechanismCandidateAdmission) Lineage() []artifact.Lineage {
	return artifact.DependencyLineage(value.ID, value.Candidate, value.Authority)
}

func canonicalizeMechanismCandidateAdmission(value *MechanismCandidateAdmission) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Candidate.Kind() != artifact.KindRecipe ||
		value.Authority.Kind() != artifact.KindEvidence {
		return errors.New("run record: invalid mechanism candidate admission")
	}
	return nil
}
