package runrecord

import (
	"context"
	"errors"

	"overgo/internal/artifact"
	"overgo/internal/trainingprogram"
)

// ValidateMechanismCandidateEvidence closes Marin-derived evidence against its
// exact subject and independent authority. It emits no eligibility decision;
// only the cross-domain AdmitCandidate owner may do that.
func ValidateMechanismCandidateEvidence(
	ctx context.Context,
	reader artifact.Reader,
	candidate trainingprogram.MechanismCandidate,
	provenance ExternalMechanismProvenance,
	authority artifact.ID,
) error {
	if ctx == nil || reader == nil || provenance.ID != candidate.Provenance() || provenance.Census != candidate.Census() {
		return errors.New("run record: mechanism candidate provenance differs")
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
			return errors.Join(err, errors.New("run record: mechanism candidate evidence is not relevant to its subject"))
		}
		if _, duplicate := sources[evidence.Source]; duplicate {
			return errors.New("run record: mechanism candidate evidence sources are not independent")
		}
		sources[evidence.Source] = struct{}{}
	}
	binding, err := RequireAdmissionBinding(ctx, reader, authority)
	if err != nil {
		return errors.Join(err, errors.New("run record: mechanism candidate authority is not an admission binding"))
	}
	for _, evidence := range append(candidate.Evidence(), binding.Proposer.Identity, binding.Evaluator.Identity) {
		if binding.Decider.Identity == evidence || authority == evidence {
			return errors.New("run record: mechanism candidate authority is not independent")
		}
	}
	for source := range sources {
		if source == authority || source == binding.Decider.Identity {
			return errors.New("run record: mechanism candidate authority authored its own evidence")
		}
	}
	return nil
}
