package composition

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/bridgetrain"
	"overgo/internal/hostoptimizer"
)

// RealizationSeamActivations carries the paired bounded seam activations
// the align-init fits over — never whole checkpoints.
type RealizationSeamActivations struct {
	Source [][]float64
	Target [][]float64
}

// RealizationTraining binds the tested bridgetrain owner's inputs: the
// dataset identity and examples, the frozen forwards for both sides, the
// training policy, optimizer configuration, and epoch count. Only the
// adapter trains; the donor and target models are read-only evidence.
type RealizationTraining struct {
	Dataset        artifact.ID
	Examples       []bridgetrain.Example
	SourceForward  bridgetrain.FrozenForward
	TargetForward  bridgetrain.FrozenForward
	TrainingPolicy artifact.ID
	Config         hostoptimizer.Config
	Epochs         int
}

// RealizationAssembly names the composite's executable authorities.
type RealizationAssembly struct {
	Architecture string
	Recipe       artifact.ID
}

// RecordedSeamForward is the frozen forward for activation-replay adapter
// training: the training examples are themselves the recorded bounded
// seam activations of the frozen model, so the forward is the identity
// over them. It executes no model — the recorded activations already are
// the model's measured behavior at the seam.
type RecordedSeamForward struct {
	Model artifact.ID
}

// ModelID names the frozen model whose recorded activations replay.
func (forward RecordedSeamForward) ModelID() artifact.ID { return forward.Model }

// Forward replays the recorded activation unchanged.
func (forward RecordedSeamForward) Forward(_ context.Context, input []float32) ([]float32, error) {
	return append([]float32(nil), input...), nil
}

// CandidateRealization is one realized composite: the trained adapter's
// evidence, the assembled composed-model record referencing every frozen
// donor tensor in place, and the align residual the adapter started from.
type CandidateRealization struct {
	Candidate CompositionCandidate
	Adapter   bridgetrain.Result
	Composite ComposedModelDocument
	Residual  float64
}

// RealizeCompositionCandidate realizes one enumerated candidate: the
// interface adapter initializes from the seam alignment's fitted linear
// map, trains donors-frozen through the tested bridgetrain owner — whose
// evidence proves both model descriptors stayed byte-identical — and the
// composite assembles through the composed-model record, referencing the
// frozen donor tensors in place as parents rather than duplicating them.
// The activations must reproduce the adapter rung the candidate was
// enumerated under; drifted activations refuse. An identity-rung
// candidate assembles directly with no adapter and no training.
func RealizeCompositionCandidate(
	ctx context.Context,
	reader artifact.Reader,
	candidate CompositionCandidate,
	target artifact.ID,
	activations RealizationSeamActivations,
	training RealizationTraining,
	assembly RealizationAssembly,
) (CandidateRealization, error) {
	if target.Kind() != artifact.KindModel || candidate.Donor.Kind() != artifact.KindModel {
		return CandidateRealization{}, errors.New("composition: realization requires exact target and donor model identities")
	}
	adapter, residual, err := AlignSeamAdapter(activations.Source, activations.Target)
	if err != nil {
		return CandidateRealization{}, err
	}
	if rung := adapterRungFor(residual); rung != candidate.Adapter {
		return CandidateRealization{}, fmt.Errorf(
			"composition: seam activations admit the %q rung but the candidate was enumerated as %q; the measurement drifted",
			rung, candidate.Adapter,
		)
	}
	realization := CandidateRealization{Candidate: candidate, Residual: residual}
	document := ComposedModelDocument{
		Architecture: assembly.Architecture, Recipe: assembly.Recipe,
		Parents: []artifact.ID{target, candidate.Donor},
	}
	if candidate.Adapter != AdapterIdentity {
		rows, columns := len(adapter), len(adapter[0])
		weights := make([]float32, 0, rows*columns)
		for _, row := range adapter {
			for _, value := range row {
				weights = append(weights, float32(value))
			}
		}
		gradients := make([]float32, len(weights))
		plan, err := hostoptimizer.CompilePlan(len(weights), []hostoptimizer.GroupSpec{{
			Name: "bridge.linear", Start: 0, End: len(weights), Rows: rows, Cols: columns,
		}})
		if err != nil {
			return CandidateRealization{}, err
		}
		result, err := bridgetrain.Trainer{}.Step(ctx, bridgetrain.Request{
			Reader: reader, Source: candidate.Donor, Target: target,
			Weights: weights, Gradients: gradients, Plan: plan, Config: training.Config,
			Dataset: training.Dataset, Examples: training.Examples,
			SourceForward: training.SourceForward, TargetForward: training.TargetForward,
			TrainingPolicy: training.TrainingPolicy, Epochs: training.Epochs,
		})
		if err != nil {
			return CandidateRealization{}, err
		}
		realization.Adapter = result
		document.Adapter = result.BridgeAfter
		document.Checkpoint = result.Checkpoint
	}
	composite, err := NewComposedModel(document)
	if err != nil {
		return CandidateRealization{}, err
	}
	realization.Composite = composite
	return realization, nil
}
