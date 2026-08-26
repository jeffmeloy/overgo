// Package modelbuilder binds construction backends to the shared workflow.
package modelbuilder

import (
	"context"
	"errors"
	"fmt"

	"overgo/internal/artifact"
	"overgo/internal/runrecord"
	"overgo/internal/scratchmodel"
	"overgo/internal/trainingprogram"
	"overgo/internal/workflowruntime"
)

const (
	buildEvidenceMediaType = "application/vnd.overgo.model-build-evidence+json"
	buildEvidenceSchema    = "overgo/model-build-evidence/v1"
	buildDecisionMediaType = "application/vnd.overgo.model-build-decision+json"
	buildDecisionSchema    = "overgo/model-build-decision/v1"
)

var (
	buildEvidenceContract = artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: buildEvidenceMediaType, Schema: buildEvidenceSchema}
	buildDecisionContract = artifact.DocumentContract{Kind: artifact.KindEvidence, MediaType: buildDecisionMediaType, Schema: buildDecisionSchema}
)

type ScratchRequest struct {
	Repository artifact.Repository
	Documents  []string
	Seed       int64
	Steps      int
}

// ScratchWorkflowSelection derives the scratch workflow identity from
// its derivation profile.
func ScratchWorkflowSelection(profile artifact.ID) (artifact.ID, error) {
	if profile.Kind() != artifact.KindProfile {
		return artifact.ID{}, errors.New("model builder: derivation profile required")
	}
	return artifact.JSONID(artifact.KindRecipe, struct {
		Runtime string      `json:"runtime"`
		Profile artifact.ID `json:"profile"`
	}{Runtime: "scratch-model-builder/v1", Profile: profile})
}

type ScratchSession struct {
	request      ScratchRequest
	profile      scratchmodel.DerivationProfile
	optimizer    trainingprogram.OptimizerPolicy
	construction scratchmodel.Construction
	training     scratchmodel.TrainingResult
	run          runrecord.Run
	evaluation   runrecord.Evaluation
}

type buildEvidence struct {
	Construction artifact.ID `json:"construction"`
	InitialModel artifact.ID `json:"initial_model"`
	Model        artifact.ID `json:"model"`
	Checkpoint   artifact.ID `json:"checkpoint"`
	Run          artifact.ID `json:"run"`
	Evaluation   artifact.ID `json:"evaluation"`
	InitialLoss  float64     `json:"initial_loss"`
	FinalLoss    float64     `json:"final_loss"`
}

type buildDecision struct {
	State      string      `json:"state"`
	Model      artifact.ID `json:"model"`
	Evaluation artifact.ID `json:"evaluation"`
	Evidence   artifact.ID `json:"evidence"`
	Rollback   artifact.ID `json:"rollback"`
}

func NewScratchSession(ctx context.Context, request ScratchRequest) (*ScratchSession, error) {
	if request.Repository == nil || len(request.Documents) < 3 || request.Steps <= 0 {
		return nil, errors.New("model builder: repository, corpus, and positive steps required")
	}
	profile, err := scratchmodel.ResolveActiveDerivationProfile(ctx, request.Repository)
	if err != nil {
		return nil, err
	}
	optimizer, err := trainingprogram.RequireOptimizerPolicy(ctx, request.Repository, profile.Optimizer)
	if err != nil {
		return nil, err
	}
	return &ScratchSession{request: request, profile: profile, optimizer: optimizer}, nil
}

func (session *ScratchSession) Initialize(context.Context) (workflowruntime.ModelBuildState, error) {
	construction, err := scratchmodel.Compile(scratchmodel.CorpusFacts{
		Documents: session.request.Documents, Seed: session.request.Seed, Steps: session.request.Steps,
	}, session.profile)
	if err != nil {
		return workflowruntime.ModelBuildState{}, err
	}
	session.construction = construction
	return workflowruntime.ModelBuildState{
		Dataset: construction.Dataset(), DerivationProfile: session.profile.ID, Optimizer: session.optimizer.ID,
		Construction: construction.Authority().ID(), Model: construction.ID(),
	}, nil
}

func (session *ScratchSession) Train(_ context.Context, state workflowruntime.ModelBuildState) (workflowruntime.ModelBuildState, error) {
	training, err := session.construction.TrainShared(session.request.Steps, session.optimizer)
	if err != nil {
		return state, err
	}
	model, err := session.construction.IdentifyTrainedModel(training.Weights)
	if err != nil {
		return state, err
	}
	checkpoint, err := artifact.JSONID(artifact.KindCheckpoint, struct {
		Construction artifact.ID `json:"construction"`
		Model        artifact.ID `json:"model"`
		Steps        int         `json:"steps"`
	}{state.Construction, model, session.request.Steps})
	if err != nil {
		return state, err
	}
	session.training = training
	state.Model, state.Checkpoint = model, checkpoint
	return state, nil
}

func (session *ScratchSession) Evaluate(_ context.Context, state workflowruntime.ModelBuildState) (workflowruntime.ModelBuildState, error) {
	run, err := runrecord.NewRun(state.Recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{state.Dataset, state.Construction}, []artifact.ID{state.Model, state.Checkpoint}, "")
	if err != nil {
		return state, err
	}
	evaluation, err := runrecord.NewEvaluation(state.Recipe, run.ID, state.Dataset, []runrecord.Metric{
		{Name: "initial-causal-loss", Value: session.training.InitialValidationLoss, Direction: runrecord.DirectionMinimize},
		{Name: "causal-loss", Value: session.training.ValidationLoss, Direction: runrecord.DirectionMinimize},
	})
	if err != nil {
		return state, err
	}
	session.run, session.evaluation = run, evaluation
	state.Run, state.Evaluation = run.ID, evaluation.ID
	return state, nil
}

func (session *ScratchSession) Record(ctx context.Context, state workflowruntime.ModelBuildState) (workflowruntime.ModelBuildState, error) {
	evidence, err := artifact.JSONContent(buildEvidenceContract, buildEvidence{
		Construction: state.Construction, InitialModel: session.construction.ID(), Model: state.Model,
		Checkpoint: state.Checkpoint, Run: session.run.ID, Evaluation: state.Evaluation,
		InitialLoss: session.training.InitialValidationLoss, FinalLoss: session.training.ValidationLoss,
	})
	if err != nil {
		return state, err
	}
	batch, err := session.run.Batch("model-builder/run/" + session.run.ID.String())
	if err != nil {
		return state, err
	}
	evaluation, err := session.evaluation.Content()
	if err != nil {
		return state, err
	}
	batch.Contents = append(batch.Contents, evaluation, evidence)
	batch.Artifacts = append(batch.Artifacts,
		artifact.Descriptor{ID: state.Dataset},
		artifact.Descriptor{ID: state.Construction}, artifact.Descriptor{ID: session.construction.ID()},
		artifact.Descriptor{ID: state.Model}, artifact.Descriptor{ID: state.Checkpoint})
	batch.Lineage = append(batch.Lineage, session.evaluation.Lineage()...)
	batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(evidence.Descriptor.ID,
		state.Construction, session.construction.ID(), state.Model, state.Checkpoint, session.run.ID, state.Evaluation)...)
	if _, err = artifact.CommitBatch(ctx, session.request.Repository, batch); err != nil {
		return state, err
	}
	state.Evidence = evidence.Descriptor.ID
	return state, nil
}

func (session *ScratchSession) Promote(ctx context.Context, state workflowruntime.ModelBuildState) (workflowruntime.ModelBuildState, error) {
	verdict := "refuse"
	if session.training.ValidationLoss < session.training.InitialValidationLoss {
		verdict = "promote"
	}
	decision, err := artifact.JSONContent(buildDecisionContract, buildDecision{
		State: verdict, Model: state.Model, Evaluation: state.Evaluation,
		Evidence: state.Evidence, Rollback: session.construction.ID(),
	})
	if err != nil {
		return state, err
	}
	batch := artifact.Batch{
		Key:      "model-builder/decision/" + decision.Descriptor.ID.String(),
		Contents: []artifact.Content{decision},
		Lineage: artifact.DependencyLineage(decision.Descriptor.ID,
			state.Model, state.Evaluation, state.Evidence, session.construction.ID()),
	}
	if _, err = artifact.CommitBatch(ctx, session.request.Repository, batch); err != nil {
		return state, fmt.Errorf("model builder: publish decision: %w", err)
	}
	state.Decision = decision.Descriptor.ID
	return state, nil
}
