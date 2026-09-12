package evaluation

import (
	"context"
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	runrecord "overgo/internal/runrecord"
)

type Campaign struct {
	repository  artifact.Repository
	documents   overgodb.DocumentReader
	runtime     Runtime
	identity    modelrecipe.ProgramIdentity
	environment runrecord.Environment
	commit      string
	lifecycle   Lifecycle
	prompting   Prompting
}

// WithPrompting binds the prompt-shaping protocol every plan this
// campaign compiles will carry; the campaign's runtime must shape
// prompts the same way, since the record describes what ran.
func (campaign *Campaign) WithPrompting(prompting Prompting) *Campaign {
	if campaign != nil {
		campaign.prompting = prompting
	}
	return campaign
}

type CampaignResult struct {
	Run        artifact.ID               `json:"run"`
	Evaluation artifact.ID               `json:"evaluation"`
	Report     artifact.ID               `json:"report"`
	Evidence   artifact.ID               `json:"evidence"`
	Metrics    []runrecord.Metric        `json:"metrics"`
	Resources  runrecord.ResourceFitness `json:"resources"`
}

func NewCampaign(
	repository artifact.Repository,
	runtime Runtime,
	identity modelrecipe.ProgramIdentity,
	environment runrecord.Environment,
	commit string,
) (*Campaign, error) {
	return newCampaign(repository, runtime, identity, environment, commit, LifecycleResident)
}

// NewIsolatedCampaign creates a campaign whose exact execution authority says
// that one containing worker process owns this candidate and no other. The
// caller remains responsible for launching that worker through processcontrol.
func NewIsolatedCampaign(
	repository artifact.Repository,
	runtime Runtime,
	identity modelrecipe.ProgramIdentity,
	environment runrecord.Environment,
	commit string,
) (*Campaign, error) {
	return newCampaign(repository, runtime, identity, environment, commit, LifecycleIsolated)
}

func newCampaign(
	repository artifact.Repository,
	runtime Runtime,
	identity modelrecipe.ProgramIdentity,
	environment runrecord.Environment,
	commit string,
	lifecycle Lifecycle,
) (*Campaign, error) {
	if repository == nil || runtime == nil {
		return nil, errors.New("evaluation: campaign dependencies are absent")
	}
	documents, ok := repository.(overgodb.DocumentReader)
	if !ok {
		return nil, errors.New("evaluation: repository lacks typed document projections")
	}
	if identity.Model.Kind() != artifact.KindModel || identity.Definition.Kind() != artifact.KindModelDefinition ||
		identity.Recipe.Kind() != artifact.KindRecipe ||
		environment.ID.Kind() != artifact.KindEvidence || !validCommit(commit) ||
		lifecycle != LifecycleResident && lifecycle != LifecycleIsolated {
		return nil, errors.New("evaluation: invalid campaign authorities")
	}
	return &Campaign{
		repository: repository, documents: documents, runtime: runtime, identity: identity,
		environment: environment, commit: commit, lifecycle: lifecycle,
	}, nil
}

func (campaign *Campaign) Authorities() ExactAuthorities {
	if campaign == nil {
		return ExactAuthorities{}
	}
	return ExactAuthorities{
		ModelDefinition: campaign.identity.Definition, RuntimeRecipe: campaign.identity.Recipe,
		CodeCommit: campaign.commit, Environment: campaign.environment.ID,
		Execution: ExecutionPolicy{Lifecycle: campaign.lifecycle, Prompting: campaign.prompting},
	}
}

func (campaign *Campaign) Evaluate(ctx context.Context, suite CompiledSuite) (CampaignResult, error) {
	if campaign == nil || ctx == nil {
		return CampaignResult{}, errors.New("evaluation: campaign is absent")
	}
	if err := campaign.publishEnvironment(ctx); err != nil {
		return CampaignResult{}, err
	}
	started := time.Now()
	result, evaluateErr := ExecuteSuite(ctx, campaign.repository, campaign.runtime, suite)
	elapsedNS := time.Since(started).Nanoseconds()
	if elapsedNS <= 0 {
		return CampaignResult{}, errors.Join(evaluateErr, errors.New("evaluation: wall measurement is not positive"))
	}
	measured := uint64(elapsedNS)
	if evaluateErr != nil {
		terminalCtx := ctx
		outcome, failure := runrecord.OutcomeFailed, "evaluation"
		if errors.Is(evaluateErr, context.Canceled) || errors.Is(evaluateErr, context.DeadlineExceeded) || ctx.Err() != nil {
			terminalCtx = context.WithoutCancel(ctx)
			outcome, failure = runrecord.OutcomeCancelled, ""
		}
		run, resources, publishErr := campaign.publishTerminal(terminalCtx, suite.plan, outcome, failure, measured)
		return CampaignResult{Run: run.ID, Resources: resources}, errors.Join(evaluateErr, publishErr)
	}
	return campaign.publishSuccess(ctx, suite, result, measured)
}

func (campaign *Campaign) publishEnvironment(ctx context.Context) error {
	if _, found, err := campaign.repository.Artifact(ctx, campaign.environment.ID); err != nil || found {
		return err
	}
	content, err := campaign.environment.Content()
	if err != nil {
		return err
	}
	batch, err := artifact.NewDocumentBatch(
		"evaluation/environment/"+campaign.environment.ID.String(), []artifact.Content{content}, nil, nil,
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, campaign.repository, batch)
	return err
}

func (campaign *Campaign) publishSuccess(
	ctx context.Context,
	suite CompiledSuite,
	result SuiteResult,
	measured uint64,
) (CampaignResult, error) {
	plan := suite.plan
	retained := CampaignResult{Report: result.Report, Metrics: result.Metrics}
	run, err := runrecord.NewBoundRun(
		campaign.identity.Recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{plan.Identity()}, []artifact.ID{result.Report}, "", campaign.commit,
		campaign.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return retained, err
	}
	resources, err := campaign.resourceFitness(plan.Identity(), run)
	if err != nil {
		return retained, err
	}
	record, err := runrecord.NewEvaluation(campaign.identity.Recipe, run.ID, plan.Dataset(), result.Metrics)
	if err != nil {
		return retained, err
	}
	batch, observation, err := campaign.preparePublication(ctx, run, &record, resources)
	if err != nil {
		return retained, err
	}
	var isolated *runrecord.ObservationChunkSummary
	if campaign.lifecycle == LifecycleIsolated {
		isolated = &observation
	}
	evidence, err := newEvaluationEvidence(
		ctx, campaign.repository, plan, suite.acceptance, suite.evaluator,
		result.Report, run, record, isolated,
	)
	if err != nil {
		return retained, err
	}
	if err := appendEvaluationEvidence(&batch, suite.acceptance, suite.evaluator, evidence); err != nil {
		return retained, err
	}
	if _, err := artifact.CommitBatch(ctx, campaign.repository, batch); err != nil {
		return retained, err
	}
	return CampaignResult{
		Run: run.ID, Evaluation: record.ID, Report: result.Report, Evidence: evidence.ID, Metrics: result.Metrics,
		Resources: resources,
	}, nil
}

func (campaign *Campaign) publishTerminal(
	ctx context.Context,
	plan Plan,
	outcome runrecord.Outcome,
	failure string,
	measured uint64,
) (runrecord.Run, runrecord.ResourceFitness, error) {
	run, err := runrecord.NewBoundRun(
		campaign.identity.Recipe, outcome,
		[]artifact.ID{plan.Identity()}, nil, failure, campaign.commit,
		campaign.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return runrecord.Run{}, runrecord.ResourceFitness{}, err
	}
	resources, err := campaign.resourceFitness(plan.Identity(), run)
	if err != nil {
		return run, runrecord.ResourceFitness{}, err
	}
	batch, _, err := campaign.preparePublication(ctx, run, nil, resources)
	if err != nil {
		return run, resources, err
	}
	if _, err := artifact.CommitBatch(ctx, campaign.repository, batch); err != nil {
		return run, resources, err
	}
	return run, resources, nil
}

func (campaign *Campaign) preparePublication(
	ctx context.Context,
	run runrecord.Run,
	record *runrecord.Evaluation,
	resources runrecord.ResourceFitness,
) (artifact.Batch, runrecord.ObservationChunkSummary, error) {
	environmentContent, err := campaign.environment.Content()
	if err != nil {
		return artifact.Batch{}, runrecord.ObservationChunkSummary{}, err
	}
	runContent, err := run.Content()
	if err != nil {
		return artifact.Batch{}, runrecord.ObservationChunkSummary{}, err
	}
	contents := []artifact.Content{environmentContent, runContent}
	lineage := run.Lineage()
	var aliases []artifact.AliasBinding
	lineage = append(lineage, artifact.Lineage{
		Child: run.ID, Parent: campaign.identity.Model, Relation: artifact.RelationDependsOn,
	})
	if record != nil {
		recordContent, err := record.Content()
		if err != nil {
			return artifact.Batch{}, runrecord.ObservationChunkSummary{}, err
		}
		contents = append(contents, recordContent)
		lineage = append(lineage, record.Lineage()...)
		lineage = append(lineage, artifact.Lineage{
			Child: record.ID, Parent: campaign.identity.Model, Relation: artifact.RelationDependsOn,
		})
		aliases = []artifact.AliasBinding{{Name: runrecord.EvaluationRunAlias(run.ID), Target: record.ID}}
	}
	batch, err := artifact.NewDocumentBatch("evaluation/run/"+run.ID.String(), contents, lineage, aliases)
	if err != nil {
		return artifact.Batch{}, runrecord.ObservationChunkSummary{}, err
	}
	chunk, err := runrecord.NewInitialObservationChunk(
		resources, runrecord.ObservationSampleExecution, run.MeasuredNS,
	)
	if err != nil {
		return artifact.Batch{}, runrecord.ObservationChunkSummary{}, err
	}
	observation, err := runrecord.BindObservationChunk(ctx, campaign.repository, &batch, chunk)
	if err != nil {
		return artifact.Batch{}, runrecord.ObservationChunkSummary{}, err
	}
	return batch, observation, nil
}
