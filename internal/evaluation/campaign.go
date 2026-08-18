package evaluation

import (
	"context"
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/runrecord"
)

type Campaign struct {
	repository  artifact.Repository
	runtime     Runtime
	identity    modelrecipe.ProgramIdentity
	environment runrecord.Environment
	commit      string
}

type CampaignResult struct {
	Run        artifact.ID        `json:"run"`
	Evaluation artifact.ID        `json:"evaluation"`
	Report     artifact.ID        `json:"report"`
	Evidence   artifact.ID        `json:"evidence"`
	Metrics    []runrecord.Metric `json:"metrics"`
}

func NewCampaign(
	repository artifact.Repository,
	runtime Runtime,
	identity modelrecipe.ProgramIdentity,
	environment runrecord.Environment,
	commit string,
) (*Campaign, error) {
	if repository == nil || runtime == nil {
		return nil, errors.New("evaluation: campaign dependencies are absent")
	}
	if identity.Model.Kind() != artifact.KindModel || identity.Definition.Kind() != artifact.KindModelDefinition ||
		identity.Recipe.Kind() != artifact.KindRecipe ||
		environment.ID.Kind() != artifact.KindEvidence || !validCommit(commit) {
		return nil, errors.New("evaluation: invalid campaign authorities")
	}
	return &Campaign{
		repository: repository, runtime: runtime, identity: identity,
		environment: environment, commit: commit,
	}, nil
}

func (campaign *Campaign) Authorities() ExactAuthorities {
	if campaign == nil {
		return ExactAuthorities{}
	}
	return ExactAuthorities{
		ModelDefinition: campaign.identity.Definition, RuntimeRecipe: campaign.identity.Recipe,
		CodeCommit: campaign.commit, Environment: campaign.environment.ID,
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident},
	}
}

func (campaign *Campaign) Evaluate(ctx context.Context, suite CompiledSuite) (CampaignResult, error) {
	if campaign == nil || ctx == nil {
		return CampaignResult{}, errors.New("evaluation: campaign is absent")
	}
	started := time.Now()
	result, evaluateErr := ExecuteSuite(ctx, campaign.repository, campaign.runtime, suite)
	measured := uint64(max(time.Since(started).Nanoseconds(), 1))
	if evaluateErr != nil {
		terminalCtx := ctx
		outcome, failure := runrecord.OutcomeFailed, "evaluation"
		if errors.Is(evaluateErr, context.Canceled) || errors.Is(evaluateErr, context.DeadlineExceeded) || ctx.Err() != nil {
			terminalCtx = context.WithoutCancel(ctx)
			outcome, failure = runrecord.OutcomeCancelled, ""
		}
		run, publishErr := campaign.publishTerminal(terminalCtx, suite.plan, outcome, failure, measured)
		return CampaignResult{Run: run}, errors.Join(evaluateErr, publishErr)
	}
	return campaign.publishSuccess(ctx, suite, result, measured)
}

func (campaign *Campaign) publishSuccess(
	ctx context.Context,
	suite CompiledSuite,
	result SuiteResult,
	measured uint64,
) (CampaignResult, error) {
	plan := suite.plan
	run, err := runrecord.NewBoundRun(
		campaign.identity.Recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{plan.Identity()}, []artifact.ID{result.Report}, "", campaign.commit,
		campaign.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return CampaignResult{}, err
	}
	record, err := runrecord.NewEvaluation(campaign.identity.Recipe, run.ID, plan.Dataset(), result.Metrics)
	if err != nil {
		return CampaignResult{}, err
	}
	if err := campaign.publish(ctx, run, &record); err != nil {
		return CampaignResult{}, err
	}
	evidence, err := PublishEvaluationEvidence(ctx, campaign.repository, plan, suite.acceptance, result.Report, run, record)
	if err != nil {
		return CampaignResult{}, err
	}
	return CampaignResult{
		Run: run.ID, Evaluation: record.ID, Report: result.Report, Evidence: evidence.ID, Metrics: result.Metrics,
	}, nil
}

func (campaign *Campaign) publishTerminal(
	ctx context.Context,
	plan Plan,
	outcome runrecord.Outcome,
	failure string,
	measured uint64,
) (artifact.ID, error) {
	run, err := runrecord.NewBoundRun(
		campaign.identity.Recipe, outcome,
		[]artifact.ID{plan.Identity()}, nil, failure, campaign.commit,
		campaign.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return artifact.ID{}, err
	}
	return run.ID, campaign.publish(ctx, run, nil)
}

func (campaign *Campaign) publish(ctx context.Context, run runrecord.Run, record *runrecord.Evaluation) error {
	environmentContent, err := campaign.environment.Content()
	if err != nil {
		return err
	}
	runContent, err := run.Content()
	if err != nil {
		return err
	}
	contents := []artifact.Content{environmentContent, runContent}
	lineage := run.Lineage()
	if record != nil {
		recordContent, err := record.Content()
		if err != nil {
			return err
		}
		contents = append(contents, recordContent)
		lineage = append(lineage, record.Lineage()...)
	}
	batch, err := artifact.NewDocumentBatch("evaluation/run/"+run.ID.String(), contents, lineage, nil)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, campaign.repository, batch)
	return err
}
