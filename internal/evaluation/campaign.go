package evaluation

import (
	"context"
	"errors"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
)

type Campaign struct {
	repository  artifact.Repository
	documents   overgodb.DocumentReader
	runtime     Runtime
	identity    modelrecipe.ProgramIdentity
	environment runrecord.Environment
	commit      string
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
	if repository == nil || runtime == nil {
		return nil, errors.New("evaluation: campaign dependencies are absent")
	}
	documents, ok := repository.(overgodb.DocumentReader)
	if !ok {
		return nil, errors.New("evaluation: repository lacks typed document projections")
	}
	if identity.Model.Kind() != artifact.KindModel || identity.Definition.Kind() != artifact.KindModelDefinition ||
		identity.Recipe.Kind() != artifact.KindRecipe ||
		environment.ID.Kind() != artifact.KindEvidence || !validCommit(commit) {
		return nil, errors.New("evaluation: invalid campaign authorities")
	}
	return &Campaign{
		repository: repository, documents: documents, runtime: runtime, identity: identity,
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
		run, publishErr := campaign.publishTerminal(terminalCtx, suite.plan, outcome, failure, measured)
		resources, resourceErr := campaign.resourceFitness(suite.plan.Identity(), run)
		return CampaignResult{Run: run.ID, Resources: resources}, errors.Join(evaluateErr, publishErr, resourceErr)
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
	run, err := runrecord.NewBoundRun(
		campaign.identity.Recipe, runrecord.OutcomeSucceeded,
		[]artifact.ID{plan.Identity()}, []artifact.ID{result.Report}, "", campaign.commit,
		campaign.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return CampaignResult{}, err
	}
	resources, err := campaign.resourceFitness(plan.Identity(), run)
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
	evidence, err := PublishEvaluationEvidence(ctx, campaign.repository, plan, suite.acceptance, suite.evaluator, result.Report, run, record)
	if err != nil {
		return CampaignResult{}, err
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
) (runrecord.Run, error) {
	run, err := runrecord.NewBoundRun(
		campaign.identity.Recipe, outcome,
		[]artifact.ID{plan.Identity()}, nil, failure, campaign.commit,
		campaign.environment.ID, measured,
		[]runrecord.PhaseMetric{{Phase: runrecord.PhaseValidate, DurationNS: measured}},
	)
	if err != nil {
		return runrecord.Run{}, err
	}
	return run, campaign.publish(ctx, run, nil)
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
	var aliases []artifact.AliasBinding
	lineage = append(lineage, artifact.Lineage{
		Child: run.ID, Parent: campaign.identity.Model, Relation: artifact.RelationDependsOn,
	})
	if record != nil {
		recordContent, err := record.Content()
		if err != nil {
			return err
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
		return err
	}
	_, err = artifact.CommitBatch(ctx, campaign.repository, batch)
	return err
}
