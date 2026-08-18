package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

func shardAlias(plan, shard artifact.ID) string {
	return "evaluation/shards/" + plan.String() + "/" + shard.String()
}

func campaignAlias(plan artifact.ID) string {
	return "evaluation/campaigns/" + plan.String()
}

func contentFor[T any](contract artifact.DocumentContract, id artifact.ID, value T) (artifact.Content, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return artifact.Content{}, err
	}
	return contract.Content(id, data)
}

func loadShardReport(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	shard caseShard,
) (shardReport, bool, error) {
	target, ok, err := artifact.ResolveAlias(ctx, repository, shardAlias(plan.identity, shard.ID))
	if err != nil || !ok {
		return shardReport{}, ok, err
	}
	content, ok, err := repository.Content(ctx, target)
	if err != nil {
		return shardReport{}, false, err
	}
	if !ok {
		return shardReport{}, false, errors.New("evaluation: completed shard report content is absent")
	}
	if err := shardReportContract.ValidateContent(content, target); err != nil {
		return shardReport{}, false, fmt.Errorf("evaluation: completed shard report: %w", err)
	}
	var report shardReport
	if err := strictjson.DecodeBytes(content.Data, &report); err != nil {
		return shardReport{}, false, fmt.Errorf("evaluation: decode shard report: %w", err)
	}
	report.ID = target
	canonical, err := newShardReport(shard, report.Observations, report.Metrics)
	if err != nil || !reflect.DeepEqual(canonical, report) {
		return shardReport{}, false, errors.New("evaluation: completed shard report differs from plan")
	}
	for _, observation := range report.Observations {
		outputContent, found, err := repository.Content(ctx, observation.Output)
		if err != nil {
			return shardReport{}, false, err
		}
		if !found {
			return shardReport{}, false, errors.New("evaluation: completed shard output is absent")
		}
		if err := textOutputContract.ValidateContent(outputContent, observation.Output); err != nil {
			return shardReport{}, false, fmt.Errorf("evaluation: completed shard output: %w", err)
		}
		var output textOutput
		if err := strictjson.DecodeBytes(outputContent.Data, &output); err != nil {
			return shardReport{}, false, fmt.Errorf("evaluation: decode shard output: %w", err)
		}
		output.ID = observation.Output
		canonicalOutput, err := newTextOutput(plan.identity, observation.Case, output.Text)
		if err != nil || canonicalOutput != output {
			return shardReport{}, false, errors.New("evaluation: completed shard output differs from plan")
		}
	}
	return report, true, nil
}

func publishShardReport(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	shard caseShard,
	output textOutput,
	report shardReport,
) error {
	planContent, err := plan.content()
	if err != nil {
		return err
	}
	shardContent, err := contentFor(caseShardContract, shard.ID, shard)
	if err != nil {
		return err
	}
	outputContent, err := contentFor(textOutputContract, output.ID, output)
	if err != nil {
		return err
	}
	reportContent, err := contentFor(shardReportContract, report.ID, report)
	if err != nil {
		return err
	}
	lineage := artifact.DependencyLineage(shard.ID, plan.identity)
	lineage = append(lineage, artifact.DependencyLineage(output.ID, shard.ID)...)
	lineage = append(lineage, artifact.DependencyLineage(report.ID, plan.identity, shard.ID, output.ID)...)
	alias := shardAlias(plan.identity, shard.ID)
	batch, err := artifact.NewDocumentBatch(
		alias,
		[]artifact.Content{planContent, shardContent, outputContent, reportContent},
		lineage,
		[]artifact.AliasBinding{{Name: alias, Target: report.ID}},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return err
}

func publishCampaignReport(
	ctx context.Context,
	repository artifact.Repository,
	plan Plan,
	report campaignReport,
) error {
	alias := campaignAlias(plan.identity)
	current, exists, err := artifact.ResolveAlias(ctx, repository, alias)
	if err != nil {
		return err
	}
	if exists {
		if current != report.ID {
			return errors.New("evaluation: campaign alias differs from compiled report")
		}
		content, found, err := repository.Content(ctx, current)
		if err != nil {
			return err
		}
		if !found {
			return errors.New("evaluation: campaign report content is absent")
		}
		return campaignReportContract.ValidateContent(content, current)
	}
	planContent, err := plan.content()
	if err != nil {
		return err
	}
	reportContent, err := contentFor(campaignReportContract, report.ID, report)
	if err != nil {
		return err
	}
	parents := []artifact.ID{plan.identity}
	for _, shard := range report.Shards {
		shardReportID, found, err := artifact.ResolveAlias(ctx, repository, shardAlias(plan.identity, shard))
		if err != nil {
			return err
		}
		if !found {
			return errors.New("evaluation: campaign shard report is absent")
		}
		parents = append(parents, shardReportID)
	}
	batch, err := artifact.NewDocumentBatch(
		alias,
		[]artifact.Content{planContent, reportContent},
		artifact.DependencyLineage(report.ID, parents...),
		[]artifact.AliasBinding{{Name: alias, Target: report.ID}},
	)
	if err != nil {
		return err
	}
	_, err = artifact.CommitBatch(ctx, repository, batch)
	return err
}
