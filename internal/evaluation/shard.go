package evaluation

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"slices"
	"sort"

	"overgo/internal/artifact"
	"overgo/internal/textcheck"
)

const (
	exactMetricName         = "exact-accuracy"
	caseShardMediaType      = "application/vnd.overgo.evaluation-shard+json"
	caseShardSchema         = "overgo/evaluation-shard/v1"
	textOutputMediaType     = "application/vnd.overgo.evaluation-output+json"
	textOutputSchema        = "overgo/evaluation-output/v1"
	shardReportMediaType    = "application/vnd.overgo.evaluation-shard-report+json"
	shardReportSchema       = "overgo/evaluation-shard-report/v1"
	campaignReportMediaType = "application/vnd.overgo.evaluation-campaign+json"
	campaignReportSchema    = "overgo/evaluation-campaign/v1"
)

var (
	caseShardContract = artifact.DocumentContract{
		Kind: artifact.KindDatasetShard, MediaType: caseShardMediaType, Schema: caseShardSchema,
	}
	textOutputContract = artifact.DocumentContract{
		Kind: artifact.KindOutput, MediaType: textOutputMediaType, Schema: textOutputSchema,
	}
	shardReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: shardReportMediaType, Schema: shardReportSchema,
	}
	campaignReportContract = artifact.DocumentContract{
		Kind: artifact.KindEvaluation, MediaType: campaignReportMediaType, Schema: campaignReportSchema,
	}
)

type caseShard struct {
	ID    artifact.ID   `json:"-"`
	Plan  artifact.ID   `json:"plan"`
	Index uint64        `json:"index"`
	Cases []artifact.ID `json:"cases"`
}

type textOutput struct {
	ID   artifact.ID `json:"-"`
	Plan artifact.ID `json:"plan"`
	Case artifact.ID `json:"case"`
	Text string      `json:"text"`
}

type caseObservation struct {
	Case   artifact.ID `json:"case"`
	Output artifact.ID `json:"output"`
	Passed bool        `json:"passed"`
}

type metricState struct {
	Name  string  `json:"name"`
	Sum   float64 `json:"sum"`
	Count uint64  `json:"count"`
}

type shardReport struct {
	ID           artifact.ID       `json:"-"`
	Version      uint16            `json:"version"`
	Plan         artifact.ID       `json:"plan"`
	Shard        artifact.ID       `json:"shard"`
	Index        uint64            `json:"index"`
	Observations []caseObservation `json:"observations"`
	Metrics      []metricState     `json:"metrics"`
}

type campaignReport struct {
	ID      artifact.ID   `json:"-"`
	Version uint16        `json:"version"`
	Plan    artifact.ID   `json:"plan"`
	Shards  []artifact.ID `json:"shards"`
	Metrics []metricState `json:"metrics"`
	Cases   uint64        `json:"cases"`
}

func EvaluateExactSharded(
	ctx context.Context,
	repository artifact.Repository,
	generator Generator,
	exact ExactPlan,
	plan Plan,
	observe func(ExactResult) error,
) (artifact.ID, error) {
	if ctx == nil || repository == nil || generator == nil || exact.identity != plan.body.CaseProfile || len(exact.suite.Cases) == 0 {
		return artifact.ID{}, errors.New("evaluation: exact plan authority differs")
	}
	shards, err := compileExactShards(exact, plan)
	if err != nil {
		return artifact.ID{}, err
	}
	if err := publishAuthorities(ctx, repository, exact, plan); err != nil {
		return artifact.ID{}, err
	}
	completed, err := loadShardReports(ctx, repository, plan, shards)
	if err != nil {
		return artifact.ID{}, err
	}
	reports := make([]shardReport, 0, len(shards))
	for index, shard := range shards {
		if stored, ok := completed[index]; ok {
			reports = append(reports, stored)
			continue
		}
		result, err := evaluateExactCase(ctx, generator, exact.suite.Cases[index])
		if err != nil {
			return artifact.ID{}, err
		}
		output, err := newTextOutput(plan.identity, shard.Cases[0], result.Text)
		if err != nil {
			return artifact.ID{}, err
		}
		metric := metricState{Name: exactMetricName}
		if err := metric.observe(1); err != nil {
			return artifact.ID{}, err
		}
		report, err := newShardReport(
			shard,
			[]caseObservation{{Case: shard.Cases[0], Output: output.ID, Passed: true}},
			[]metricState{metric},
		)
		if err != nil {
			return artifact.ID{}, err
		}
		if err := publishShardReport(ctx, repository, plan, shard, output, report); err != nil {
			return artifact.ID{}, err
		}
		reports = append(reports, report)
		if observe != nil {
			if err := observe(result); err != nil {
				return artifact.ID{}, fmt.Errorf("evaluation: exact case %q observation: %w", result.Name, err)
			}
		}
	}
	campaign, err := mergeShardReports(reports)
	if err != nil {
		return artifact.ID{}, err
	}
	if err := publishCampaignReport(ctx, repository, plan, campaign); err != nil {
		return artifact.ID{}, err
	}
	return campaign.ID, nil
}

func compileExactShards(exact ExactPlan, plan Plan) ([]caseShard, error) {
	cases := make([]artifact.ID, len(exact.suite.Cases))
	for index := range cases {
		id, err := artifact.JSONID(artifact.KindDatasetShard, struct {
			Split artifact.ID `json:"split"`
			Index int         `json:"index"`
		}{Split: exact.split, Index: index})
		if err != nil {
			return nil, err
		}
		cases[index] = id
	}
	return compileCaseShards(plan, cases)
}

func compileCaseShards(plan Plan, cases []artifact.ID) ([]caseShard, error) {
	if !plan.identity.Valid() || len(cases) == 0 {
		return nil, errors.New("evaluation: invalid shard inputs")
	}
	for _, id := range cases {
		if id.Kind() != artifact.KindDatasetShard {
			return nil, errors.New("evaluation: invalid case identity")
		}
	}
	shards := make([]caseShard, len(cases))
	for index, caseID := range cases {
		shard := caseShard{Plan: plan.identity, Index: uint64(index), Cases: []artifact.ID{caseID}}
		data, err := json.Marshal(shard)
		if err != nil {
			return nil, err
		}
		id, err := caseShardContract.Identify(data)
		if err != nil {
			return nil, err
		}
		shard.ID = id
		shards[index] = shard
	}
	return shards, nil
}

func newTextOutput(plan, caseID artifact.ID, text string) (textOutput, error) {
	if plan.Kind() != artifact.KindProfile || caseID.Kind() != artifact.KindDatasetShard {
		return textOutput{}, errors.New("evaluation: invalid text output authority")
	}
	output := textOutput{Plan: plan, Case: caseID, Text: text}
	data, err := json.Marshal(output)
	if err != nil {
		return textOutput{}, err
	}
	id, err := textOutputContract.Identify(data)
	if err != nil {
		return textOutput{}, err
	}
	output.ID = id
	return output, nil
}

func (m *metricState) observe(value float64) error {
	if m == nil || !textcheck.LowerIdentifier(m.Name, len(m.Name)) || math.IsNaN(value) || math.IsInf(value, 0) ||
		m.Count == math.MaxUint64 || math.IsNaN(m.Sum+value) || math.IsInf(m.Sum+value, 0) {
		return errors.New("evaluation: invalid metric observation")
	}
	m.Sum += value
	m.Count++
	return nil
}

func newShardReport(shard caseShard, observations []caseObservation, metrics []metricState) (shardReport, error) {
	if !shard.ID.Valid() || shard.Plan.Kind() != artifact.KindProfile || len(shard.Cases) == 0 ||
		len(observations) != len(shard.Cases) {
		return shardReport{}, errors.New("evaluation: incomplete shard report")
	}
	observations = slices.Clone(observations)
	for index, observation := range observations {
		if observation.Case != shard.Cases[index] || observation.Output.Kind() != artifact.KindOutput {
			return shardReport{}, errors.New("evaluation: shard observation differs from case program")
		}
	}
	metrics = slices.Clone(metrics)
	sort.Slice(metrics, func(i, j int) bool { return metrics[i].Name < metrics[j].Name })
	for index, metric := range metrics {
		if !textcheck.LowerIdentifier(metric.Name, len(metric.Name)) || metric.Count == 0 ||
			math.IsNaN(metric.Sum) || math.IsInf(metric.Sum, 0) ||
			index > 0 && metrics[index-1].Name == metric.Name {
			return shardReport{}, errors.New("evaluation: invalid shard metric")
		}
	}
	report := shardReport{
		Version: artifact.InitialDocumentVersion, Plan: shard.Plan, Shard: shard.ID, Index: shard.Index,
		Observations: observations, Metrics: metrics,
	}
	data, err := json.Marshal(report)
	if err != nil {
		return shardReport{}, err
	}
	id, err := shardReportContract.Identify(data)
	if err != nil {
		return shardReport{}, err
	}
	report.ID = id
	return report, nil
}

func mergeShardReports(reports []shardReport) (campaignReport, error) {
	if len(reports) == 0 {
		return campaignReport{}, errors.New("evaluation: shard reports are absent")
	}
	reports = slices.Clone(reports)
	sort.Slice(reports, func(i, j int) bool { return reports[i].Index < reports[j].Index })
	plan := reports[0].Plan
	metricByName := make(map[string]metricState)
	shards := make([]artifact.ID, len(reports))
	var cases uint64
	for index, report := range reports {
		if report.Version != artifact.InitialDocumentVersion || report.Plan != plan || report.ID.Kind() != artifact.KindEvaluation ||
			report.Shard.Kind() != artifact.KindDatasetShard || report.Index != uint64(index) ||
			len(report.Observations) == 0 || cases > math.MaxUint64-uint64(len(report.Observations)) {
			return campaignReport{}, errors.New("evaluation: incompatible shard report")
		}
		if index > 0 && report.Shard == reports[index-1].Shard {
			return campaignReport{}, errors.New("evaluation: duplicate shard report")
		}
		shards[index] = report.Shard
		cases += uint64(len(report.Observations))
		for _, metric := range report.Metrics {
			state := metricByName[metric.Name]
			state.Name = metric.Name
			if state.Count > math.MaxUint64-metric.Count || math.IsNaN(state.Sum+metric.Sum) || math.IsInf(state.Sum+metric.Sum, 0) {
				return campaignReport{}, errors.New("evaluation: metric aggregate overflow")
			}
			state.Sum += metric.Sum
			state.Count += metric.Count
			metricByName[metric.Name] = state
		}
	}
	metrics := make([]metricState, 0, len(metricByName))
	for _, metric := range metricByName {
		metrics = append(metrics, metric)
	}
	sort.Slice(metrics, func(i, j int) bool { return metrics[i].Name < metrics[j].Name })
	report := campaignReport{
		Version: artifact.InitialDocumentVersion, Plan: plan, Shards: shards, Metrics: metrics, Cases: cases,
	}
	data, err := json.Marshal(report)
	if err != nil {
		return campaignReport{}, err
	}
	id, err := campaignReportContract.Identify(data)
	if err != nil {
		return campaignReport{}, err
	}
	report.ID = id
	return report, nil
}
