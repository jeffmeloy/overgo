package evaluation

import (
	"bytes"
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/processmeasure"
	"overgo/internal/runrecord"
)

// replayCases is how many cases a replay runs. Measured 2026-10-02
// (eval-identity.md) on the store's 20 changes to per-case outputs: 16
// length-spread cases saw all 20, one with a single hit, and 8 missed one.
const replayCases = 32

// replayLongestShare divides a replay's cases into the share taken from the
// longest prompts outright: a quarter, the owner's design (2026-10-02), so the
// cases a length-dependent change reaches first are always replayed.
const replayLongestShare = 4

var replayReceiptContract = artifact.DocumentContract{
	Kind: artifact.KindEvaluation, MediaType: "application/vnd.overgo.evaluation-replay+json", Schema: "overgo/evaluation-replay/v1",
}

// ReplayReceipt is the report of a run that carried Carried's result to
// Plan's commit: Replay, a replay of Cases (suite indices), reproduced their
// observations in Report, the full report that result stands on.
type ReplayReceipt struct {
	Version uint16      `json:"version"`
	Plan    artifact.ID `json:"plan"`
	Carried artifact.ID `json:"carried"`
	Report  artifact.ID `json:"report"`
	Replay  artifact.ID `json:"replay"`
	Cases   []int       `json:"cases"`
}

// EvaluateOrCarry evaluates suite as Evaluate does unless, in a reproducible
// environment, a replay of a length-spread sample of its cases reproduces
// byte for byte their observations in previous, an evaluation under the same
// plan but its commit: then previous's metrics are carried forward in a run
// at this commit, timed by the replay, reporting a ReplayReceipt.
func (campaign *Campaign) EvaluateOrCarry(ctx context.Context, suite CompiledSuite, previous artifact.ID) (CampaignResult, error) {
	full := func(why string) (CampaignResult, error) {
		trackProgress(ctx, suite.descriptor.Source, int(suite.descriptor.Cases)).sample("evaluating in full, not carrying " + previous.String() + ": " + why)
		return campaign.Evaluate(ctx, suite)
	}
	if campaign == nil || !previous.Valid() || !campaign.environment.Reproducible() {
		return campaign.Evaluate(ctx, suite)
	}
	record, err := runrecord.RequireEvaluation(ctx, campaign.repository, previous)
	run := runrecord.Run{}
	if err == nil {
		run, err = runrecord.RequireRun(ctx, campaign.repository, record.Run)
	}
	if err != nil || run.Outcome != runrecord.OutcomeSucceeded || len(run.Inputs) != 1 || len(run.Outputs) != 1 {
		return full(fmt.Sprintf("it is not one succeeded evaluation of one plan (%v)", err))
	}
	plan, err := loadEvidencePlan(ctx, campaign.repository, run.Inputs[0])
	if err != nil {
		return full(err.Error())
	}
	if plan.body.CodeCommit = suite.plan.body.CodeCommit; plan.body != suite.plan.body {
		return full("its plan differs from this one beyond the commit")
	}
	report, kept, err := keptObservations(ctx, campaign.repository, run.Outputs[0])
	var source map[string]json.RawMessage
	var cases []json.RawMessage
	if err = errors.Join(err, json.Unmarshal(suite.source, &source), json.Unmarshal(source["cases"], &cases)); err != nil || len(kept) != len(cases) || len(cases) <= replayCases {
		return full(fmt.Sprintf("%s keeps %d observations of its %d cases, which a %d-case replay cannot stand for (%v)", report, len(kept), len(cases), replayCases, err))
	}
	indices := replaySelection(cases, replayCases)
	selected := make([]json.RawMessage, len(indices))
	for at, index := range indices {
		selected[at] = cases[index]
	}
	source["cases"], err = json.Marshal(selected)
	restricted, marshalErr := json.Marshal(source)
	var sample CompiledSuite
	if err = errors.Join(err, marshalErr); err == nil {
		sample, err = CompileSuite(restricted, campaign.Authorities())
	}
	if err != nil {
		return full("its replay does not compile: " + err.Error())
	}
	if err := campaign.publishEnvironment(ctx); err != nil {
		return CampaignResult{}, err
	}
	stopwatch := processmeasure.NewStopwatch()
	replayed, err := executeSuite(ctx, campaign.repository, campaign.runtime, sample)
	measured, elapsedErr := stopwatch.Elapsed()
	var now []json.RawMessage
	if err = errors.Join(err, elapsedErr); err == nil {
		_, now, err = keptObservations(ctx, campaign.repository, replayed.Report)
	}
	if err != nil || len(now) != len(indices) {
		return full(fmt.Sprintf("its replay failed (%v)", err))
	}
	for at, index := range indices {
		if !bytes.Equal(now[at], kept[index]) {
			return full(fmt.Sprintf("case %d replayed as %.200s; %s kept %.200s", index, now[at], report, kept[index]))
		}
	}
	receipt := ReplayReceipt{Version: artifact.InitialDocumentVersion, Plan: suite.plan.identity, Carried: previous, Report: report, Replay: replayed.Report, Cases: indices}
	id, err := artifact.JSONID(artifact.KindEvaluation, receipt)
	if err == nil {
		err = publishPlanAuthorities(ctx, campaign.repository, suite.plan, nil)
	}
	if err == nil {
		err = publishCampaignDocument(ctx, campaign.repository, receipt.Plan, suite.plan.Dataset(), id, replayReceiptContract, receipt)
	}
	if err != nil {
		return CampaignResult{}, err
	}
	result, err := campaign.publishSuccess(ctx, suite, suiteResult{Report: id, Metrics: record.Metrics}, measured)
	result.Carried = previous
	return result, err
}

// replaySelection picks count (fewer than the suite's) cases as sorted suite
// indices, spread by encoded size, which orders cases as their prompts do: the
// longest quarter of count outright (owner design 2026-10-02), the rest at
// even ranks of the others from the shortest, ties in suite order.
func replaySelection(cases []json.RawMessage, count int) []int {
	order := make([]int, len(cases))
	for index := range order {
		order[index] = index
	}
	slices.SortStableFunc(order, func(left, right int) int { return cmp.Compare(len(cases[left]), len(cases[right])) })
	longest := count / replayLongestShare
	rest, chosen := order[:len(order)-longest], slices.Clone(order[len(order)-longest:])
	for rank := range count - longest {
		chosen = append(chosen, rest[rank*(len(rest)-1)/(count-longest-1)])
	}
	slices.Sort(chosen)
	return chosen
}

// keptObservations reads a report's per-case observations as encoded, in suite
// order, and the report read: for a receipt, the full report it names.
func keptObservations(ctx context.Context, reader artifact.Reader, report artifact.ID) (artifact.ID, []json.RawMessage, error) {
	content, found, err := artifact.ReadContent(ctx, reader, report)
	var kept struct {
		Report       artifact.ID       `json:"report"`
		Observations []json.RawMessage `json:"observations"`
	}
	if err == nil && found {
		err = json.Unmarshal(content.Data, &kept)
	}
	switch {
	case err != nil || !found:
		return report, nil, errors.Join(err, fmt.Errorf("report %s is unreadable", report))
	case kept.Report.Valid():
		return keptObservations(ctx, reader, kept.Report)
	}
	return report, kept.Observations, nil
}
