package trainingworkflow

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/densecausal"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/tensor"
	"overgo/internal/tensorstats"
	"overgo/internal/trainingprogram"
)

const (
	changeJudgmentMediaType = "application/vnd.overgo.change-judgment+json"
	changeJudgmentSchema    = "overgo/change-judgment/v1"

	// The candidates a judgment scores: the one the code of the changes
	// proposes, and the plain controls it has to beat.
	changeProposed     = "proposed"
	changeBestRun      = "best-run"
	changeMean         = "mean"
	changeMeanImproved = "mean-of-improved"
	changeRandom       = "random-direction"

	// changeResamplingDraws is how many times the held-out records are
	// redrawn to bound the difference between two candidates' scores: enough
	// that the bounds below move in their third digit between seeds.
	changeResamplingDraws = 10000
	// changeLowerQuantile and changeUpperQuantile bound the middle of the
	// redrawn differences that a kept proposal must hold above zero: the
	// usual nineteen parts in twenty.
	changeLowerQuantile = 0.025
	changeUpperQuantile = 0.975
	// changeRecordAssumption states what the resampling of held-out records
	// rests on.
	changeRecordAssumption = "overgo/change-judgment/assumption/held-out records are separate documents, each read whole, and a record is the unit of independence"
	// changeRecordStratum names the one stratum the held-out records share.
	changeRecordStratum = "overgo/change-judgment/stratum/held-out records"
	// changeRecordUnit prefixes the identity a held-out record is resampled under.
	changeRecordUnit = "overgo/change-judgment/record/"
)

// ChangeCandidate is one candidate change of a corpus's base model: the base
// plus a weighted sum of the corpus's changes, the weights in the corpus's own
// order, with the report that scored it on the corpus's held-out inputs and
// the metric it read.
type ChangeCandidate struct {
	Name    string      `json:"name"`
	Weights []float64   `json:"weights"`
	Report  artifact.ID `json:"report"`
	Score   float64     `json:"score"`
}

// ChangeJudgment is a change proposed from a corpus, judged against plain
// controls. The proposer is a linear code of the changes: the directions the
// runs' changes differ along most, as many of them as best predict the gain
// of a run left out, and the proposed change is the mean change moved along
// the direction that code says gain rises in, as far as a typical run sits
// from the mean. The controls are the best single run, the mean of all runs,
// the mean of the runs that improved, and the mean moved the same distance in
// a random direction. Every candidate is scored on the corpus's held-out
// inputs, and the proposal is kept only when its score exceeds the best
// control's across the middle of the redrawn held-out records.
type ChangeJudgment struct {
	ID         artifact.ID `json:"-"`
	Version    uint16      `json:"version"`
	Corpus     artifact.ID `json:"corpus"`
	Resampling artifact.ID `json:"resampling"`
	// Rank is how many directions the code keeps, and Explained the share of
	// the runs' gain differences it predicts for a run left out: at or below
	// zero, the changes say nothing of the gain that the mean gain does not.
	Rank      int     `json:"rank"`
	Explained float64 `json:"explained"`
	// GainSpread is the standard deviation of the runs' gains, and GainNoise
	// the standard error of one run's gain over the held-out records: a
	// spread no larger than the noise leaves no gain difference to explain.
	GainSpread float64 `json:"gain_spread"`
	GainNoise  float64 `json:"gain_noise"`
	// Base is the base model's score, read again beside the candidates.
	Base       float64           `json:"base"`
	Candidates []ChangeCandidate `json:"candidates"`
	// Control names the control with the best score. Difference is the
	// proposal's score less that control's, and Lower and Upper bound the
	// middle of that difference over the redrawn records.
	Control    string  `json:"control"`
	Difference float64 `json:"difference"`
	Lower      float64 `json:"lower"`
	Upper      float64 `json:"upper"`
	Kept       bool    `json:"kept"`
}

var changeJudgmentCodec = artifact.JSONDocumentCodec(
	"change judgment", artifact.KindEvaluation, changeJudgmentMediaType, changeJudgmentSchema,
	canonicalizeChangeJudgment,
	func(value ChangeJudgment) artifact.ID { return value.ID },
	func(value *ChangeJudgment, id artifact.ID) { value.ID = id },
	func(value ChangeJudgment) ChangeJudgment {
		value.Candidates = slices.Clone(value.Candidates)
		for index := range value.Candidates {
			value.Candidates[index].Weights = slices.Clone(value.Candidates[index].Weights)
		}
		return value
	},
)

// canonicalizeChangeJudgment refuses a judgment without its proposal, its
// control or their reports, and recomputes whether the proposal is kept from
// the recorded bounds.
func canonicalizeChangeJudgment(value *ChangeJudgment) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Corpus.Kind() != artifact.KindEvidence ||
		value.Resampling.Kind() != artifact.KindProfile || value.Rank <= 0 {
		return errors.New("training workflow: incomplete change judgment")
	}
	names := map[string]bool{}
	for _, candidate := range value.Candidates {
		if candidate.Name == "" || names[candidate.Name] || candidate.Report.Kind() != artifact.KindEvaluation || len(candidate.Weights) == 0 {
			return errors.New("training workflow: a change candidate lacks a name of its own, its weights or its report")
		}
		names[candidate.Name] = true
	}
	if !names[changeProposed] || !names[value.Control] || value.Control == changeProposed {
		return errors.New("training workflow: a change judgment needs its proposal and the control it is judged against")
	}
	value.Kept = value.Lower > 0
	return nil
}

// changeCode is a linear code of a corpus's changes fitted to their gains:
// how many directions it keeps, the share of the gain differences it predicts
// for a run left out, the kept directions and the direction it says gain
// rises in, each as weights over the runs' centered changes, the gain
// direction's length, and the typical distance of a run's change from the
// mean change. Each kept direction is one unit long in weight space.
type changeCode struct {
	rank      int
	explained float64
	axes      [][]float64
	direction []float64
	length    float64
	spread    float64
}

// linearChangeCode fits the code. The changes' inner products, centered on
// their mean, give the directions the runs differ along, in order of how much
// they differ along each. For each count of leading directions, the gains are
// regressed on the runs' coordinates along them and the fit is judged by how
// well it predicts each run with that run left out; the count that predicts
// best is kept. The direction returned is the gradient of that fit: a step
// along it is the step of its length that the fit says raises gain most.
func linearChangeCode(gram, gains []float64) (changeCode, error) {
	count := len(gains)
	if count*count != len(gram) {
		return changeCode{}, errors.New("training workflow: a change code needs one gain a change")
	}
	rowMeans := make([]float64, count)
	var mean, meanGain float64
	for row := range count {
		for column := range count {
			rowMeans[row] += gram[row*count+column] / float64(count)
		}
		mean += rowMeans[row] / float64(count)
		meanGain += gains[row] / float64(count)
	}
	centered := make([]float64, len(gram))
	var trace float64
	for row := range count {
		for column := range count {
			centered[row*count+column] = gram[row*count+column] - rowMeans[row] - rowMeans[column] + mean
		}
		trace += centered[row*count+row]
	}
	values, vectors, ok := tensorstats.SymmetricEigen(centered, count)
	if !ok || !(trace > 0) {
		return changeCode{}, errors.New("training workflow: the changes do not differ, so there is nothing to code")
	}
	// The mean is fitted too, and a fit that passes through every run predicts none left out.
	most := 0
	for most < count && values[most] > values[0]*float64(count)*epsilon64() {
		most++
	}
	most = min(most, count-tensor.PairedExtent)
	if most <= 0 {
		return changeCode{}, fmt.Errorf("training workflow: %d changes are too few to code and judge the code", count)
	}
	var unexplained float64
	for _, gain := range gains {
		left := (gain - meanGain) / (1 - 1/float64(count))
		unexplained += left * left
	}
	code := changeCode{spread: math.Sqrt(trace / float64(count)), explained: math.Inf(-1)}
	fitted, leverage := make([]float64, count), slices.Repeat([]float64{1 / float64(count)}, count)
	projections := make([]float64, most)
	for rank := range most {
		for run, gain := range gains {
			projections[rank] += vectors[run*count+rank] * (gain - meanGain)
		}
		var press float64
		for run, gain := range gains {
			fitted[run] += vectors[run*count+rank] * projections[rank]
			leverage[run] += vectors[run*count+rank] * vectors[run*count+rank]
			left := (gain - meanGain - fitted[run]) / (1 - leverage[run])
			press += left * left
		}
		var explained float64
		if unexplained > 0 {
			explained = 1 - press/unexplained
		}
		if explained > code.explained {
			code.rank, code.explained = rank+1, explained
		}
	}
	// A run's coordinate along a kept direction is its share of the eigenvector
	// times the root of the eigenvalue, so the direction, as weights over the
	// centered changes, is the eigenvector over that root, and the fit's slope
	// along it is the gains' projection over the same root.
	code.direction, code.axes = make([]float64, count), make([][]float64, code.rank)
	var squared float64
	for rank := range code.rank {
		root := math.Sqrt(values[rank])
		code.axes[rank] = make([]float64, count)
		for run := range count {
			code.axes[rank][run] = vectors[run*count+rank] / root
			code.direction[run] += projections[rank] / root * code.axes[rank][run]
		}
		squared += projections[rank] * projections[rank] / values[rank]
	}
	code.length = math.Sqrt(squared)
	return code, nil
}

// epsilon64 is the spacing of float64 at one.
func epsilon64() float64 { return math.Nextafter(1, math.Inf(1)) - 1 }

// changeLength is the length, in weight space, of a weighted sum of the
// runs' changes.
func changeLength(gram, weights []float64) float64 {
	count := len(weights)
	var squared float64
	for row := range count {
		for column := range count {
			squared += weights[row] * gram[row*count+column] * weights[column]
		}
	}
	return math.Sqrt(max(squared, 0))
}

// changeCandidates is the proposal and its controls as weights over the
// corpus's changes. gains are the runs' gains, signed so that larger is
// better. The proposal and the random control each move the mean change the
// typical distance of a run from it, the first along the code's gain
// direction and the second along a direction drawn from the seeds among the
// directions the code keeps, every angle as likely as another: the two differ
// in nothing but where they point. A control that is another one under a
// second name, as the mean of the improved runs is when every run improved,
// is left out.
func changeCandidates(gram, gains []float64, code changeCode, seedHigh, seedLow uint64) []ChangeCandidate {
	count := len(gains)
	share := 1 / float64(count)
	moved := func(direction []float64) []float64 {
		var mean float64
		for _, value := range direction {
			mean += value * share
		}
		centered := make([]float64, count)
		for run, value := range direction {
			centered[run] = value - mean
		}
		weights := slices.Repeat([]float64{share}, count)
		if length := changeLength(gram, centered); length > 0 {
			for run, value := range centered {
				weights[run] += code.spread / length * value
			}
		}
		return weights
	}
	best, improved := make([]float64, count), make([]float64, count)
	best[slices.Index(gains, slices.Max(gains))] = 1
	improving := 0
	for _, gain := range gains {
		if gain > 0 {
			improving++
		}
	}
	for run, gain := range gains {
		if gain > 0 {
			improved[run] = 1 / float64(improving)
		}
	}
	random := rand.New(rand.NewPCG(seedHigh, seedLow))
	drawn := make([]float64, count)
	for _, axis := range code.axes {
		along := random.NormFloat64()
		for run, weight := range axis {
			drawn[run] += along * weight
		}
	}
	candidates := []ChangeCandidate{
		{Name: changeProposed, Weights: moved(code.direction)},
		{Name: changeBestRun, Weights: best},
		{Name: changeMean, Weights: slices.Repeat([]float64{share}, count)},
		{Name: changeRandom, Weights: moved(drawn)},
	}
	if improving > 0 && improving < count {
		candidates = append(candidates, ChangeCandidate{Name: changeMeanImproved, Weights: improved})
	}
	return candidates
}

// ChangeProposalRequest asks for a change to be proposed from a corpus and
// judged: the corpus, its base model's directory, the directory its runs
// published under, where the candidates are written, and the sequence length
// its runs were judged at.
type ChangeProposalRequest struct {
	Corpus          artifact.ID
	ModelDirectory  string
	RunsDirectory   string
	OutputDirectory string
	MaximumSequence int
	// Progress, when set, receives a line a stage.
	Progress io.Writer
}

// ProposeChange proposes a change from a corpus and judges it against its
// controls. It reads the runs' changes from their checkpoints and fits the
// linear code to their gains; writes the proposal and each control that is
// not one run as a model directory under the output directory; scores every
// candidate, and the base again, on the corpus's held-out inputs under the
// corpus's own target plan; and publishes the judgment with the reports and
// the resampling plan it rests on. A sequence length other than the runs'
// compiles another plan and is refused.
func ProposeChange(ctx context.Context, store *overgodb.Store, request ChangeProposalRequest) (ChangeJudgment, error) {
	corpus, err := changeCorpusCodec.Require(ctx, store, request.Corpus)
	if err != nil {
		return ChangeJudgment{}, err
	}
	say := func(format string, values ...any) {
		if request.Progress != nil {
			fmt.Fprintf(request.Progress, format+"\n", values...)
		}
	}
	var sign float64 = 1
	if corpus.Direction == runrecord.DirectionMinimize {
		sign = -1
	}
	runs := make([]string, len(corpus.Changes))
	gains := make([]float64, len(corpus.Changes))
	var noise float64
	for index, change := range corpus.Changes {
		runs[index] = filepath.Join(request.RunsDirectory, fmt.Sprintf("order-%04d", change.Order))
		checkpoint, err := trainingprogram.LoadCheckpoint(runs[index])
		if err != nil || checkpoint.ID() != change.Trained {
			return ChangeJudgment{}, errors.Join(fmt.Errorf("training workflow: %s does not hold the checkpoint of the corpus's change of order %d", runs[index], change.Order), err)
		}
		gains[index] = sign * (change.Changed - change.Base)
		facts, err := evaluation.ReadHeldoutVerdict(ctx, store, change.Verdict)
		if err != nil {
			return ChangeJudgment{}, err
		}
		noise += standardError(pairedDifferences(facts.CandidateScores, facts.BaselineScores)) / float64(len(corpus.Changes))
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, corpus.Objective)
	if err != nil {
		return ChangeJudgment{}, err
	}
	base, err := loadTrainableModel(request.ModelDirectory)
	if err != nil {
		return ChangeJudgment{}, err
	}
	tokenizer, err := loadTrainableTokenizer(request.ModelDirectory)
	if err != nil {
		return ChangeJudgment{}, err
	}
	sequence := base.Dims.ContextLength
	if request.MaximumSequence > 0 {
		sequence = min(sequence, request.MaximumSequence)
	}
	data, _, err := textObjectiveData(ctx, store, objective, tokenizer, sequence, false)
	if err != nil {
		return ChangeJudgment{}, err
	}
	training, heldout, err := objectiveMemberships(ctx, store, objective)
	if err != nil {
		return ChangeJudgment{}, err
	}
	view, err := evaluation.CompileSFTEvaluationView(objective, training, heldout)
	if err != nil {
		return ChangeJudgment{}, err
	}
	var cases []evaluation.HeldoutCase
	for _, record := range view.Records {
		cases = append(cases, evaluation.HeldoutCase{Record: record.ID, Values: data.Examples[record.ID].Target})
	}
	plan, err := evaluation.CompileHeldoutPlan(objective, view, cases)
	if err != nil {
		return ChangeJudgment{}, err
	}
	if plan.ID() != corpus.Plan || data.Heldout != corpus.Inputs {
		return ChangeJudgment{}, errors.New("training workflow: the corpus's runs were judged on other held-out inputs than this sequence length reads; judge the proposal at the runs' sequence length")
	}

	tensors, err := openChangeTensors(request.ModelDirectory, runs)
	if err != nil {
		return ChangeJudgment{}, err
	}
	defer tensors.Close()
	say("reading the changes of %d runs", len(runs))
	gram, err := changeGram(ctx, tensors)
	if err != nil {
		return ChangeJudgment{}, err
	}
	code, err := linearChangeCode(gram, gains)
	if err != nil {
		return ChangeJudgment{}, err
	}
	seedHigh, seedLow := corpusSeeds(corpus.ID)
	candidates := changeCandidates(gram, gains, code, seedHigh, seedLow)
	say("code of rank %d explains %.3f of the gain of a run left out; runs sit %.4g from the mean change", code.rank, code.explained, code.spread)

	// A candidate that is one run is that run's checkpoint; the others are written.
	directories := make([]string, len(candidates))
	var written []string
	var weights [][]float64
	for index, candidate := range candidates {
		if candidate.Name == changeBestRun {
			directories[index] = runs[slices.Index(candidate.Weights, 1)]
			continue
		}
		directories[index] = filepath.Join(request.OutputDirectory, candidate.Name)
		written, weights = append(written, directories[index]), append(weights, candidate.Weights)
	}
	if err := materializeChanges(ctx, tensors, request.ModelDirectory, written, weights); err != nil {
		return ChangeJudgment{}, err
	}

	batch := artifact.Batch{Key: "training/change-judgment/" + corpus.ID.String()}
	score := func(model *densecausal.Model) (evaluation.HeldoutReport, error) {
		observations := make([]evaluation.HeldoutObservation, 0, len(cases))
		for _, target := range cases {
			prediction, err := nextTokens(model, data.Examples[target.Record], nil)
			if err != nil {
				return evaluation.HeldoutReport{}, fmt.Errorf("training workflow: held-out record %q: %w", target.Record, err)
			}
			observations = append(observations, evaluation.HeldoutObservation{Record: target.Record, Values: prediction.Values})
		}
		return plan.Score(observations)
	}
	baseReport, err := score(base)
	if err != nil {
		return ChangeJudgment{}, err
	}
	judgment := ChangeJudgment{
		Corpus: corpus.ID, Rank: code.rank, Explained: code.explained,
		GainSpread: standardDeviation(gains), GainNoise: noise, Base: reportMean(baseReport), Candidates: candidates,
	}
	if judgment.Base != corpus.Changes[0].Base {
		return ChangeJudgment{}, fmt.Errorf("training workflow: the base model reads %v here and read %v under the corpus's runs", judgment.Base, corpus.Changes[0].Base)
	}
	scores := make([][]evaluation.RecordScore, len(candidates))
	parents := []artifact.ID{corpus.ID}
	for index := range candidates {
		model, err := loadTrainableModel(directories[index])
		if err != nil {
			return ChangeJudgment{}, err
		}
		report, err := score(model)
		if err != nil {
			return ChangeJudgment{}, err
		}
		part, err := report.Batch(batch.Key)
		if err != nil {
			return ChangeJudgment{}, err
		}
		mergeBatch(&batch, part)
		scores[index] = report.RecordScores()
		judgment.Candidates[index].Report, judgment.Candidates[index].Score = report.ID(), reportMean(report)
		parents = append(parents, report.ID())
		say("%s: %s %.4f (base %.4f)", candidates[index].Name, corpus.Metric, judgment.Candidates[index].Score, judgment.Base)
	}

	// The control to beat is the one that scored best; the proposal is index zero.
	control := 0
	for index := range candidates {
		if index > 0 && (control == 0 || sign*judgment.Candidates[index].Score > sign*judgment.Candidates[control].Score) {
			control = index
		}
	}
	resampling, differences, err := resampledDifferences(ctx, corpus, scores[0], scores[control], sign, seedHigh, seedLow)
	if err != nil {
		return ChangeJudgment{}, err
	}
	planBatch, err := resampling.Batch(batch.Key)
	if err != nil {
		return ChangeJudgment{}, err
	}
	// The plan names the records it redraws and what it assumes of them; they are declared beside it.
	for _, edge := range planBatch.Lineage {
		if _, held, err := store.Artifact(ctx, edge.Parent); err != nil {
			return ChangeJudgment{}, err
		} else if !held {
			planBatch.Artifacts = append(planBatch.Artifacts, artifact.Descriptor{ID: edge.Parent})
		}
	}
	mergeBatch(&batch, planBatch)
	judgment.Resampling, judgment.Control = resampling.ID, candidates[control].Name
	judgment.Difference = sign * (judgment.Candidates[0].Score - judgment.Candidates[control].Score)
	judgment.Lower, judgment.Upper = tensorstats.Quantile(differences, changeLowerQuantile), tensorstats.Quantile(differences, changeUpperQuantile)
	if judgment, err = changeJudgmentCodec.NewInitial(judgment); err != nil {
		return ChangeJudgment{}, err
	}
	// A corpus judged again is judged the same: the judgment is already there.
	if held, err := store.HasContent(ctx, judgment.ID); err != nil || held {
		return judgment, err
	}
	batch.Key += "/" + judgment.ID.String()
	document, err := changeJudgmentCodec.Batch(batch.Key, judgment, artifact.UniqueDependencyLineage(judgment.ID, append(parents, resampling.ID)...), nil)
	if err != nil {
		return ChangeJudgment{}, err
	}
	mergeBatch(&batch, document)
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return ChangeJudgment{}, err
	}
	return judgment, nil
}

// mergeBatch adds one part of a publication to it. Two candidates that answer
// every held-out record alike share one report, so what an earlier part
// already holds is not added again.
func mergeBatch(batch *artifact.Batch, part artifact.Batch) {
	for _, content := range part.Contents {
		if !slices.ContainsFunc(batch.Contents, func(held artifact.Content) bool { return held.Descriptor.ID == content.Descriptor.ID }) {
			batch.Contents = append(batch.Contents, content)
		}
	}
	for _, declared := range part.Artifacts {
		if !slices.Contains(batch.Artifacts, declared) {
			batch.Artifacts = append(batch.Artifacts, declared)
		}
	}
	for _, edge := range part.Lineage {
		if !slices.Contains(batch.Lineage, edge) {
			batch.Lineage = append(batch.Lineage, edge)
		}
	}
}

// resampledDifferences redraws the held-out records, whole and with
// replacement, under a plan it returns, and gives the mean of the proposal's
// score less the control's over each draw, ascending and signed so that
// larger favours the proposal.
func resampledDifferences(ctx context.Context, corpus ChangeCorpus, proposed, control []evaluation.RecordScore, sign float64, seedHigh, seedLow uint64) (tensorstats.ResamplingPlan, []float64, error) {
	paired := pairedDifferences(proposed, control)
	if len(paired) == 0 || len(paired) != len(proposed) {
		return tensorstats.ResamplingPlan{}, nil, errors.New("training workflow: the proposal and its control were not scored on the same held-out records")
	}
	derived := func(kind artifact.Kind, name string) (artifact.ID, error) {
		return artifact.IdentifyBytes(kind, []byte(name))
	}
	assumption, err := derived(artifact.KindProfile, changeRecordAssumption)
	if err != nil {
		return tensorstats.ResamplingPlan{}, nil, err
	}
	stratum, err := derived(artifact.KindProfile, changeRecordStratum)
	if err != nil {
		return tensorstats.ResamplingPlan{}, nil, err
	}
	difference := make(map[artifact.ID]float64, len(paired))
	units := make([]tensorstats.SamplingUnit, len(paired))
	for index, score := range proposed {
		unit, err := derived(artifact.KindProfile, changeRecordUnit+score.Record)
		if err != nil {
			return tensorstats.ResamplingPlan{}, nil, err
		}
		units[index] = tensorstats.SamplingUnit{ID: unit, Stratum: stratum, Members: []artifact.ID{unit}}
		difference[unit] = sign * paired[index]
	}
	plan, err := tensorstats.NewResamplingPlan(tensorstats.ResamplingPlan{
		Population: corpus.Inputs, Assumption: assumption, Evidence: corpus.ID, Method: tensorstats.BlockBootstrap,
		SeedHigh: seedHigh, SeedLow: seedLow, Draws: changeResamplingDraws, Units: units,
	})
	if err != nil {
		return tensorstats.ResamplingPlan{}, nil, err
	}
	means := make([]float64, 0, changeResamplingDraws)
	err = tensorstats.ResampleBlocks(ctx, plan, func(draw tensorstats.BlockDraw) error {
		var mean float64
		for _, unit := range draw.Units {
			mean += difference[unit] / float64(len(draw.Units))
		}
		means = append(means, mean)
		return nil
	})
	slices.Sort(means)
	return plan, means, err
}

// corpusSeeds are the seeds a corpus's judgment draws by: the leading bytes
// of the corpus's own identity, so no seed is chosen and another corpus draws
// otherwise.
func corpusSeeds(corpus artifact.ID) (high, low uint64) {
	digest := []byte(corpus.DigestHex())
	return binary.LittleEndian.Uint64(digest), binary.LittleEndian.Uint64(digest[binary.Size(high):])
}

// pairedDifferences is each record's score in left less its score in right,
// in left's order; a record right does not hold ends the pairing.
func pairedDifferences(left, right []evaluation.RecordScore) []float64 {
	other := make(map[string]float64, len(right))
	for _, score := range right {
		other[score.Record] = score.Value
	}
	var differences []float64
	for _, score := range left {
		value, found := other[score.Record]
		if !found {
			return differences
		}
		differences = append(differences, score.Value-value)
	}
	return differences
}

// reportMean is the mean of a report's record scores, the metric it reads.
func reportMean(report evaluation.HeldoutReport) float64 {
	var sum float64
	scores := report.RecordScores()
	for _, score := range scores {
		sum += score.Value
	}
	// Summed, then divided, as the report's own metric is.
	return sum / float64(len(scores))
}

// standardDeviation is the sample standard deviation of values, zero for
// fewer than two.
func standardDeviation(values []float64) float64 {
	if len(values) <= tensor.SingletonExtent {
		return 0
	}
	var mean, squared float64
	for _, value := range values {
		mean += value / float64(len(values))
	}
	for _, value := range values {
		squared += (value - mean) * (value - mean)
	}
	return math.Sqrt(squared / float64(len(values)-tensor.SingletonExtent))
}

// standardError is the standard error of the mean of values.
func standardError(values []float64) float64 {
	if len(values) == 0 {
		return 0
	}
	return standardDeviation(values) / math.Sqrt(float64(len(values)))
}
