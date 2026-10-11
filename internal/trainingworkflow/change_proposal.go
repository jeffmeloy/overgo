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
	"overgo/internal/tensor/dtype"
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
	// changeUnconditioned is the proposal the same code makes of the gains read
	// on the pool alone: the control a proposer shown past verdicts is held to.
	changeUnconditioned = "without-evidence"

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
// order, with the reading that scored it on the selection part of its pool
// (evaluation.SelectionReading) and the mean it read there, of its documents'
// token log-likelihoods. A judgment recorded before proposals were made on a
// pool names instead the report that scored the candidate on the held-out
// records its corpus's runs were judged on, and the corpus's metric.
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
// a random direction. The gains the code is fitted to are read on the
// development part of a held-out pool; every candidate is scored on the pool's
// selection part, and the proposal is kept only when its score exceeds the
// best control's across the middle of the redrawn selection documents. Kept
// ranks the proposal among its controls. Whether a candidate is a gain over
// the base is for the pool's promotion part to say (ConfirmChange).
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
	// the standard error of one run's gain over the documents it was read on:
	// a spread no larger than the noise leaves no gain difference to explain.
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
	// Shown names the past held-out verdicts the proposer was shown of what
	// was tried before it, and Used how many of the corpus's runs they judged.
	// When they judged enough to fit, the code was fitted to those runs'
	// recorded gains beside the gains read on the pool (conditionedGains); the
	// proposal the code makes of the pool's gains alone is scored among the
	// candidates as without-evidence; and EvidenceDifference is the proposal's
	// score less that one's, with EvidenceLower and EvidenceUpper bounding the
	// middle of it over the redrawn documents. A proposer shown nothing
	// records none of these.
	Shown              []artifact.ID `json:"shown,omitempty"`
	Used               int           `json:"used,omitzero"`
	EvidenceDifference float64       `json:"evidence_difference,omitzero"`
	EvidenceLower      float64       `json:"evidence_lower,omitzero"`
	EvidenceUpper      float64       `json:"evidence_upper,omitzero"`
}

var changeJudgmentCodec = artifact.JSONDocumentCodec(
	"change judgment", artifact.KindEvaluation, changeJudgmentMediaType, changeJudgmentSchema,
	canonicalizeChangeJudgment,
	func(value ChangeJudgment) artifact.ID { return value.ID },
	func(value *ChangeJudgment, id artifact.ID) { value.ID = id },
	func(value ChangeJudgment) ChangeJudgment {
		value.Shown = slices.Clone(value.Shown)
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

// conditionedGains is what a proposer shown past verdicts fits its code to.
// For each run a shown verdict judged, the gain that verdict recorded, in its
// metric's own direction, is put beside the gain the run read on the pool's
// development part: each kind is taken about its mean in units of its spread
// over the runs it covers, and the two are added. A run no shown verdict
// judged adds nothing to its pool gain, which is what a run of the mean
// evidence would add. It returns how many runs the verdicts judged, and the
// pool's gains as they were with none when either kind does not differ from
// run to run: there is then nothing in the evidence to fit.
func conditionedGains(corpus ChangeCorpus, gains []float64, shown []artifact.ID) ([]float64, int) {
	var recorded []float64
	var judged []int
	for run, change := range corpus.Changes {
		if slices.Contains(shown, change.Verdict) {
			gain := change.Changed - change.Base
			if corpus.Direction == runrecord.DirectionMinimize {
				gain = -gain
			}
			recorded, judged = append(recorded, gain), append(judged, run)
		}
	}
	fitted, evidence := standardized(gains), standardized(recorded)
	if fitted == nil || evidence == nil {
		return gains, 0
	}
	for at, run := range judged {
		fitted[run] += evidence[at]
	}
	return fitted, len(judged)
}

// standardized is values about their mean in units of their standard
// deviation, and nil when they do not differ.
func standardized(values []float64) []float64 {
	spread := standardDeviation(values)
	if !(spread > 0) {
		return nil
	}
	var mean float64
	for _, value := range values {
		mean += value / float64(len(values))
	}
	units := make([]float64, len(values))
	for index, value := range values {
		units[index] = (value - mean) / spread
	}
	return units
}

// ChangeProposalRequest asks for a change to be proposed from a corpus and
// ranked: the corpus, the acceptance of the held-out pool it is fitted and
// ranked on, the corpus's base model's directory, the directory its runs
// published under, and where the candidates are written. The pool's documents
// are read as far as its acceptance fixed.
type ChangeProposalRequest struct {
	Corpus          artifact.ID
	Acceptance      artifact.ID
	ModelDirectory  string
	RunsDirectory   string
	OutputDirectory string
	// Shown names the held-out verdicts the proposer is shown of what was
	// tried before it (evaluation.PastHeldoutVerdicts). It fits the gains
	// those of them that judged its corpus's runs recorded beside the gains it
	// reads on the pool, and reads no other; shown none, it fits the pool's
	// gains alone.
	Shown []artifact.ID
	// Progress, when set, receives a line a stage.
	Progress io.Writer
}

// ProposeChange proposes a change from a corpus and ranks it against its
// controls on a held-out pool, none of whose documents the corpus's verdicts
// read. It reads every run on the pool's development part and fits the linear
// code to the gains it finds there; writes the proposal and each control that
// is not one run as a model directory under the output directory; scores
// every candidate, and the base, on the pool's selection part; and publishes
// the judgment with each candidate's reading and the resampling plan it rests
// on. Neither step reads the promotion part: ConfirmChange has it answer, for
// one candidate at a time. A proposal that names no pool is refused, and so is
// a pool fixed for another objective or base than the corpus's.
func ProposeChange(ctx context.Context, store *overgodb.Store, request ChangeProposalRequest) (ChangeJudgment, error) {
	corpus, err := changeCorpusCodec.Require(ctx, store, request.Corpus)
	if err != nil {
		return ChangeJudgment{}, err
	}
	pool, err := evaluation.RequireHeldoutPool(ctx, store, request.Acceptance)
	if err != nil {
		return ChangeJudgment{}, fmt.Errorf("training workflow: a change is fitted and ranked on a held-out pool, named by its acceptance: %w", err)
	}
	if pool.Acceptance.Objective != corpus.Objective || pool.Acceptance.Base != corpus.Model {
		return ChangeJudgment{}, fmt.Errorf("training workflow: pool %s was fixed for objective %s and base %s, and the corpus's runs trained base %s under objective %s",
			pool.Acceptance.ID, pool.Acceptance.Objective, pool.Acceptance.Base, corpus.Model, corpus.Objective)
	}
	for _, shown := range request.Shown {
		if descriptor, held, err := store.Artifact(ctx, shown); err != nil || !held || !evaluation.IsHeldoutVerdict(descriptor) {
			return ChangeJudgment{}, errors.Join(fmt.Errorf("training workflow: a proposer is shown held-out verdicts the store holds, and %s is not one", shown), err)
		}
	}
	base, examples, err := poolExamples(ctx, store, pool.Acceptance, request.ModelDirectory)
	if err != nil {
		return ChangeJudgment{}, err
	}
	return proposeOnPool(ctx, store, request, corpus, pool, base, examples)
}

// poolExamples loads a pool's base model and reads its objective's documents
// as that model's tokens, each as far as the pool's acceptance fixed: what
// the pool's measure is taken on. An acceptance that fixed no window is one
// published before the window was part of it, and nothing is read for it.
func poolExamples(ctx context.Context, store *overgodb.Store, acceptance evaluation.PoolAcceptance, directory string) (*densecausal.Model, map[string]HeldoutExample, error) {
	if acceptance.Sequence == 0 {
		return nil, nil, fmt.Errorf("training workflow: pool %s fixed no tokens to read each document as far as; build a pool that does", acceptance.ID)
	}
	objective, err := trainingprogram.LoadObjective(ctx, store, acceptance.Objective)
	if err != nil {
		return nil, nil, err
	}
	base, err := loadTrainableModel(directory)
	if err != nil {
		return nil, nil, err
	}
	tokenizer, err := loadTrainableTokenizer(directory)
	if err != nil {
		return nil, nil, err
	}
	data, _, err := textObjectiveData(ctx, store, objective, tokenizer, min(base.Dims.ContextLength, int(acceptance.Sequence)), false)
	return base, data.Examples, err
}

// documentLikelihoods is a pool's measure of one model, before the base is
// taken from it: for each document, the mean log-likelihood of its tokens,
// each predicted from the ones before it.
func documentLikelihoods(model *densecausal.Model, examples map[string]HeldoutExample, documents []string) ([]evaluation.RecordScore, error) {
	scores := make([]evaluation.RecordScore, len(documents))
	for index, document := range documents {
		example, found := examples[document]
		if !found {
			return nil, fmt.Errorf("training workflow: the pool's document %q has no example", document)
		}
		loss, _, err := model.Loss(dtype.ConvertSlice[int](example.Input))
		if err != nil {
			return nil, fmt.Errorf("training workflow: the pool's document %q: %w", document, err)
		}
		scores[index] = evaluation.RecordScore{Record: document, Value: -loss}
	}
	return scores, nil
}

// runDirectory is where the run of an order seed published, under the
// directory its corpus was collected in.
func runDirectory(runsDirectory string, order uint64) string {
	return filepath.Join(runsDirectory, fmt.Sprintf("order-%04d", order))
}

// candidateDirectory is where a candidate's model sits. One that is a single
// run is that run's checkpoint; the others are written under the output
// directory, each by its name.
func candidateDirectory(corpus ChangeCorpus, runsDirectory, outputDirectory string, candidate ChangeCandidate) string {
	if candidate.Name == changeBestRun {
		return runDirectory(runsDirectory, corpus.Changes[slices.Index(candidate.Weights, 1)].Order)
	}
	return filepath.Join(outputDirectory, candidate.Name)
}

// proposeOnPool is ProposeChange once the pool, its base and its documents
// are loaded. examples is all a proposal reads of the documents: the gains
// the code is fitted to come from the development part's and the candidates'
// scores from the selection part's, so nothing of another part's can move a
// candidate's weights or its rank.
func proposeOnPool(ctx context.Context, store *overgodb.Store, request ChangeProposalRequest, corpus ChangeCorpus, pool evaluation.CommittedPool, base *densecausal.Model, examples map[string]HeldoutExample) (ChangeJudgment, error) {
	say := func(format string, values ...any) {
		if request.Progress != nil {
			fmt.Fprintf(request.Progress, format+"\n", values...)
		}
	}
	baseDevelopment, err := documentLikelihoods(base, examples, pool.Development)
	if err != nil {
		return ChangeJudgment{}, err
	}
	runs := make([]string, len(corpus.Changes))
	gains := make([]float64, len(corpus.Changes))
	var noise float64
	for index, change := range corpus.Changes {
		runs[index] = runDirectory(request.RunsDirectory, change.Order)
		checkpoint, err := trainingprogram.LoadCheckpoint(runs[index])
		if err != nil || checkpoint.ID() != change.Trained {
			return ChangeJudgment{}, errors.Join(fmt.Errorf("training workflow: %s does not hold the checkpoint of the corpus's change of order %d", runs[index], change.Order), err)
		}
		model, err := loadTrainableModel(runs[index])
		if err != nil {
			return ChangeJudgment{}, err
		}
		read, err := documentLikelihoods(model, examples, pool.Development)
		if err != nil {
			return ChangeJudgment{}, err
		}
		gains[index] = scoreMean(read) - scoreMean(baseDevelopment)
		noise += standardError(pairedDifferences(read, baseDevelopment)) / float64(len(corpus.Changes))
	}

	tensors, err := openChangeTensors(request.ModelDirectory, runs)
	if err != nil {
		return ChangeJudgment{}, err
	}
	defer tensors.Close()
	say("reading the changes of %d runs, their gains read on the %d documents of the pool's development part", len(runs), len(pool.Development))
	gram, err := changeGram(ctx, tensors)
	if err != nil {
		return ChangeJudgment{}, err
	}
	fitted, used := conditionedGains(corpus, gains, request.Shown)
	code, err := linearChangeCode(gram, fitted)
	if err != nil {
		return ChangeJudgment{}, err
	}
	seedHigh, seedLow := corpusSeeds(corpus.ID)
	candidates := changeCandidates(gram, gains, code, seedHigh, seedLow)
	if used > 0 {
		// The proposal of the pool's gains alone stands last among the candidates.
		plain, err := linearChangeCode(gram, gains)
		if err != nil {
			return ChangeJudgment{}, err
		}
		candidates = append(candidates, ChangeCandidate{Name: changeUnconditioned, Weights: changeCandidates(gram, gains, plain, seedHigh, seedLow)[0].Weights})
		say("shown %d past verdicts, which judged %d of the runs; their gains are fitted beside the pool's", len(request.Shown), used)
	}
	say("code of rank %d explains %.3f of the gain of a run left out; runs sit %.4g from the mean change", code.rank, code.explained, code.spread)

	// A candidate that is one run is that run's checkpoint; the others are written.
	directories := make([]string, len(candidates))
	var written []string
	var weights [][]float64
	for index, candidate := range candidates {
		directories[index] = candidateDirectory(corpus, request.RunsDirectory, request.OutputDirectory, candidate)
		if candidate.Name != changeBestRun {
			written, weights = append(written, directories[index]), append(weights, candidate.Weights)
		}
	}
	if err := materializeChanges(ctx, tensors, request.ModelDirectory, written, weights); err != nil {
		return ChangeJudgment{}, err
	}

	baseSelection, err := documentLikelihoods(base, examples, pool.Selection)
	if err != nil {
		return ChangeJudgment{}, err
	}
	batch := artifact.Batch{Key: "training/change-judgment/" + corpus.ID.String()}
	judgment := ChangeJudgment{
		Corpus: corpus.ID, Rank: code.rank, Explained: code.explained,
		GainSpread: standardDeviation(gains), GainNoise: noise, Base: scoreMean(baseSelection), Candidates: candidates,
		Shown: request.Shown, Used: used,
	}
	scores := make([][]evaluation.RecordScore, len(candidates))
	parents := append([]artifact.ID{corpus.ID}, request.Shown...)
	for index := range candidates {
		model, err := loadTrainableModel(directories[index])
		if err != nil {
			return ChangeJudgment{}, err
		}
		// The reading names the weights it scored, which is what a confirmation later holds the directory to.
		identity, err := identifyModel(directories[index])
		if err != nil {
			return ChangeJudgment{}, err
		}
		if scores[index], err = documentLikelihoods(model, examples, pool.Selection); err != nil {
			return ChangeJudgment{}, err
		}
		reading, part, err := evaluation.NewSelectionReading(batch.Key, pool.Acceptance.ID, identity, scores[index])
		if err != nil {
			return ChangeJudgment{}, err
		}
		mergeBatch(&batch, part)
		judgment.Candidates[index].Report, judgment.Candidates[index].Score = reading.ID, scoreMean(scores[index])
		parents = append(parents, reading.ID)
		say("%s: %.4f on the %d documents of the selection part (base %.4f)", candidates[index].Name, judgment.Candidates[index].Score, len(pool.Selection), judgment.Base)
	}

	// The control to beat is the one that scored best; the proposal is index zero.
	control := 0
	for index := range candidates {
		if index > 0 && (control == 0 || judgment.Candidates[index].Score > judgment.Candidates[control].Score) {
			control = index
		}
	}
	resampling, differences, err := resampledDifferences(ctx, pool.Partition.Selection, corpus.ID, scores[0], scores[control], seedHigh, seedLow)
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
	judgment.Difference = judgment.Candidates[0].Score - judgment.Candidates[control].Score
	judgment.Lower, judgment.Upper = tensorstats.Quantile(differences, changeLowerQuantile), tensorstats.Quantile(differences, changeUpperQuantile)
	if used > 0 {
		// The same documents, redrawn the same way, bound what the evidence moved.
		last := len(candidates) - tensor.SingletonExtent
		_, apart, err := resampledDifferences(ctx, pool.Partition.Selection, corpus.ID, scores[0], scores[last], seedHigh, seedLow)
		if err != nil {
			return ChangeJudgment{}, err
		}
		judgment.EvidenceDifference = judgment.Candidates[0].Score - judgment.Candidates[last].Score
		judgment.EvidenceLower, judgment.EvidenceUpper = tensorstats.Quantile(apart, changeLowerQuantile), tensorstats.Quantile(apart, changeUpperQuantile)
	}
	if judgment, err = changeJudgmentCodec.NewInitial(judgment); err != nil {
		return ChangeJudgment{}, err
	}
	// A corpus judged again on the same pool is judged the same: the judgment is already there.
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

// ChangeConfirmationRequest asks for one candidate of a judgment to be
// confirmed: the judgment, the candidate by the name the judgment gives it,
// and the directories its proposal was made with.
type ChangeConfirmationRequest struct {
	Judgment        artifact.ID
	Candidate       string
	ModelDirectory  string
	RunsDirectory   string
	OutputDirectory string
}

// ConfirmChange has the promotion part of a judgment's pool answer for one of
// the judgment's candidates: whether, on documents no run was fitted to and
// no candidate was ranked by, the candidate is a gain over the pool's base.
// The candidate's selection reading names its weights and the pool's
// acceptance by identity, so both were fixed before this reads anything. A
// directory that no longer holds those weights is refused, and so is a base
// other than the pool's, before anything is charged. A judgment that scored
// its candidates on the records its corpus's runs were judged on names no
// pool, and is refused.
//
// Every confirmation is charged to the promotion part's budget
// (evaluation.ConfirmOnPool). What returns is the recorded confirmation,
// whose verdict is all it tells of the reading, and what the budget has left.
func ConfirmChange(ctx context.Context, store *overgodb.Store, request ChangeConfirmationRequest) (evaluation.PoolConfirmation, uint64, error) {
	judgment, err := changeJudgmentCodec.Require(ctx, store, request.Judgment)
	if err != nil {
		return evaluation.PoolConfirmation{}, 0, err
	}
	at := slices.IndexFunc(judgment.Candidates, func(candidate ChangeCandidate) bool { return candidate.Name == request.Candidate })
	if at < 0 {
		return evaluation.PoolConfirmation{}, 0, fmt.Errorf("training workflow: judgment %s scores no candidate named %q", judgment.ID, request.Candidate)
	}
	reading, err := evaluation.RequireSelectionReading(ctx, store, judgment.Candidates[at].Report)
	if err != nil {
		return evaluation.PoolConfirmation{}, 0, fmt.Errorf("training workflow: judgment %s scored %s on no held-out pool, so none kept a part from its choice to confirm it on: %w", judgment.ID, request.Candidate, err)
	}
	pool, err := evaluation.RequireHeldoutPool(ctx, store, reading.Acceptance)
	if err != nil {
		return evaluation.PoolConfirmation{}, 0, err
	}
	corpus, err := changeCorpusCodec.Require(ctx, store, judgment.Corpus)
	if err != nil {
		return evaluation.PoolConfirmation{}, 0, err
	}
	directory := candidateDirectory(corpus, request.RunsDirectory, request.OutputDirectory, judgment.Candidates[at])
	for _, named := range []struct {
		directory, role string
		identity        artifact.ID
	}{{directory, "the candidate its judgment ranked", reading.Model}, {request.ModelDirectory, "the pool's base", pool.Acceptance.Base}} {
		held, err := identifyModel(named.directory)
		if err != nil || held != named.identity {
			return evaluation.PoolConfirmation{}, 0, errors.Join(fmt.Errorf("training workflow: %s holds %s, which is not %s, %s; nothing was charged", named.directory, held, named.role, named.identity), err)
		}
	}
	// The models and the documents are prepared by the reading alone, which a
	// confirmation recorded before, or one the budget cannot pay for, never runs.
	return evaluation.ConfirmOnPool(ctx, store, reading.Acceptance, reading.Model, func(documents []string) ([]float64, error) {
		base, examples, err := poolExamples(ctx, store, pool.Acceptance, request.ModelDirectory)
		if err != nil {
			return nil, err
		}
		model, err := loadTrainableModel(directory)
		if err != nil {
			return nil, err
		}
		candidate, err := documentLikelihoods(model, examples, documents)
		if err != nil {
			return nil, err
		}
		baseline, err := documentLikelihoods(base, examples, documents)
		if err != nil {
			return nil, err
		}
		return pairedDifferences(candidate, baseline), nil
	})
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

// resampledDifferences redraws the documents of a population, whole and with
// replacement, under a plan it returns, and gives the mean of the proposal's
// score less the control's over each draw, ascending: larger favours the
// proposal. evidence is the corpus the two were proposed from.
func resampledDifferences(ctx context.Context, population, evidence artifact.ID, proposed, control []evaluation.RecordScore, seedHigh, seedLow uint64) (tensorstats.ResamplingPlan, []float64, error) {
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
		difference[unit] = paired[index]
	}
	plan, err := tensorstats.NewResamplingPlan(tensorstats.ResamplingPlan{
		Population: population, Assumption: assumption, Evidence: evidence, Method: tensorstats.BlockBootstrap,
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

// scoreMean is the mean of a model's scores over the documents it was read on.
func scoreMean(scores []evaluation.RecordScore) float64 {
	var sum float64
	for _, score := range scores {
		sum += score.Value
	}
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
