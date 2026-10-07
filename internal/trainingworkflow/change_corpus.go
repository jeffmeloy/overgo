package trainingworkflow

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/jsonfile"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/trainingprogram"
)

const (
	changeCorpusMediaType = "application/vnd.overgo.change-corpus+json"
	changeCorpusSchema    = "overgo/change-corpus/v1"
)

// WeightChange is one bounded fine-tune as the change it made to its base
// model: the checkpoint the run published, which with the base's weights is
// the change; the order seed the run read its training records by, which is
// all that sets it apart from the corpus's other runs; and the held-out
// verdict the change earned, with the metric it read for the base model and
// for the changed one.
type WeightChange struct {
	Order       uint64      `json:"order"`
	Observation artifact.ID `json:"observation"`
	Trained     artifact.ID `json:"trained"`
	Verdict     artifact.ID `json:"verdict"`
	Base        float64     `json:"base"`
	Changed     float64     `json:"changed"`
}

// ChangeCorpus is the changes of one base model that fine-tunes under one
// objective made, each bound to its verdict. Every verdict scored its two
// models under the same target plan on the same held-out inputs, so the
// changes are judged alike and their verdicts compare record by record.
type ChangeCorpus struct {
	ID        artifact.ID                      `json:"-"`
	Version   uint16                           `json:"version"`
	Objective artifact.ID                      `json:"objective"`
	Model     artifact.ID                      `json:"model"`
	Plan      artifact.ID                      `json:"plan"`
	Inputs    artifact.ID                      `json:"inputs"`
	Metric    trainingprogram.EvaluationMetric `json:"metric"`
	Direction runrecord.Direction              `json:"direction"`
	Changes   []WeightChange                   `json:"changes"`
}

var changeCorpusCodec = artifact.JSONDocumentCodec(
	"change corpus", artifact.KindEvidence, changeCorpusMediaType, changeCorpusSchema,
	canonicalizeChangeCorpus,
	func(value ChangeCorpus) artifact.ID { return value.ID },
	func(value *ChangeCorpus, id artifact.ID) { value.ID = id },
	func(value ChangeCorpus) ChangeCorpus {
		value.Changes = slices.Clone(value.Changes)
		return value
	},
)

// canonicalizeChangeCorpus orders the changes by their order seed and
// refuses a corpus two of whose changes share a seed, a checkpoint or a
// verdict: each of those is one change counted twice.
func canonicalizeChangeCorpus(value *ChangeCorpus) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion || value.Objective.Kind() != artifact.KindProfile ||
		value.Model.Kind() != artifact.KindModel || value.Plan.Kind() != artifact.KindProfile || value.Inputs.Kind() != artifact.KindProfile ||
		value.Metric == "" || len(value.Changes) == 0 {
		return errors.New("training workflow: incomplete change corpus")
	}
	slices.SortFunc(value.Changes, func(left, right WeightChange) int { return cmp.Compare(left.Order, right.Order) })
	trained, verdicts := map[artifact.ID]bool{}, map[artifact.ID]bool{}
	for index, change := range value.Changes {
		if change.Observation.Kind() != artifact.KindEvidence || change.Trained.Kind() != artifact.KindCheckpoint || change.Verdict.Kind() != artifact.KindEvaluation {
			return errors.New("training workflow: a change names no session, checkpoint or verdict")
		}
		if index > 0 && value.Changes[index-1].Order == change.Order || trained[change.Trained] || verdicts[change.Verdict] {
			return fmt.Errorf("training workflow: the change of order %d is in the corpus twice", change.Order)
		}
		trained[change.Trained], verdicts[change.Verdict] = true, true
	}
	return nil
}

// changeRun is one fine-tune a corpus collects: the order seed it read its
// training records by and what its held-out run published, the session, the
// checkpoint and the verdict.
type changeRun struct {
	Order       uint64      `json:"order"`
	Observation artifact.ID `json:"observation"`
	Trained     artifact.ID `json:"trained"`
	Verdict     artifact.ID `json:"verdict"`
}

const (
	// changeRunFile is the file a corpus run leaves in its output directory
	// once its verdict is published: its order seed and what it published,
	// so a corpus that stopped part way takes its finished runs as they are.
	changeRunFile = "change.json"
	// changeRunFileMode keeps a run's record to its owner.
	changeRunFileMode = 0o600
)

// CollectChangeCorpus trains count bounded runs of one base model under one
// objective and publishes them as a change corpus. The runs are the request's
// run, each reading its training records by its own order seed, the seeds
// counting up from first, and each publishing in a directory named after its
// seed under the request's output directory. A run whose directory already
// holds its finished record is not trained again, so a corpus that stopped
// part way, or one asked to grow, trains only the runs it lacks.
func CollectChangeCorpus(ctx context.Context, store *overgodb.Store, request TextHeldoutRequest, first uint64, count int) (ChangeCorpus, error) {
	if request.Proof || request.Resume != "" || request.CheckpointEvery > 0 || request.Order != 0 {
		return ChangeCorpus{}, errors.New("training workflow: a change corpus trains whole runs under its own order seeds and judges each: it takes no proof run, resumed run, saved steps or order seed of the request's")
	}
	if first == 0 || count <= 0 || request.OutputDirectory == "" {
		return ChangeCorpus{}, errors.New("training workflow: a change corpus needs a first order seed that is not zero, a positive count of runs and an output directory")
	}
	runs := make([]changeRun, count)
	for index := range runs {
		order := first + uint64(index)
		run := request
		run.Order, run.OutputDirectory = order, filepath.Join(request.OutputDirectory, fmt.Sprintf("order-%04d", order))
		record := filepath.Join(run.OutputDirectory, changeRunFile)
		if err := jsonfile.DecodeStrict(record, &runs[index]); err == nil && runs[index].Order == order {
			if request.Progress != nil {
				fmt.Fprintf(request.Progress, "order %d: finished before, verdict %s\n", order, runs[index].Verdict)
			}
			continue
		}
		result, err := RunTextHeldout(ctx, store, run)
		if err != nil {
			return ChangeCorpus{}, fmt.Errorf("training workflow: the run of order %d: %w", order, err)
		}
		runs[index] = changeRun{Order: order, Observation: result.Observation, Trained: result.Trained, Verdict: result.Verdict}
		if err := jsonfile.Write(record, runs[index], changeRunFileMode); err != nil {
			return ChangeCorpus{}, err
		}
	}
	return publishChangeCorpus(ctx, store, runs)
}

// publishChangeCorpus binds each run's change to the verdict it earned and
// publishes the corpus. Each verdict is read back from the store and
// reconstructed, and must judge the checkpoint and the session its run
// names; the runs must have trained one base model under one objective and
// been scored under one target plan on the same held-out inputs. A run that
// fails any of these is refused with its order, and nothing is published.
func publishChangeCorpus(ctx context.Context, store artifact.Repository, runs []changeRun) (ChangeCorpus, error) {
	var corpus ChangeCorpus
	parents := make([]artifact.ID, 0, len(runs)+len(runs))
	for index, run := range runs {
		facts, err := evaluation.ReadHeldoutVerdict(ctx, store, run.Verdict)
		if err != nil {
			return ChangeCorpus{}, fmt.Errorf("training workflow: the change of order %d: %w", run.Order, err)
		}
		if facts.Trained != run.Trained || facts.Observation != run.Observation {
			return ChangeCorpus{}, fmt.Errorf("training workflow: the change of order %d names verdict %s, which judged another run's checkpoint", run.Order, run.Verdict)
		}
		session, err := runrecord.RequireServingObservation(ctx, store, facts.Observation)
		if err != nil {
			return ChangeCorpus{}, err
		}
		if index == 0 {
			corpus = ChangeCorpus{Objective: facts.Objective, Model: session.Model, Plan: facts.Plan, Inputs: facts.Inputs, Metric: facts.Metric, Direction: facts.Direction}
		}
		if facts.Objective != corpus.Objective || session.Model != corpus.Model || facts.Plan != corpus.Plan || facts.Inputs != corpus.Inputs {
			return ChangeCorpus{}, fmt.Errorf("training workflow: the change of order %d trained another model or objective, or was scored on other held-out inputs, than the corpus's first", run.Order)
		}
		corpus.Changes = append(corpus.Changes, WeightChange{
			Order: run.Order, Observation: facts.Observation, Trained: facts.Trained, Verdict: run.Verdict,
			Base: facts.BaselineValue, Changed: facts.CandidateValue,
		})
		parents = append(parents, facts.Trained, run.Verdict)
	}
	corpus, err := changeCorpusCodec.NewInitial(corpus)
	if err != nil {
		return ChangeCorpus{}, err
	}
	batch, err := changeCorpusCodec.Batch("training/change-corpus/"+corpus.ID.String(), corpus, artifact.DependencyLineage(corpus.ID, parents...), nil)
	if err != nil {
		return ChangeCorpus{}, err
	}
	if _, err := artifact.Publish(ctx, store, batch); err != nil {
		return ChangeCorpus{}, err
	}
	return corpus, nil
}
