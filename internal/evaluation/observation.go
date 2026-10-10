package evaluation

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/strictjson"
)

const (
	// observedKey prefixes the key each record of answers is published under,
	// before what was asked and the case the record begins at. The store holds
	// one content under a key, so the first record written for a case stands
	// and a second that says otherwise is refused.
	observedKey = "evaluation/observations/"
	// observedPending is the most answers an evaluation holds before it writes
	// them down, and observedWait the longest it goes without writing while it
	// holds one: between them, what a process killed outright loses. A suite a
	// model answers in a fraction of a second a case is written a record every
	// observedPending cases; one it takes that long a case to answer, as it does
	// the instruction rules' long generations, is written case by case. An
	// evaluation that is cancelled, or whose model fails on a case, writes what
	// it holds first.
	observedPending = 32
	observedWait    = 10 * time.Second
)

// caseAnswers is a run of one execution's answers to consecutive cases, as
// the model gave them and before anything scores them. Observed names what
// was asked of what (Plan.observed), First is the case the run begins at, and
// each of Cases is one case's answer in the content its kind of evaluation
// keeps: what a model wrote, or the score it gave each candidate.
type caseAnswers struct {
	ID       artifact.ID       `json:"-"`
	Version  uint16            `json:"version"`
	Observed artifact.ID       `json:"observed"`
	First    uint64            `json:"first"`
	Cases    []json.RawMessage `json:"cases"`
}

var caseAnswersCodec = artifact.JSONDocumentCodec(
	"evaluation answers", artifact.KindOutput,
	"application/vnd.overgo.evaluation-answers+json", "overgo/evaluation-answers/v1",
	func(value *caseAnswers) error {
		if value == nil || value.Version != artifact.InitialDocumentVersion || value.Observed.Kind() != artifact.KindProfile || len(value.Cases) == 0 {
			return errors.New("evaluation: invalid record of answers")
		}
		return nil
	},
	func(value caseAnswers) artifact.ID { return value.ID },
	func(value *caseAnswers, id artifact.ID) { value.ID = id },
	func(value caseAnswers) caseAnswers {
		value.Cases = slices.Clone(value.Cases)
		return value
	},
)

// observed names what a plan's execution is asked: everything a case's answer
// depends on, and nothing its score does. That is the plan's model, recipe,
// environment, commit and execution policy, and asked, what each case gives
// the model, in the cases' order. Two plans that differ in their scorer, or in
// what their cases' answers are scored against, name the same answers.
func (p Plan) observed(asked any) (artifact.ID, error) {
	return artifact.JSONID(artifact.KindProfile, struct {
		ModelDefinition artifact.ID `json:"model_definition"`
		RuntimeRecipe   artifact.ID `json:"runtime_recipe"`
		Execution       artifact.ID `json:"execution"`
		CodeCommit      string      `json:"code_commit"`
		Environment     artifact.ID `json:"environment"`
		Asked           any         `json:"asked"`
	}{p.body.ModelDefinition, p.body.RuntimeRecipe, p.body.Execution, p.body.CodeCommit, p.body.Environment, asked})
}

// answers is what one execution has answered of the cases it is asked, in
// their order: the answers the store held when the evaluation began, and each
// one given since. An evaluation asks for its cases' answers in order and
// scores what it is given, so a case answered once, by this run or one before
// it, under this scorer or another, is not put to the model again.
type answers[T any] struct {
	repository artifact.Repository
	observed   artifact.ID
	values     []T
	// written is how many of values the store holds, and wrote when it last
	// held them all.
	written int
	wrote   time.Time
}

// loadAnswers reads the answers the store holds to what a plan's execution is
// asked: the records that name it, from the first case on, as far as they
// follow one another.
func loadAnswers[T any](ctx context.Context, repository artifact.Repository, plan Plan, asked any) (*answers[T], error) {
	observed, err := plan.observed(asked)
	if err != nil {
		return nil, err
	}
	edges, err := repository.Children(ctx, observed)
	if err != nil {
		return nil, err
	}
	// A report scored from the answers names what they were asked of as well; the records of answers are the outputs.
	var ids []artifact.ID
	for _, edge := range edges {
		if edge.Child.Kind() == artifact.KindOutput {
			ids = append(ids, edge.Child)
		}
	}
	var records []caseAnswers
	if err := artifact.ReadContents(ctx, repository, ids, func(content artifact.Content) error {
		record, err := caseAnswersCodec.Parse(content.Data)
		if err != nil || record.Observed != observed {
			return errors.Join(fmt.Errorf("evaluation: record %s is not of the answers it names", content.Descriptor.ID), err)
		}
		records = append(records, record)
		return nil
	}); err != nil {
		return nil, err
	}
	slices.SortFunc(records, func(left, right caseAnswers) int { return cmp.Compare(left.First, right.First) })
	held := &answers[T]{repository: repository, observed: observed, wrote: time.Now()}
	for _, record := range records {
		if record.First != uint64(len(held.values)) {
			break
		}
		for _, encoded := range record.Cases {
			var value T
			if err := strictjson.DecodeBytes(encoded, &value); err != nil {
				return nil, fmt.Errorf("evaluation: record %s of answers: %w", record.ID, err)
			}
			held.values = append(held.values, value)
		}
	}
	held.written = len(held.values)
	return held, nil
}

// answer is the answer to the case at index, which is one already answered or
// the next: for the next, run puts the case to the model, and its answer is
// written down with those before it once observedPending are held or
// observedWait has passed since the last were. When the evaluation has been
// cancelled, or run fails, the answers held are written down before the error
// returns.
func (held *answers[T]) answer(ctx context.Context, index int, run func() (T, error)) (T, error) {
	var none T
	if index < len(held.values) {
		return held.values[index], nil
	}
	if index != len(held.values) {
		return none, fmt.Errorf("evaluation: case %d is asked before case %d is answered", index, len(held.values))
	}
	if err := ctx.Err(); err != nil {
		return none, errors.Join(err, held.write(ctx))
	}
	value, err := run()
	if err != nil {
		return none, errors.Join(err, held.write(ctx))
	}
	held.values = append(held.values, value)
	if len(held.values)-held.written < observedPending && time.Since(held.wrote) < observedWait {
		return value, nil
	}
	return value, held.write(ctx)
}

// write writes down the answers the store does not hold yet, as one record
// that begins where the last ended. It is not stopped by a cancellation: what
// ended the evaluation does not take with it an answer the model has given.
func (held *answers[T]) write(ctx context.Context) error {
	if held.written == len(held.values) {
		return nil
	}
	record := caseAnswers{Observed: held.observed, First: uint64(held.written)}
	for _, value := range held.values[held.written:] {
		encoded, err := json.Marshal(value)
		if err != nil {
			return err
		}
		record.Cases = append(record.Cases, encoded)
	}
	record, err := caseAnswersCodec.NewInitial(record)
	if err != nil {
		return err
	}
	batch, err := caseAnswersCodec.Batch(fmt.Sprintf("%s%s/%d", observedKey, held.observed, held.written), record,
		artifact.DependencyLineage(record.ID, held.observed), nil)
	if err != nil {
		return err
	}
	// What was asked is an identity and holds no content of its own: it is declared with each record that names it.
	batch.Artifacts = append(batch.Artifacts, artifact.Descriptor{ID: held.observed})
	if _, err := artifact.Publish(context.WithoutCancel(ctx), held.repository, batch); err != nil {
		return fmt.Errorf("evaluation: the answers to cases %d to %d were not written down: %w", held.written, len(held.values)-1, err)
	}
	held.written, held.wrote = len(held.values), time.Now()
	return nil
}
