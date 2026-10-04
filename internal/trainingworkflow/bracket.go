package trainingworkflow

import (
	"cmp"
	"context"
	"errors"
	"slices"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/overgodb"
)

const (
	trainingBracketMediaType = "application/vnd.overgo.training-bracket+json"
	trainingBracketSchema    = "overgo/training-bracket/v1"
)

// BracketSlice is one side of a training bracket: the committed
// benchmark and evaluation evidence standing for a model at a point in
// time. An empty slice is exact -- it records that no evidence stood.
type BracketSlice struct {
	Benchmark *evaluation.BenchmarkSummary `json:"benchmark,omitempty"`
	Evals     []evaluation.EvalSummary     `json:"evals,omitempty"`
}

// MetricDelta is one machine-checked improvement claim: the same suite
// metric before and after the session.
type MetricDelta struct {
	Suite      string      `json:"suite"`
	Metric     string      `json:"metric"`
	Pre        float64     `json:"pre"`
	Post       float64     `json:"post"`
	Delta      float64     `json:"delta"`
	PreRecord  artifact.ID `json:"pre_record,omitzero"`
	PostRecord artifact.ID `json:"post_record,omitzero"`
}

// TrainingBracket binds a training session to the evidence standing
// before and after it: improvement is the delta between committed
// records, never a narrated claim.
type TrainingBracket struct {
	ID      artifact.ID   `json:"-"`
	Version uint16        `json:"version"`
	Session artifact.ID   `json:"session"`
	Model   artifact.ID   `json:"model"`
	Recipe  artifact.ID   `json:"recipe"`
	Pre     BracketSlice  `json:"pre"`
	Post    BracketSlice  `json:"post"`
	Deltas  []MetricDelta `json:"deltas,omitempty"`
}

var trainingBracketCodec = artifact.JSONDocumentCodec(
	"training bracket", artifact.KindEvidence, trainingBracketMediaType, trainingBracketSchema,
	validateTrainingBracket, func(value TrainingBracket) artifact.ID { return value.ID },
	func(value *TrainingBracket, id artifact.ID) { value.ID = id },
	func(value TrainingBracket) TrainingBracket {
		value.Pre.Evals = slices.Clone(value.Pre.Evals)
		value.Post.Evals = slices.Clone(value.Post.Evals)
		value.Deltas = slices.Clone(value.Deltas)
		return value
	},
)

func validateTrainingBracket(value *TrainingBracket) error {
	if value == nil || value.Version != artifact.InitialDocumentVersion ||
		value.Session.Kind() != artifact.KindEvidence ||
		value.Model.Kind() != artifact.KindModel || value.Recipe.Kind() != artifact.KindRecipe {
		return errors.New("training workflow: invalid bracket")
	}
	return nil
}

// captureBracketSlice reads the evidence standing for one model and
// recipe: the latest benchmark claim registered against any of the
// model's recorded file locations, and the latest evaluation per
// derived suite for the recipe.
func captureBracketSlice(
	ctx context.Context,
	store *overgodb.Store,
	model, recipeID artifact.ID,
) BracketSlice {
	names, _ := evaluation.DerivedSuiteCatalog(ctx, store, evaluation.ListingAuthorities())
	index := evaluation.LatestEvidence(ctx, store, names, bracketEvidenceScanLimit)
	slice := BracketSlice{Evals: index.EvaluationsByRecipe[recipeID]}
	if locations, err := store.Locations(ctx, model); err == nil {
		for _, location := range locations {
			if location.Kind != artifact.LocationFile {
				continue
			}
			if summary, measured := index.BenchmarksByLocation[location.Value]; measured {
				slice.Benchmark = &summary
				break
			}
		}
	}
	return slice
}

// bracketEvidenceScanLimit bounds each evidence scan behind a bracket
// capture; brackets read the latest records, not the full history.
const bracketEvidenceScanLimit = 4096

// bracketDeltas derives the machine-checked improvements: one delta
// per suite metric present on both sides, plus the decode rate when
// both benchmarks bind the same known measurement contract and pair records.
func bracketDeltas(pre, post BracketSlice) []MetricDelta {
	preMetrics := map[artifact.ID]evaluation.EvalSummary{}
	for _, entry := range pre.Evals {
		if entry.ExecutionKnown && entry.Comparison.Valid() {
			preMetrics[entry.Comparison] = entry
		}
	}
	var deltas []MetricDelta
	for _, entry := range post.Evals {
		before, measured := preMetrics[entry.Comparison]
		if !entry.ExecutionKnown || !measured {
			continue
		}
		for metric, value := range entry.Metrics {
			previous, present := before.Metrics[metric]
			if !present || entry.Directions[metric] == "" || entry.Directions[metric] != before.Directions[metric] {
				continue
			}
			deltas = append(deltas, MetricDelta{
				Suite: entry.Suite, Metric: metric, Pre: previous, Post: value, Delta: value - previous,
				PreRecord: before.Record, PostRecord: entry.Record,
			})
		}
	}
	if pre.Benchmark != nil && post.Benchmark != nil &&
		pre.Benchmark.Comparison.Kind() == artifact.KindProfile && pre.Benchmark.Comparison == post.Benchmark.Comparison &&
		pre.Benchmark.Record.Kind() == artifact.KindEvidence && post.Benchmark.Record.Kind() == artifact.KindEvidence &&
		pre.Benchmark.DecodeTokensPerSecond > 0 && post.Benchmark.DecodeTokensPerSecond > 0 {
		deltas = append(deltas, MetricDelta{
			Suite: "benchmark", Metric: "decode_tokens_per_second_p50",
			Pre:       pre.Benchmark.DecodeTokensPerSecond,
			Post:      post.Benchmark.DecodeTokensPerSecond,
			Delta:     post.Benchmark.DecodeTokensPerSecond - pre.Benchmark.DecodeTokensPerSecond,
			PreRecord: pre.Benchmark.Record, PostRecord: post.Benchmark.Record,
		})
	}
	slices.SortFunc(deltas, func(left, right MetricDelta) int {
		return cmp.Or(cmp.Compare(left.Suite, right.Suite), cmp.Compare(left.Metric, right.Metric), artifact.CompareID(left.PreRecord, right.PreRecord), artifact.CompareID(left.PostRecord, right.PostRecord))
	})
	return deltas
}

// publishTrainingBracket commits the bracket with lineage to its
// session observation, model, and recipe.
func publishTrainingBracket(
	ctx context.Context,
	store artifact.Repository,
	session, model, recipeID artifact.ID,
	pre, post BracketSlice,
) (artifact.ID, error) {
	bracket, batch, err := trainingBracketBatch(session, model, recipeID, pre, post)
	if err != nil {
		return artifact.ID{}, err
	}
	if _, err := artifact.CommitBatch(ctx, store, batch); err != nil {
		return artifact.ID{}, err
	}
	return bracket.ID, nil
}

func trainingBracketBatch(
	session, model, recipeID artifact.ID,
	pre, post BracketSlice,
) (TrainingBracket, artifact.Batch, error) {
	bracket, err := trainingBracketCodec.New(TrainingBracket{
		Version: artifact.InitialDocumentVersion, Session: session, Model: model, Recipe: recipeID,
		Pre: pre, Post: post, Deltas: bracketDeltas(pre, post),
	})
	if err != nil {
		return TrainingBracket{}, artifact.Batch{}, err
	}
	batch, err := trainingBracketCodec.Batch(
		"training-bracket/"+bracket.ID.String(), bracket,
		[]artifact.Lineage{
			{Child: bracket.ID, Parent: session, Relation: artifact.RelationDependsOn},
			{Child: bracket.ID, Parent: model, Relation: artifact.RelationDependsOn},
		}, nil,
	)
	if err != nil {
		return TrainingBracket{}, artifact.Batch{}, err
	}
	return bracket, batch, nil
}
