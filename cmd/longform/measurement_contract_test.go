package main

import (
	"context"
	"errors"
	"io"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/discovery"
	"overgo/internal/longform"
	"overgo/internal/overgodb"
	"overgo/internal/tokenizer"
)

func contractResult(t *testing.T, name string) longform.Result {
	t.Helper()
	model, _, err := artifact.Identify(artifact.KindModel, strings.NewReader(name))
	if err != nil {
		t.Fatal(err)
	}
	floors := longform.DeclaredFloors()
	ids := make([]int32, floors.ShortOutputTokens)
	for i := range ids {
		ids[i] = int32(i)
	}
	measure := longform.Measure{PromptTokens: floors.PromptTokens, OutputTokens: floors.OutputTokens,
		PromptTokensPerSecond: 2000, DecodeTokensPerSecond: 90,
		PromptMilliseconds: 1000, DecodeMilliseconds: 3000,
		Score: longform.ContextScore{ScoreTokens: floors.ScoreTokens, LongContextNLL: 1.1, ShortContextNLL: 1.5, ContextGain: 0.4}}
	result := longform.Result{Inputs: longform.BindInputs(model, "fixed corpus", []tokenizer.TokenID{1, 2, 3}, longform.RawContinuation),
		ModelPath: filepath.Join(t.TempDir(), name+".gguf"), ModelName: name,
		Commit: strings.Repeat("a", 40), Surface: strings.Repeat("b", 64), Floors: floors,
		Measure: measure, Short: longform.ShortRates{PromptTokensPerSecond: 3000, DecodeTokensPerSecond: 100},
		Shape: longform.ShortShape{PromptTokens: floors.ShortPromptTokens, OutputIDs: ids, NLL: 1.1, Measure: measure},
		Rungs: []longform.Rung{{Measure: measure, OutputIDs: ids}}}
	result.Verdict = longform.Judge(measure, result.Short, floors)
	if err := os.WriteFile(result.ModelPath, []byte(name), 0o644); err != nil {
		t.Fatal(err)
	}
	return result
}

func TestCampaignMeasurementContract(t *testing.T) {
	t.Run("corpus export is immutable", func(t *testing.T) {
		root := t.TempDir()
		if err := os.Mkdir(filepath.Join(root, "docs"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("fixed source corpus"), 0o644); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(t.TempDir(), "corpus.txt")
		args := []string{"-root", root, "-export-corpus", path, "-corpus-bytes", "1"}
		if err := run(args, io.Discard); err != nil {
			t.Fatal(err)
		}
		data, err := os.ReadFile(path)
		if err != nil || string(data) != "fixed source corpus\n\n" {
			t.Fatalf("export=%q err=%v", data, err)
		}
		if err := run(args, io.Discard); err == nil {
			t.Fatal("existing frozen corpus overwritten")
		}
		cached := options{Corpus: path, corpusText: string(data)}
		if err := os.WriteFile(path, []byte("later edit"), 0o644); err != nil {
			t.Fatal(err)
		}
		if text, err := readCorpus(cached, 0); err != nil || text != string(data) {
			t.Fatal("in-flight corpus snapshot changed")
		}
	})
	t.Run("explicit comparison inputs", func(t *testing.T) {
		for _, args := range [][]string{{"-all"}, {"-all", "-budget", "1m", "-check"}, {"-all", "-budget", "1m", "-baseline", "absent"}} {
			if _, err := parseOptions(args); err == nil {
				t.Fatalf("accepted %v", args)
			}
		}
		if _, err := parseOptions([]string{"-all", "-budget", "1m", "-model-budget", "10s", "-check", "-corpus", "fixed.txt", "-baseline", "selected"}); err != nil {
			t.Fatal(err)
		}
	})
	t.Run("input changes and missing rungs refuse", func(t *testing.T) {
		record := contractResult(t, "model")
		if verdict := longform.Compare(record, record, record.Floors, 0); !verdict.Passed {
			t.Fatal(verdict)
		}
		for name, mutate := range map[string]func(*longform.Result){
			"non-finite score": func(r *longform.Result) { r.Shape.NLL = math.NaN() },
			"non-finite rate":  func(r *longform.Result) { r.Shape.Measure.DecodeTokensPerSecond = math.Inf(1) },
			"corpus":           func(r *longform.Result) { r.Inputs.CorpusDigest = strings.Repeat("c", 64) },
			"tokens":           func(r *longform.Result) { r.Inputs.TokenDigest = strings.Repeat("d", 64) },
			"model":            func(r *longform.Result) { r.Inputs.Model = artifact.ID{} },
			"protocol":         func(r *longform.Result) { r.Inputs.Protocol = "chat" },
			"floors":           func(r *longform.Result) { r.Floors.NLLTolerance *= 2 },
			"missing rung":     func(r *longform.Result) { r.Rungs = nil },
			"missing output":   func(r *longform.Result) { r.Shape.OutputIDs = nil },
			"slower":           func(r *longform.Result) { r.Shape.Measure.DecodeTokensPerSecond /= 2 },
		} {
			t.Run(name, func(t *testing.T) {
				fresh := record
				mutate(&fresh)
				if verdict := longform.Compare(record, fresh, record.Floors, 0); verdict.Passed {
					t.Fatal("invalid comparison passed")
				}
			})
		}
	})
	t.Run("selected record wins over latest and must be valid", func(t *testing.T) {
		repository := t.TempDir()
		store, err := overgodb.Open(repository)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		record := contractResult(t, "model")
		accepted, err := longform.Publish(t.Context(), store, record.Inputs.Model, record)
		if err != nil {
			t.Fatal(err)
		}
		failed := record
		failed.Verdict = longform.Verdict{Reasons: []string{"known regression"}}
		bad, err := longform.Publish(t.Context(), store, record.Inputs.Model, failed)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := longform.ReadBaseline(t.Context(), store, bad); err == nil {
			t.Fatal("failed baseline accepted")
		}
		legacy := record
		legacy.Inputs = longform.Inputs{}
		old, err := longform.Publish(t.Context(), store, record.Inputs.Model, legacy)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := longform.ReadBaseline(t.Context(), store, old); err == nil {
			t.Fatal("unbound baseline accepted")
		}
		if err := store.Close(); err != nil {
			t.Fatal(err)
		}
		corpus := filepath.Join(t.TempDir(), "corpus.txt")
		if err := os.WriteFile(corpus, []byte("fixed corpus"), 0o644); err != nil {
			t.Fatal(err)
		}
		opts := options{Repository: repository, Corpus: corpus, Baselines: []string{accepted.String()}}
		targets := []target{{weights: record.Inputs.Model}}
		if err := bindBaselines(t.Context(), opts, targets); err != nil {
			t.Fatal(err)
		}
		if targets[0].record.Record != accepted {
			t.Fatal("explicit record was replaced by latest")
		}
		opts.Baselines = append(opts.Baselines, accepted.String())
		if err := bindBaselines(t.Context(), opts, targets); err == nil {
			t.Fatal("duplicate baseline accepted")
		}
		opts.Baselines = opts.Baselines[:1]
		if err := os.WriteFile(corpus, []byte("changed corpus"), 0o644); err != nil {
			t.Fatal(err)
		}
		if err := bindBaselines(t.Context(), opts, targets); err == nil {
			t.Fatal("changed corpus accepted")
		}
	})
	t.Run("same surface still measures", func(t *testing.T) {
		record := contractResult(t, "model")
		calls := 0
		measure := func(context.Context, io.Writer, options, target, string, string, longform.Floors, int) (longform.Result, error) {
			calls++
			fresh := record
			fresh.Shape.Measure.DecodeTokensPerSecond /= 2
			return fresh, nil
		}
		err := runTargets(t.Context(), io.Discard, options{Check: true}, []target{{record: longform.Summary{Result: record}}}, record.Commit, record.Surface, measure, nil)
		if calls != 1 || err == nil {
			t.Fatalf("calls=%d error=%v; same surface bypassed fresh comparison", calls, err)
		}
	})
	t.Run("aggregate cancellation preserves earlier publication", func(t *testing.T) {
		repository := t.TempDir()
		record := contractResult(t, "first")
		ctx, cancel := context.WithCancelCause(t.Context())
		defer cancel(nil)
		calls := 0
		measure := func(context.Context, io.Writer, options, target, string, string, longform.Floors, int) (longform.Result, error) {
			calls++
			return record, nil
		}
		var output strings.Builder
		targets := []target{{weights: record.Inputs.Model, entry: discovery.Entry{Location: record.ModelPath}}, {entry: discovery.Entry{Location: "unfinished.gguf"}}}
		opts := options{Publish: true, Repository: repository}
		err := withPublisher(opts, func(publish publishModel) error {
			publishThenCancel := func(ctx context.Context, model artifact.ID, result longform.Result) (artifact.ID, error) {
				id, err := publish(ctx, model, result)
				cancel(context.Canceled)
				return id, err
			}
			return runTargets(ctx, &output, opts, targets, record.Commit, record.Surface, measure, publishThenCancel)
		})
		if !errors.Is(err, context.Canceled) || calls != 1 || !strings.Contains(output.String(), "UNFINISHED: unfinished.gguf") {
			t.Fatalf("calls=%d err=%v output=%s", calls, err, output.String())
		}
		store, err := overgodb.OpenReadOnly(repository)
		if err != nil {
			t.Fatal(err)
		}
		defer store.Close()
		if _, found := longform.Latest(t.Context(), store, record.ModelPath); !found {
			t.Fatal("completed model evidence was lost")
		}
	})
	t.Run("model deadline does not cancel next model", func(t *testing.T) {
		record := contractResult(t, "model")
		calls, published := 0, 0
		measure := func(ctx context.Context, _ io.Writer, _ options, _ target, _, _ string, _ longform.Floors, _ int) (longform.Result, error) {
			calls++
			if calls == 1 {
				<-ctx.Done()
				return longform.Result{}, ctx.Err()
			}
			return record, nil
		}
		publish := func(context.Context, artifact.ID, longform.Result) (artifact.ID, error) {
			published++
			return artifact.ID{}, nil
		}
		err := runTargets(t.Context(), io.Discard, options{Publish: true, ModelBudget: time.Millisecond}, []target{{}, {}}, record.Commit, record.Surface, measure, publish)
		if !errors.Is(err, context.DeadlineExceeded) || calls != 2 || published != 1 {
			t.Fatalf("calls=%d published=%d err=%v", calls, published, err)
		}
	})
}
