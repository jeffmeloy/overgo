package main

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/runrecord"
	"overgo/internal/sequencescore"
)

type evaluationSession interface {
	Evaluate(context.Context, string) error
	Close() error
}

type sessionOpener func(context.Context, manifest, modelRequest) (evaluationSession, error)

func executeModel(ctx context.Context, value manifest, request modelRequest, open sessionOpener) error {
	if ctx == nil || open == nil || len(request.Suites) == 0 {
		return errors.New("evaluate: incomplete model worker")
	}
	session, err := open(ctx, value, request)
	if err != nil {
		return err
	}
	for _, suite := range request.Suites {
		if err := session.Evaluate(ctx, suite); err != nil {
			return errors.Join(err, session.Close())
		}
	}
	return session.Close()
}

type nativeSession struct {
	store    *overgodb.Store
	runner   *inference.Runner
	campaign *evaluation.Campaign
	model    artifact.ID
}

func openEvaluationSession(ctx context.Context, value manifest, request modelRequest) (evaluationSession, error) {
	store, err := overgodb.Open(value.Repository)
	if err != nil {
		return nil, err
	}
	fail := func(cause error) (evaluationSession, error) {
		return nil, errors.Join(cause, store.Close())
	}
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, request.Path)
	if err != nil {
		return fail(err)
	}
	identity, err := loaded.Identity()
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{DeviceOrdinal: value.Device})
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	environment, err := runrecord.CurrentEnvironment(fmt.Sprintf("cuda:%d", value.Device), "cuda")
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
	runtime := evaluation.Runtime(runner)
	prompting := evaluation.PromptingRawCompletion
	if value.ChatProtocol {
		runtime = chatShapedRuntime{runner}
		prompting = evaluation.PromptingChatTemplate
	}
	campaign, err := evaluation.NewIsolatedCampaign(store, runtime, identity, environment, value.CodeCommit)
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
	campaign.WithPrompting(prompting)
	return &nativeSession{store: store, runner: runner, campaign: campaign, model: identity.Model}, nil
}

// chatShapedRuntime scores multiple-choice prompts through the model's
// own declared conversation framing: an instruct-tuned model measures
// the format it was trained to answer in, and a model without any chat
// declaration scores the raw prompt unchanged. Pure-sequence scoring —
// the empty prompt — stays raw, because shaping would corrupt
// full-sequence likelihoods.
type chatShapedRuntime struct {
	*inference.Runner
}

// ScoreContinuations shapes a non-empty prompt through the declared
// chat template before scoring; see the type comment for the contract.
// A shaped prompt ends at the assistant turn opener, so candidates
// score as that turn's opening tokens: the raw-completion leading space
// belongs to the base-style "Answer:" continuation and mis-tokenizes
// after the opener's newline, measured as below-random letter scores.
func (r chatShapedRuntime) ScoreContinuations(
	ctx context.Context,
	prompt string,
	candidates []string,
) ([]sequencescore.Score, error) {
	if prompt != "" {
		shaped, err := r.Runner.FormatChatWithOptions(
			[]inference.ChatMessage{{Role: inference.ChatRoleUser, Content: prompt}},
			inference.ChatFormatOptions{AddGenerationPrompt: true},
		)
		if err == nil {
			opening := make([]string, len(candidates))
			for index, candidate := range candidates {
				opening[index] = cmp.Or(strings.TrimPrefix(candidate, " "), candidate)
			}
			return r.Runner.ScoreContinuations(ctx, shaped, opening)
		}
	}
	return r.Runner.ScoreContinuations(ctx, prompt, candidates)
}

func (s *nativeSession) Evaluate(ctx context.Context, path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	compiled, err := evaluation.CompileSuite(data, s.campaign.Authorities())
	if err != nil {
		return fmt.Errorf("evaluate: compile suite %q: %w", path, err)
	}
	_, err = s.campaign.Evaluate(evaluation.WithProgress(ctx, printProgress), compiled)
	return err
}

// printProgress reports a suite's case loop at a geometric cadence: at
// every power-of-two case count and at completion. The cadence is
// derived from the count itself, so a suite of N cases prints about
// log2(N) lines whatever its speed, each carrying the running score,
// the measured rate, and the remaining time that rate projects -- a
// 13-hour suite (the 27B over BBH) is distinguishable from a hang
// within its first seconds, and its cost is known long before it ends.
func printProgress(value evaluation.Progress) {
	if value.Done != value.Total && value.Done&(value.Done-1) != 0 {
		return
	}
	perCase := value.Elapsed / time.Duration(value.Done)
	remaining := perCase * time.Duration(value.Total-value.Done)
	score := "-"
	if value.Scored {
		// The printed precision resolves one case of the suite: as many
		// decimals as the case count has digits.
		score = strconv.FormatFloat(value.Score, 'f', len(strconv.Itoa(value.Total)), 64)
	}
	fmt.Printf("  progress %s: %d/%d score=%s elapsed=%s per-case=%s remaining=%s\n",
		value.Suite, value.Done, value.Total, score,
		value.Elapsed.Round(time.Second), perCase.Round(time.Millisecond), remaining.Round(time.Second))
}

func (s *nativeSession) Close() error {
	if s == nil {
		return nil
	}
	var runnerErr error
	if s.runner != nil {
		runnerErr = s.runner.Close()
		s.runner = nil
	}
	storeErr := s.store.Close()
	s.store = nil
	return errors.Join(runnerErr, storeErr)
}
