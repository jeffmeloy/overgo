package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

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
	if value.ChatProtocol {
		runtime = chatShapedRuntime{runner}
	}
	campaign, err := evaluation.NewCampaign(store, runtime, identity, environment, value.CodeCommit)
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
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
				trimmed := strings.TrimPrefix(candidate, " ")
				if trimmed == "" {
					trimmed = candidate
				}
				opening[index] = trimmed
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
	_, err = s.campaign.Evaluate(ctx, compiled)
	return err
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
