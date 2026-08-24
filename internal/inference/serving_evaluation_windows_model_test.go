//go:build windows && modeltest

package inference_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/servingtest"
)

// evaluateExactGGUF runs an exact suite through the compiled campaign path
// against a temporary, fully grounded serving repository.
func evaluateExactGGUF(ctx context.Context, path string, suite []byte) (evaluation.CampaignResult, error) {
	if ctx == nil {
		return evaluation.CampaignResult{}, errors.New("serving test: evaluation context is absent")
	}
	root, err := os.MkdirTemp("", "overgo-evaluation-fixture-")
	if err != nil {
		return evaluation.CampaignResult{}, err
	}
	defer os.RemoveAll(root)
	store, err := overgodb.Open(root)
	if err != nil {
		return evaluation.CampaignResult{}, err
	}
	fail := func(cause error) (evaluation.CampaignResult, error) {
		return evaluation.CampaignResult{}, errors.Join(cause, store.Close())
	}
	if err := servingtest.PublishActiveGGUFWithPolicy(
		ctx, store, path, recipe.PlacementHybrid,
		modelrecipe.DecodeSessionCapacity, recipe.ResidencyDeviceNative,
	); err != nil {
		return fail(err)
	}
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, path)
	if err != nil {
		return fail(err)
	}
	identity, err := loaded.Identity()
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	runner, err := inference.OpenWithProgram(ctx, &loaded, inference.OpenOptions{})
	if err != nil {
		_ = loaded.Close()
		return fail(err)
	}
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
	campaign, err := evaluation.NewCampaign(store, runner, identity, environment, strings.Repeat("0", 40))
	if err != nil {
		_ = runner.Close()
		return fail(err)
	}
	compiled, err := evaluation.CompileSuite(suite, campaign.Authorities())
	if err != nil {
		_ = runner.Close()
		return fail(fmt.Errorf("serving test: compile exact suite: %w", err))
	}
	result, evaluateErr := campaign.Evaluate(ctx, compiled)
	return result, errors.Join(evaluateErr, runner.Close(), store.Close())
}
