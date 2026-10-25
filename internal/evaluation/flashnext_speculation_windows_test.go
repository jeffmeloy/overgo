//go:build windows && modeltest

package evaluation

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"overgo/internal/artifact"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelcli"
	"overgo/internal/overgodb"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

// Flash-Next's first shard under a model root, and Strata's MTP draft layer for it under the references beside
// the store.
var (
	flashNextModel = filepath.Join("Qwen3.8-Flash-Next", "unsloth-UD-IQ4_XS", "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf")
	flashNextDraft = filepath.Join("strata", "mtp", "mtp-q2_0.gguf")
)

// speculationHeldOut is IFEval's held-out part, three hundred cases taken evenly through the whole (the hundred-case
// part among them), and speculationDrop the share of strict prompts speculation may lose against plain decoding: 5%,
// about two standard errors of a share near 0.7 at that size. Speculation and plain decoding part at rounding level
// within a few dozen tokens, and sooner since the routed products take int8 activations (a quantum flips where the
// arms' sums differ in their last bits), so their answers differ as two decodings of the same model do: on
// 2026-10-21 the hundred-case part, where 5% is one standard error, held 77 against 79 and then moved 73 against 80
// while its three hundred cases held 72.3% against 74.0%.
const (
	speculationHeldOut = "ifeval/300"
	speculationDrop    = 0.05
)

// speculationRuntime answers a case as the chat-template protocol asks it, one user turn with thinking off, and
// counts the tokens it generated and drafted.
//   - plain => it decodes a token a step though the runner holds a draft
type speculationRuntime struct {
	*inference.Runner
	plain                        bool
	generated, drafted, accepted int
}

func (r *speculationRuntime) Generate(ctx context.Context, prompt string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	options.PlainDecode = r.plain
	options.OnDraftSpan = func(drafted, accepted int) { r.drafted, r.accepted = r.drafted+drafted, r.accepted+accepted }
	evaluated := options.OnPromptEvaluated
	prompted := 0
	options.OnPromptEvaluated = func(evaluation inference.PromptEvaluation) {
		prompted = evaluation.Tokens
		if evaluated != nil {
			evaluated(evaluation)
		}
	}
	ids, text, err := r.Runner.Generate(ctx, prompt, options)
	r.generated += len(ids) - prompted
	return ids, text, err
}

func (r *speculationRuntime) ShapeChatPrompt(prompt string) (string, error) {
	return r.FormatChatWithOptions(
		[]inference.ChatMessage{{Role: inference.ChatRoleUser, Content: prompt}},
		inference.ChatFormatOptions{AddGenerationPrompt: true},
	)
}

// TestFlashNextSpeculationHoldsIFEval: Flash-Next follows the held-out IFEval part's instructions as well when its
// greedy decoding speculates through the attached MTP draft, as a runner opened with a draft does by default, as
// when it decodes a token a step; each arm answers into a store of its own, so neither reads the other's answers.
func TestFlashNextSpeculationHoldsIFEval(t *testing.T) {
	// Serial: both arms share the one card and the resident model.
	cudatest.Require(t)
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	model, draft := roots.ResolveModelPath(flashNextModel), filepath.Join(roots.References(), flashNextDraft)
	for _, path := range []string{model, draft} {
		if _, err := os.Stat(path); err != nil {
			t.Skipf("Flash-Next's held-out check needs %s: %v", path, err)
		}
	}
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	authorities := ExactAuthorities{
		ModelDefinition: planID(t, artifact.KindModelDefinition, "flash-next"), RuntimeRecipe: planID(t, artifact.KindRecipe, "flash-next"),
		CodeCommit: planTestCommit, Environment: planID(t, artifact.KindEvidence, "device"),
		Execution: ExecutionPolicy{Lifecycle: LifecycleResident, Prompting: PromptingChatTemplate},
	}
	suites, _, err := DeriveStoreSuiteFamily(t.Context(), store, authorities, speculationHeldOut)
	if closeErr := store.Close(); err != nil || closeErr != nil || len(suites) != 1 {
		t.Fatalf("the store derived %d suites for %s: %v %v", len(suites), speculationHeldOut, err, closeErr)
	}
	suite := suites[0]
	runner, err := modelcli.OpenRunner(t.Context(), roots.Store, model, inference.OpenOptions{Draft: draft})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := runner.Close(); err != nil {
			t.Error(err)
		}
	})
	if !runner.SpeculatesByDefault() {
		t.Fatal("Flash-Next opened with its draft does not speculate by default")
	}
	// answer scores the part under one arm and returns its headline metric's value.
	answer := func(runtime *speculationRuntime) float64 {
		t.Helper()
		answers, err := overgodb.Open(t.TempDir())
		if err != nil {
			t.Fatal(err)
		}
		defer answers.Close()
		publishPlanFixtureAuthorities(t, answers, suite.plan)
		started := time.Now()
		result, err := suite.execute(t.Context(), answers, runtime)
		if err != nil {
			t.Fatal(err)
		}
		metrics := map[string]float64{}
		for _, metric := range result.Metrics {
			metrics[metric.Name] = metric.Value
		}
		name, value, ok := HeadlineMetric(metrics)
		if !ok {
			t.Fatalf("the part reports no headline metric: %v", metrics)
		}
		elapsed := time.Since(started)
		t.Logf("plain %t: %s %.4f; %d tokens in %s (%.1f a second), drafts accepted %d of %d", runtime.plain, name, value,
			runtime.generated, elapsed.Round(time.Second), float64(runtime.generated)/elapsed.Seconds(), runtime.accepted, runtime.drafted)
		return value
	}
	plain := answer(&speculationRuntime{Runner: runner, plain: true})
	speculative := answer(&speculationRuntime{Runner: runner})
	if cases := float64(suite.descriptor.Cases); plain-speculative > speculationDrop {
		t.Fatalf("speculation follows %.0f of %.0f held-out instructions' prompts strictly, plain decoding %.0f", speculative*cases, cases, plain*cases)
	}
}
