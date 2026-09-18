package main

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/mediacapability"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
	"overgo/internal/tokenizer"
)

type deferredExactFixture struct {
	text          string
	calls, closes int
	closeErr      error
}

func (fixture *deferredExactFixture) Generate(_ context.Context, _ string, options inference.GenerateOptions) ([]tokenizer.TokenID, string, error) {
	fixture.calls++
	options.OnPromptEvaluated(inference.PromptEvaluation{Tokens: 1})
	if err := options.OnToken(inference.TokenEvent{ID: 1, Piece: fixture.text}); err != nil {
		return nil, "", err
	}
	return []tokenizer.TokenID{0, 1}, "", nil
}

func (fixture *deferredExactFixture) Close() error { fixture.closes++; return fixture.closeErr }

func TestDeferredExactRuntime(t *testing.T) {
	t.Run("terminal authority", testExactTerminalReplay)
	for _, outcome := range []string{"pass", "mismatch", "partial"} {
		t.Run(outcome, func(t *testing.T) {
			root := t.TempDir()
			store, err := overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { store.Close() })
			suite := evaluation.ExactSuite{Schema: "test/v1", Source: "deferred-acquisition", Cases: []evaluation.ExactCase{
				{Name: "first", Prompt: "first", Text: "ok", MaxTokens: 1, PromptTokens: 1, GeneratedTokens: 1},
				{Name: "second", Prompt: "second", Text: "ok", MaxTokens: 1, PromptTokens: 1, GeneratedTokens: 1},
			}}
			exact, err := evaluation.CompileExact(suite)
			if err != nil {
				t.Fatal(err)
			}
			authority := evaluation.ExactAuthorities{
				ModelDefinition: testutil.ArtifactID(t, artifact.KindModelDefinition, "model"),
				RuntimeRecipe:   testutil.ArtifactID(t, artifact.KindRecipe, "runtime"),
				Environment:     testutil.ArtifactID(t, artifact.KindEvidence, "environment"),
				CodeCommit:      strings.Repeat("a", 40), Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
			}
			bind := func(authority evaluation.ExactAuthorities) evaluation.Plan {
				t.Helper()
				plan, err := evaluation.BindExact(exact, authority)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := store.Commit(t.Context(), artifact.Batch{Key: "fixture/" + plan.Identity().String(), Artifacts: []artifact.Descriptor{{ID: authority.ModelDefinition}, {ID: authority.RuntimeRecipe}, {ID: authority.Environment}}}); err != nil && !errors.Is(err, artifact.ErrNoChange) {
					t.Fatal(err)
				}
				return plan
			}
			plan := bind(authority)
			fixture := &deferredExactFixture{text: "ok"}
			if outcome == "mismatch" {
				fixture.text = "no"
			}
			opens := 0
			makeRuntime := func() *deferredExactRuntime {
				return &deferredExactRuntime{open: func(context.Context) (exactRuntime, error) { opens++; return fixture, nil }}
			}
			runtime := makeRuntime()
			interrupted := errors.New("publication interrupted")
			var observe func(evaluation.ExactResult) error
			if outcome == "partial" {
				observe = func(evaluation.ExactResult) error { return interrupted }
			}
			first, err := evaluation.EvaluateExactSharded(t.Context(), store, runtime, exact, plan, observe)
			if outcome == "pass" && err != nil || outcome == "partial" && !errors.Is(err, interrupted) || outcome == "mismatch" && (err == nil || !strings.Contains(err.Error(), "exact mismatch")) {
				t.Fatalf("first result=%s error=%v", first, err)
			}
			if err := runtime.Close(); err != nil {
				t.Fatal(err)
			}
			if opens != 1 || fixture.closes != 1 {
				t.Fatalf("first lifetime opens=%d closes=%d", opens, fixture.closes)
			}
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			store, err = overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			beforeCalls, beforeOpens := fixture.calls, opens
			runtime = makeRuntime()
			replayed, err := evaluation.EvaluateExactSharded(t.Context(), store, runtime, exact, plan, nil)
			if outcome == "mismatch" {
				if err == nil || !strings.Contains(err.Error(), "exact mismatch") {
					t.Fatalf("retained mismatch became success: %v", err)
				}
			} else if err != nil {
				t.Fatal(err)
			}
			if outcome == "partial" {
				if fixture.calls != beforeCalls+1 || opens != beforeOpens+1 {
					t.Fatal("partial resume repeated completed acquisition")
				}
			} else if first != replayed || fixture.calls != beforeCalls || opens != beforeOpens {
				t.Fatal("complete replay acquired a runtime or changed result")
			}
			if err := runtime.Close(); err != nil {
				t.Fatal(err)
			}
			for _, change := range []string{"source", "model", "environment", "protocol"} {
				changed := authority
				switch change {
				case "source":
					changed.CodeCommit = strings.Repeat("b", 40)
				case "model":
					changed.ModelDefinition = testutil.ArtifactID(t, artifact.KindModelDefinition, "other model")
				case "environment":
					changed.Environment = testutil.ArtifactID(t, artifact.KindEvidence, "other environment")
				case "protocol":
					changed.Execution.Lifecycle = evaluation.LifecycleIsolated
				}
				changedPlan := bind(changed)
				missing := errors.New("new acquisition required")
				attempts := 0
				changedRuntime := &deferredExactRuntime{open: func(context.Context) (exactRuntime, error) { attempts++; return nil, missing }}
				if _, err := evaluation.EvaluateExactSharded(t.Context(), store, changedRuntime, exact, changedPlan, nil); !errors.Is(err, missing) || attempts != 1 {
					t.Fatalf("%s reused foreign result: %v attempts=%d", change, err, attempts)
				}
				if err := changedRuntime.Close(); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
	t.Run("cancellation and release", func(t *testing.T) {
		cleanup := errors.New("cleanup failed")
		released, opened := 0, 0
		runtime := &deferredExactRuntime{open: func(context.Context) (exactRuntime, error) { opened++; return nil, errors.New("unreachable") }, release: func() error { released++; return cleanup }}
		ctx, cancel := context.WithCancelCause(t.Context())
		cancel(context.Canceled)
		if _, _, err := runtime.Generate(ctx, "", inference.GenerateOptions{}); !errors.Is(err, context.Canceled) || opened != 0 {
			t.Fatalf("canceled acquisition: %v", err)
		}
		if err := runtime.Close(); !errors.Is(err, cleanup) {
			t.Fatal("cleanup failure lost", err)
		}
		if err := runtime.Close(); err != nil || released != 1 {
			t.Fatal("cleanup repeated", err)
		}
		if _, _, err := runtime.Generate(t.Context(), "", inference.GenerateOptions{}); err == nil || opened != 0 {
			t.Fatal("closed runtime reopened")
		}
	})
}

func testExactTerminalReplay(t *testing.T) {
	for _, outcome := range []string{"pass", "mismatch", "cleanup failure", "unpublished", "missing ledger"} {
		t.Run(outcome, func(t *testing.T) {
			root := t.TempDir()
			store, err := overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { store.Close() })
			model := testutil.ArtifactID(t, artifact.KindModel, "terminal fixture")
			modelDefinition := testutil.ArtifactID(t, artifact.KindModelDefinition, "terminal model")
			testutil.PublishArtifact(t, store, model)
			testutil.PublishArtifact(t, store, modelDefinition)
			definition, err := modelrecipe.CapabilityDefinition(recipe.TaskSeq2Seq, model)
			if err != nil {
				t.Fatal(err)
			}
			if _, _, err := modelrecipe.PublishCandidate(t.Context(), store, "fixture/terminal", definition); err != nil {
				t.Fatal(err)
			}
			suite := evaluation.ExactSuite{Schema: "test/v1", Source: "terminal replay", Cases: []evaluation.ExactCase{
				{Name: "case", Prompt: "prompt", Text: "ok", MaxTokens: 1, PromptTokens: 1, GeneratedTokens: 1},
			}}
			exact, err := evaluation.CompileExact(suite)
			if err != nil {
				t.Fatal(err)
			}
			environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
			if err != nil {
				t.Fatal(err)
			}
			revision := strings.Repeat("a", 40)
			plan, err := evaluation.BindExact(exact, evaluation.ExactAuthorities{
				ModelDefinition: modelDefinition, RuntimeRecipe: definition.ID, Environment: environment.ID,
				CodeCommit: revision, Execution: evaluation.ExecutionPolicy{Lifecycle: evaluation.LifecycleResident},
			})
			if err != nil {
				t.Fatal(err)
			}
			fixture := &deferredExactFixture{text: "ok"}
			if outcome == "mismatch" {
				fixture.text = "no"
			}
			if outcome == "cleanup failure" {
				fixture.closeErr = errors.New("device cleanup failed")
			}
			opens, releases := 0, 0
			makeRuntime := func() *deferredExactRuntime {
				return &deferredExactRuntime{
					open:    func(context.Context) (exactRuntime, error) { opens++; return fixture, nil },
					release: func() error { releases++; return nil },
				}
			}
			if outcome == "unpublished" {
				content, err := environment.Content()
				if err != nil {
					t.Fatal(err)
				}
				batch, err := artifact.NewDocumentBatch("fixture/environment", []artifact.Content{content}, nil, nil)
				if err != nil {
					t.Fatal(err)
				}
				if _, err := store.Commit(t.Context(), batch); err != nil {
					t.Fatal(err)
				}
				if _, err := evaluation.EvaluateExactSharded(t.Context(), store, fixture, exact, plan, nil); err != nil {
					t.Fatal(err)
				}
				fixture.calls = 0
			}
			if outcome == "missing ledger" {
				report := testutil.ArtifactID(t, artifact.KindEvaluation, "absent report")
				if _, err := modelintake.PublishVerification(t.Context(), store, definition, revision, 1, "cuda:0", "cuda",
					"contract=exact;plan="+plan.Identity().String()+";report="+report.String()+";cases=1"); err != nil {
					t.Fatal(err)
				}
			}
			err = verifyExactRuntime(t.Context(), store, definition, revision, suite, exact, modelDefinition, makeRuntime())
			wantFailure := outcome == "mismatch" || outcome == "cleanup failure" || outcome == "missing ledger"
			if (err != nil) != wantFailure {
				t.Fatalf("initial verification: %v", err)
			}
			if outcome == "missing ledger" {
				if opens != 0 || fixture.calls != 0 || releases != 1 {
					t.Fatal("terminal contradiction acquired a runtime")
				}
				return
			}
			wantCalls := 1
			if outcome == "unpublished" {
				wantCalls = 0
			}
			if opens != 1 || fixture.closes != 1 || fixture.calls != wantCalls {
				t.Fatalf("initial opens=%d closes=%d calls=%d", opens, fixture.closes, fixture.calls)
			}
			retained, report, err := retainedExactVerification(t.Context(), store, definition.ID, revision, environment.ID, plan.Identity())
			if err != nil || retained.Gate.ID == (artifact.ID{}) {
				t.Fatal("terminal record missing", err)
			}
			head, sequence := store.Head()
			if err := store.Close(); err != nil {
				t.Fatal(err)
			}
			store, err = overgodb.Open(root)
			if err != nil {
				t.Fatal(err)
			}
			err = verifyExactRuntime(t.Context(), store, definition, revision, suite, exact, modelDefinition, makeRuntime())
			if (err != nil) != wantFailure {
				t.Fatalf("replayed verification: %v", err)
			}
			if outcome == "cleanup failure" && !strings.Contains(err.Error(), fixture.closeErr.Error()) {
				t.Fatal("cleanup failure lost", err)
			}
			if opens != 1 || fixture.closes != 1 || fixture.calls != wantCalls || releases != 1 {
				t.Fatal("replay repeated acquisition or failed to release")
			}
			if after, count := store.Head(); after != head || count != sequence {
				t.Fatal("replay republished evidence")
			}
			again, replayReport, err := retainedExactVerification(t.Context(), store, definition.ID, revision, environment.ID, plan.Identity())
			if err != nil || again.Gate.ID != retained.Gate.ID || again.Run.ID != retained.Run.ID || replayReport != report {
				t.Fatal("terminal identity changed", err)
			}
			if _, err := modelintake.PublishFailure(t.Context(), store, definition, revision, 1, "cuda:0", "cuda",
				"contract=exact; plan="+plan.Identity().String()+";report="+report.String()+"; contradictory result", "exact-mismatch"); err != nil {
				t.Fatal(err)
			}
			if _, _, err := retainedExactVerification(t.Context(), store, definition.ID, revision, environment.ID, plan.Identity()); err == nil {
				t.Fatal("conflicting terminal outcomes accepted")
			}
		})
	}
}

func TestProjectionVerificationRequiresExecution(t *testing.T) {
	err := verifyCapability(t.TempDir(), "unused", recipe.TaskProjection, mediacapability.Capability{}, "")
	if err == nil || !strings.Contains(err.Error(), "executor") {
		t.Fatalf("compile-only projection reached verification: %v", err)
	}
	capabilities := projector.SessionCapabilities{Image: true, MultiImage: true, Audio: true, Video: true, MediaHistory: true}
	file := projectionFile{Kind: "image", Path: "fixture.png", Artifact: testutil.ArtifactID(t, artifact.KindFile, "image")}
	audio := projectionFile{Kind: "audio-f32le", Path: "fixture.f32", Artifact: testutil.ArtifactID(t, artifact.KindFile, "audio")}
	inputs := []projectionInput{
		{Kind: "image", Files: []projectionFile{file}, Text: []string{"", "describe"}},
		{Kind: "images", Files: []projectionFile{file, file}, Text: []string{"", "", "count"}},
		{Kind: "audio", Files: []projectionFile{audio}, Text: []string{"", "transcribe"}},
		{Kind: "video", Files: []projectionFile{file, file}, Text: []string{"", "describe"}, FPS: 1},
		{Kind: "mixed", Files: []projectionFile{file, audio}, Text: []string{"", "", "transcribe"}},
	}
	encode := func(input projectionInput) string {
		data, err := json.Marshal(input)
		if err != nil {
			t.Fatal(err)
		}
		return string(data)
	}
	var suite evaluation.ExactSuite
	for _, input := range inputs {
		suite.Cases = append(suite.Cases, evaluation.ExactCase{Name: input.Kind, Prompt: encode(input)})
	}
	if err := requireProjectionCoverage(suite, capabilities); err != nil {
		t.Fatal(err)
	}
	for index, input := range inputs {
		t.Run(input.Kind, func(t *testing.T) {
			partial := suite
			partial.Cases = slices.Delete(slices.Clone(suite.Cases), index, index+1)
			if err := requireProjectionCoverage(partial, capabilities); err == nil {
				t.Fatal("missing declared modality accepted")
			}
			input.Files = nil
			if _, err := decodeProjectionInput(encode(input), capabilities); err == nil {
				t.Fatal("empty fixture accepted")
			}
		})
	}
	for _, invalid := range []projectionInput{
		{Kind: "unknown", Files: []projectionFile{file}, Text: []string{"", "describe"}},
		{Kind: "image", Files: []projectionFile{audio}, Text: []string{"", "describe"}},
		{Kind: "images", Files: []projectionFile{file}, Text: []string{"", "describe"}},
		{Kind: "video", Files: []projectionFile{file, file}, Text: []string{"", "describe"}},
		{Kind: "mixed", Files: []projectionFile{file, audio}, Text: []string{"", "describe"}},
		{Kind: "image", Files: []projectionFile{{Kind: "image", Path: "unbound", Artifact: testutil.ArtifactID(t, artifact.KindOutput, "wrong-kind")}}, Text: []string{"", "describe"}},
	} {
		if _, err := decodeProjectionInput(encode(invalid), capabilities); err == nil {
			t.Fatalf("invalid input accepted: %+v", invalid)
		}
	}
	// These fail before any projector or language execution; the embedded nil
	// session makes accidental execution panic instead of manufacturing a pass.
	runtime := &projectedExactRuntime{projection: projectionCapabilitiesOnly{capabilities: capabilities}}
	file.Path = filepath.Join(t.TempDir(), "fixture.png")
	input := projectionInput{Kind: "image", Files: []projectionFile{file}, Text: []string{"", "describe"}}
	if _, _, err := runtime.Generate(t.Context(), encode(input), inference.GenerateOptions{}); err == nil {
		t.Fatal("missing fixture accepted")
	}
	if err := os.WriteFile(file.Path, []byte("changed bytes"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, _, err := runtime.Generate(t.Context(), encode(input), inference.GenerateOptions{}); err == nil || !strings.Contains(err.Error(), "differ") {
		t.Fatalf("changed fixture accepted: %v", err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	cancel(context.Canceled)
	if _, _, err := runtime.Generate(ctx, encode(input), inference.GenerateOptions{}); !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation lost: %v", err)
	}
}

type projectionCapabilitiesOnly struct {
	projector.Session
	capabilities projector.SessionCapabilities
}

func (session projectionCapabilitiesOnly) Capabilities() projector.SessionCapabilities {
	return session.capabilities
}
