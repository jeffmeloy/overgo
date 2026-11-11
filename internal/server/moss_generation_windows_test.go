//go:build windows

package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"slices"
	"sync"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/capabilityruntime"
	"overgo/internal/latentimage"
	"overgo/internal/libraryintake"
	"overgo/internal/mediacapability"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/strictjson"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

// mossGenerationFixture binds real source registration to hermetic execution and activation.
//   - Model execution is hermetic here and is compared with native fixtures in latentimage.
//   - Functional validation is exercised with the native compact model in libraryintake.
type mossGenerationFixture struct {
	handler   *Handler
	workspace *storeGenerationWorkspace
	store     *overgodb.Store
	root      string
	mu        sync.Mutex
	inputs    []latentimage.Request
}

func (fixture *mossGenerationFixture) requests() []latentimage.Request {
	fixture.mu.Lock()
	defer fixture.mu.Unlock()
	return append([]latentimage.Request(nil), fixture.inputs...)
}

func newMOSSGenerationFixture(t *testing.T, onRequest func(context.Context, artifact.Repository, modelrecipe.CapabilityEvidenceSelection, latentimage.Request) error) *mossGenerationFixture {
	t.Helper()
	root := testutil.ExtractTarGzip(t, testutil.FixturePath(t, "moss", "generator-compact.tar.gz"), "150c4263057d72476ef4399a442c2b7dbadcb955670232fc2ea869c71f07c5db")
	store := overgodbtest.Open(t, t.TempDir())
	fixture := &mossGenerationFixture{store: store, root: root}
	image := tinyPNG(t)
	inputContract := artifact.JSONContract(artifact.KindFile, "overgo.test-moss-request.v1")
	catalog := map[recipe.Task]mediacapability.Capability{recipe.TaskImageGen: {
		Execute: func(ctx context.Context, repository artifact.Repository, path string, execution modelrecipe.CapabilityEvidenceSelection, raw string) (any, error) {
			var executed latentimage.Request
			if path != root || len(execution.Program.Stages()) != 3 {
				t.Error("component execution lost", path)
			}
			if err := strictjson.DecodeBytes([]byte(raw), &executed); err != nil {
				return nil, err
			}
			fixture.mu.Lock()
			fixture.inputs = append(fixture.inputs, executed)
			fixture.mu.Unlock()
			if err := latentimage.ValidateRequest(executed); err != nil {
				return nil, err
			}
			if onRequest != nil {
				if err := onRequest(ctx, repository, execution, executed); err != nil {
					return nil, err
				}
			}
			if executed.Prompt == "cancel" {
				<-ctx.Done()
				return nil, ctx.Err()
			}
			input, err := inputContract.ContentBytes([]byte(raw))
			return capabilityruntime.Measured{Output: image, Input: input}, err
		},
	}}
	intake := LibraryIntake{ModelFiles: libraryintake.ModelFiles, Register: libraryintake.Register,
		Validate: func(ctx context.Context, store *overgodb.Store, path, projector string) (operation.Request, operation.Executor, error) {
			receipt, err := libraryintake.Register(ctx, store, path, projector)
			if err != nil {
				return operation.Request{}, nil, err
			}
			definition, err := recipe.RequireDefinition(ctx, store, receipt["recipe"].(artifact.ID))
			if err != nil {
				return operation.Request{}, nil, err
			}
			return operation.Request{Task: definition.Task, Recipe: definition.ID}, func(ctx context.Context, reporter operation.Reporter) (operation.Completion, error) {
				verification, err := modelrecipetest.PublishVerification(ctx, store, "fixture/moss-http", definition.ID)
				if err == nil {
					err = modelrecipe.ActivateCapability(ctx, store, definition, verification, recipe.EvidenceVerified, "hermetic HTTP activation fixture")
				}
				return operation.Completion{Run: verification.Run}, err
			}, nil
		},
	}
	workspace := NewStoreGenerationWorkspace(store, BindGenerationCatalog(catalog, mediacapability.Controls, mediacapability.OutputContent), 16)
	handler := testcheck.Must(New(Config{Repository: store, Workflows: workspace, LibraryIntake: intake}, &fakeGenerator{}))(t)
	t.Cleanup(func() { _ = handler.Close() })
	fixture.handler, fixture.workspace = handler, workspace
	return fixture
}

// TestMOSSImageGenerationContract checks registration, activation and retained HTTP outcomes.
func TestMOSSImageGenerationContract(t *testing.T) {
	t.Parallel()
	fixture := newMOSSGenerationFixture(t, nil)
	handler, workspace, store, root := fixture.handler, fixture.workspace, fixture.store, fixture.root
	image := tinyPNG(t)
	request := func(method, path string, body any, status int, out any) {
		t.Helper()
		var raw []byte
		if body != nil {
			raw = testcheck.Must(json.Marshal(body))(t)
		}
		response := serveTestRequest(handler, method, path, string(raw))
		if response.Code != status {
			t.Fatalf("%s %s: %d %s", method, path, response.Code, response.Body)
		}
		if out != nil {
			testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), out))
		}
	}
	var receipt struct{ Recipe artifact.ID }
	request(http.MethodPost, "/library/register", map[string]any{"kind": "model", "path": root}, http.StatusOK, &receipt)
	if !receipt.Recipe.Valid() {
		t.Fatal("registered recipe missing")
	}
	t.Run("cancel-before-prepare", func(t *testing.T) {
		ctx, cancel := context.WithCancel(t.Context())
		cancel()
		called := false
		completion, err := handler.executeObservedOperation(ctx, &recordingReporter{}, recipe.TaskImageGen, receipt.Recipe, artifact.ID{}, func(context.Context, operation.Reporter) (operation.Completion, error) {
			called = true
			return operation.Completion{}, nil
		})
		if called || !errors.Is(err, context.Canceled) {
			t.Fatal("cancelled preparation executed", called, err)
		}
		run, err := runrecord.RequireExactRun(t.Context(), store, completion.Run)
		if err != nil || run.Outcome != runrecord.OutcomeCancelled {
			t.Fatal("cancelled preparation lacks durable run", run, err)
		}
	})
	listed, err := workspace.WorkflowCapabilities(t.Context(), workflowGeneration)
	// An unvalidated candidate is offered to run, its state beside it.
	if err != nil || len(listed) != 1 || listed[0].Refusal != "" || listed[0].Recipe != receipt.Recipe || listed[0].Verification != modelrecipe.StateNotVerified {
		t.Fatal("unvalidated candidate not offered", listed, err)
	}
	var submission workflowResponse
	request(http.MethodPost, "/library/validate", map[string]any{"path": root}, http.StatusAccepted, &submission)
	var status operation.Status
	request(http.MethodGet, "/operations/wait?id="+submission.Operation.String(), nil, http.StatusOK, &status)
	if status.State != operation.StateCompleted || status.Task != recipe.TaskImageGen {
		t.Fatal("validation task or completion lost", status)
	}
	listed, err = workspace.WorkflowCapabilities(t.Context(), workflowGeneration)
	if err != nil || len(listed) != 1 || listed[0].Refusal != "" || listed[0].Recipe != receipt.Recipe || listed[0].Verification != modelrecipe.StateVerified {
		t.Fatal("activation not offered", listed, err)
	}
	controls := listed[0].Controls
	if len(controls) != 9 {
		t.Fatal("unsupported controls offered", controls)
	}
	for _, c := range controls {
		if c.Name == "guidance_scale" && (!c.Advanced || c.Bounds == nil || c.Bounds.Default != 3) {
			t.Fatal("guidance declaration lost", c)
		}
		// The page offers the model's own presets beside the authors' recipe, its guidance presets once their readings admit them.
		if c.Name == "preset" && (!c.Advanced || !slices.Equal(c.Choices, []string{"comfy-tutorial"})) {
			t.Fatal("MOSS's presets are not offered", c)
		}
	}
	for _, prompt := range []string{"  blue cube\n", "<objects>\nblue cube (0, 0, 16, 16)\n</objects>"} {
		input := map[string]any{"prompt": prompt, "width": 32, "height": 48, "steps": 3, "guidance_scale": 0}
		request(http.MethodPost, "/generation/run", map[string]any{"task": recipe.TaskImageGen, "recipe": receipt.Recipe, "input": input}, http.StatusAccepted, &submission)
		request(http.MethodGet, "/operations/wait?id="+submission.Operation.String(), nil, http.StatusOK, &status)
		inputs := fixture.requests()
		executed := inputs[len(inputs)-1]
		if status.State != operation.StateCompleted || len(status.Outputs) != 1 || executed.Prompt != prompt || executed.GuidanceScale == nil || *executed.GuidanceScale != 0 {
			t.Fatal("request or output differs", status, executed)
		}
		content, present, err := artifact.ReadContent(t.Context(), store, status.Outputs[0])
		if err != nil || !present || !bytes.Equal(content.Data, image.Data) {
			t.Fatal("published PNG differs", err)
		}
	}
	request(http.MethodPost, "/generation/run", map[string]any{"task": recipe.TaskImageGen, "recipe": receipt.Recipe, "input": map[string]any{"prompt": "cube", "mu": 1}}, http.StatusBadRequest, nil)
	for _, mode := range []string{"invalid", "cancel", "retry"} {
		input := map[string]any{"prompt": "blue cube", "width": 32, "height": 48}
		if mode == "invalid" {
			input["width"] = 0
		}
		if mode == "cancel" {
			input["prompt"] = "cancel"
		}
		request(http.MethodPost, "/generation/run", map[string]any{"task": recipe.TaskImageGen, "recipe": receipt.Recipe, "input": input}, http.StatusAccepted, &submission)
		if mode == "cancel" {
			request(http.MethodPost, "/operations/cancel", map[string]any{"id": submission.Operation}, http.StatusAccepted, nil)
		}
		status = operation.Status{}
		request(http.MethodGet, "/operations/wait?id="+submission.Operation.String(), nil, http.StatusOK, &status)
		want := map[string]operation.State{"invalid": operation.StateFailed, "cancel": operation.StateCancelled, "retry": operation.StateCompleted}[mode]
		if status.State != want || mode != "retry" && len(status.Outputs) != 0 {
			t.Fatal("failure or recovery state differs", mode, status)
		}
	}
}
