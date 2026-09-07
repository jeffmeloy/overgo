package server

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"slices"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/clioptions"
	"overgo/internal/cuda/device"
	"overgo/internal/cuda/driver"
	cudatest "overgo/internal/cuda/testutil"
	"overgo/internal/dataroot"
	"overgo/internal/inference"
	"overgo/internal/modelintake"
	"overgo/internal/modelrecipe"
	"overgo/internal/overgodb"
	"overgo/internal/projector"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

// TestE4BMediaResourceRecovery exercises real request preparation, projection,
// interrupted generation and complete native-matching recovery. The first full
// six-mode cycle sets the retention ceiling; neither subsequent cycles nor
// reloads may grow it. The separate 32768-token text guard remains required.
func TestE4BMediaResourceRecovery(t *testing.T) {
	cudatest.Require(t)
	startedAll := time.Now()
	publish := os.Getenv("OVERGO_E4B_PUBLISH_RESOURCES") == "1"
	var revision string
	if publish {
		var err error
		revision, err = runrecord.VerifyingCommit(testutil.RepoRoot(t))
		if err != nil {
			t.Fatal(err)
		}
	}
	environment, err := runrecord.CurrentEnvironment("cuda:0", "cuda")
	if err != nil {
		t.Fatal(err)
	}
	var observations []runrecord.ServingObservation
	var steps []runrecord.GateStep
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeoutCause(t.Context(), 11*time.Minute, errors.New("E4B media resource budget exhausted"))
	defer cancel()
	store, err := overgodb.OpenReadOnly(roots.Store)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	var paths []string
	for _, input := range []struct {
		id   string
		kind artifact.LocationKind
	}{
		{"tensor-set:sha256:cd4ada4703c2b76a84a10da94f09b9199b6d4dad7e3dcabbe79d8cee745f4501", artifact.LocationFile},
		{"tensor-set:sha256:185786ec6d77c31f87e6ebdcf8a0d095dbb7175999122229f8e82dcdad25004e", artifact.LocationFile},
		{"dataset:sha256:39b4aa4872600afff4be0633dc066adfe5ae36ac17ade4a6ac4a16e26852b2e3", artifact.LocationDirectory},
	} {
		id, err := artifact.ParseID(input.id)
		if err != nil {
			t.Fatal(err)
		}
		path, err := artifact.AvailablePath(ctx, store, id, input.kind)
		if err != nil {
			t.Fatal(err)
		}
		paths = append(paths, path)
	}
	candidate, err := modelintake.PrepareProjectionCandidate(ctx, store, paths[0], paths[1])
	if err != nil {
		t.Fatal(err)
	}
	cases, oracleBytes := prepareE4BServingCases(t, ctx, paths[2])
	loaded, err := modelrecipe.ResolveActiveGGUF(ctx, store, paths[0])
	if err != nil {
		t.Fatal(err)
	}
	identity, err := loaded.Identity()
	if err != nil {
		t.Fatal(err)
	}
	steps = append(steps, runrecord.GateStep{Name: "media-resource-contract", Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: uint64(time.Since(startedAll)), Evidence: fmt.Sprintf("contract=e4b-media-resources-v1;oracle_sha256=%x;model_definition=%s;modes=6;reloads=3;cycles=3;actions=5", sha256.Sum256(oracleBytes), identity.Definition)})
	observer, err := device.New(0)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := observer.Close(); err != nil {
			t.Error(err)
		}
	}()
	freeDevice := func() uint64 {
		t.Helper()
		var free uint64
		if err := observer.Do(ctx, func(state *device.State) error {
			var err error
			free, _, err = state.Driver.MemInfo()
			return err
		}); err != nil {
			t.Fatal(err)
		}
		return free
	}
	var ceiling driver.MemoryStats
	for reload := range 3 {
		t.Run(fmt.Sprintf("reload-%d", reload), func(t *testing.T) {
			runner, err := clioptions.OpenRunner(ctx, roots.Store, paths[0], inference.OpenOptions{})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := runner.Close(); err != nil {
					t.Error(err)
				}
			}()
			projection, err := projector.OpenSession(ctx, paths[1], projector.OpenOptions{CUDA: true, MediaPreprocess: candidate.Processor})
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := projection.Close(); err != nil {
					t.Error(err)
				}
			}()
			trace := &e4bServingTrace{Runner: runner}
			handler, err := New(Config{RuntimePolicy: runner.RuntimePolicy(), MaxConcurrent: 1, MaxTokens: 128, ImageProjector: projection, AudioProjector: projection, VideoFPS: 1, VideoMaxFrames: 2}, trace)
			if err != nil {
				t.Fatal(err)
			}
			defer func() {
				if err := handler.Close(); err != nil {
					t.Error(err)
				}
			}()
			memory := func() (driver.MemoryStats, driver.MemoryStats) {
				t.Helper()
				language, err := runner.DeviceMemoryStats(ctx)
				if err != nil {
					t.Fatal(err)
				}
				media, err := projection.DeviceMemoryStats(ctx, false)
				if err != nil {
					t.Fatal(err)
				}
				return language, media
			}
			var projectionPeak uint64
			var phaseSample runrecord.ServingHardwareSample
			var requestStarted time.Time
			var currentCycle, action int
			trace.beforeGenerate = func(callCtx context.Context) error {
				language, err := runner.DeviceMemoryStats(callCtx)
				if err != nil {
					return err
				}
				media, err := projection.DeviceMemoryStats(callCtx, false)
				if err != nil {
					return err
				}
				// Projection finishes before language execution; language live bytes
				// remain fixed throughout this phase. This is not a sum of two peaks.
				projectionPeak = language.CurrentBytes + media.PeakBytes
				return nil
			}
			trace.afterPrompt = func(callCtx context.Context) error {
				language, err := runner.DeviceMemoryStats(callCtx)
				if err != nil {
					return err
				}
				media, err := projection.DeviceMemoryStats(callCtx, false)
				if err != nil {
					return err
				}
				phaseSample = runrecord.ServingHardwareSample{Stage: runrecord.ServingHardwarePrefill, ElapsedNS: uint64(time.Since(requestStarted)), DeviceCurrentBytes: language.CurrentBytes + media.CurrentBytes, DevicePeakBytes: max(projectionPeak, language.PeakBytes+media.CurrentBytes), DeviceAllocations: language.Allocations + media.Allocations}
				return nil
			}
			request := func(test e4bServingCase, interrupt string) {
				t.Helper()
				if err := runner.ResetDeviceMemoryPeak(ctx); err != nil {
					t.Fatal(err)
				}
				if _, err := projection.DeviceMemoryStats(ctx, true); err != nil {
					t.Fatal(err)
				}
				projectionPeak, trace.generationErr, trace.ids, trace.observationErr = 0, nil, nil, nil
				phaseSample = runrecord.ServingHardwareSample{}
				callCtx, stop := context.WithCancelCause(ctx)
				defer stop(context.Canceled)
				injected := errors.New("injected E4B media consumer failure")
				callbacks := 0
				trace.onToken = nil
				if interrupt != "" {
					trace.onToken = func(inference.TokenEvent) error {
						callbacks++
						if interrupt == "cancel" {
							stop(context.Canceled)
							return nil
						}
						return injected
					}
				}
				encoded := encodeE4BServingRequest(t, runner, test, "/v1/chat/completions")
				response := httptest.NewRecorder()
				beforeLanguage, beforeMedia := memory()
				started := time.Now()
				requestStarted = started
				handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(encoded)).WithContext(callCtx))
				if trace.observationErr != nil || phaseSample.Stage != runrecord.ServingHardwarePrefill {
					t.Fatalf("missing prefill observation: %v", trace.observationErr)
				}
				if !slices.Equal(trace.ids, test.promptIDs) {
					t.Fatalf("%s prompt differs from native oracle", test.name)
				}
				actual := ""
				if interrupt != "" {
					want := injected
					if interrupt == "cancel" {
						want = context.Canceled
					}
					if callbacks == 0 || !errors.Is(trace.generationErr, want) {
						t.Fatalf("%s %s did not interrupt actual generation: callbacks=%d err=%v", test.name, interrupt, callbacks, trace.generationErr)
					}
				} else {
					var result struct {
						Choices []struct {
							Message      struct{ Content string }
							FinishReason string `json:"finish_reason"`
						}
					}
					if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
						t.Fatal(err)
					}
					if response.Code != http.StatusOK || len(result.Choices) != 1 || result.Choices[0].FinishReason != "stop" || !slices.Contains(test.outputs, result.Choices[0].Message.Content) {
						t.Fatalf("%s recovery differs from complete native output: HTTP %d %s", test.name, response.Code, response.Body.String())
					}
					actual = result.Choices[0].Message.Content
				}
				language, media := memory()
				peak := max(projectionPeak, language.PeakBytes+media.CurrentBytes)
				if projectionPeak == 0 || peak < language.CurrentBytes+media.CurrentBytes {
					t.Fatal("missing or inconsistent device peak")
				}
				duration := uint64(time.Since(started))
				outcome, failure := executionOutcome(trace.generationErr)
				observation, err := runrecord.NewServingObservation(runrecord.ServingObservation{
					Model: runner.ModelID(), Recipe: candidate.Definition.ID, Environment: environment.ID, Task: recipe.TaskInference,
					Outcome: outcome, Failure: failure, StartedUnixNS: started.UnixNano(), MeasuredNS: duration,
					SessionReused: currentCycle != 0 || action != 0 || test.name != cases[0].name,
					Usage:         runrecord.ServingUsage{InputTokens: uint64(len(trace.ids)), InputBytes: uint64(len(encoded)), OutputBytes: uint64(response.Body.Len())},
					Resources:     runrecord.ServingResources{PeakDeviceBytes: peak},
					Hardware: []runrecord.ServingHardwareSample{
						{Stage: runrecord.ServingHardwareStart, DeviceCurrentBytes: beforeLanguage.CurrentBytes + beforeMedia.CurrentBytes, DevicePeakBytes: beforeLanguage.CurrentBytes + beforeMedia.CurrentBytes, DeviceAllocations: beforeLanguage.Allocations + beforeMedia.Allocations},
						phaseSample,
						{Stage: runrecord.ServingHardwareFinish, ElapsedNS: duration, DeviceCurrentBytes: language.CurrentBytes + media.CurrentBytes, DevicePeakBytes: peak, DeviceAllocations: language.Allocations + media.Allocations},
					},
				})
				if err != nil {
					t.Fatal(err)
				}
				observations = append(observations, observation)
				promptBytes, err := json.Marshal(trace.ids)
				if err != nil {
					t.Fatal(err)
				}
				evidence, err := json.Marshal(map[string]any{"observation": observation.ID, "output_sha256": fmt.Sprintf("%x", sha256.Sum256([]byte(actual))), "input_sha256": fmt.Sprintf("%x", sha256.Sum256(encoded)), "prompt_sha256": fmt.Sprintf("%x", sha256.Sum256(promptBytes)), "mode": test.name, "reload": reload, "cycle": currentCycle, "action": action})
				if err != nil {
					t.Fatal(err)
				}
				steps = append(steps, runrecord.GateStep{Name: fmt.Sprintf("media-r%d-c%d-%s-a%d", reload, currentCycle, test.name, action), Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: duration, Evidence: string(evidence)})
				action++
				t.Logf("mode=%s interrupt=%q wall=%s peak=%d retained=%d allocations=%d", test.name, interrupt, time.Since(started), peak, language.CurrentBytes+media.CurrentBytes, language.Allocations+media.Allocations)
			}
			for cycle := range 3 {
				currentCycle = cycle
				for _, test := range cases {
					action = 0
					request(test, "")
					request(test, "cancel")
					request(test, "")
					request(test, "consumer")
					request(test, "")
				}
				language, media := memory()
				retained := driver.MemoryStats{CurrentBytes: language.CurrentBytes + media.CurrentBytes, Allocations: language.Allocations + media.Allocations}
				if reload == 0 && cycle == 0 {
					ceiling = retained
				}
				if retained.CurrentBytes > ceiling.CurrentBytes || retained.Allocations > ceiling.Allocations {
					t.Fatalf("six-mode retention grew: baseline=%+v current=%+v", ceiling, retained)
				}
				t.Logf("reload=%d cycle=%d complete six-mode retention=%d allocations=%d", reload, cycle, retained.CurrentBytes, retained.Allocations)
			}
			language, media := memory()
			owned := language.CurrentBytes + media.CurrentBytes
			releaseStarted := time.Now()
			before := freeDevice()
			if err := errors.Join(handler.Close(), projection.Close(), runner.Close()); err != nil {
				t.Fatal(err)
			}
			after := freeDevice()
			if after < before || after-before < owned {
				t.Fatalf("close failed to release allocations: free %d -> %d owned=%d", before, after, owned)
			}
			if _, err := projection.DeviceMemoryStats(ctx, false); err == nil {
				t.Fatal("closed projector accepted device observation")
			}
			if _, _, err := runner.Generate(ctx, "closed", inference.GenerateOptions{MaxNewTokens: 1}); err == nil {
				t.Fatal("closed runner accepted generation")
			}
			t.Logf("reload=%d released=%d owned=%d", reload, after-before, owned)
			release, err := json.Marshal(map[string]uint64{"before": before, "after": after, "owned": owned})
			if err != nil {
				t.Fatal(err)
			}
			steps = append(steps, runrecord.GateStep{Name: fmt.Sprintf("media-release-%d", reload), Phase: runrecord.PhaseTest, Outcome: runrecord.StepSucceeded, DurationNS: uint64(time.Since(releaseStarted)), Evidence: string(release)})
		})
		if t.Failed() {
			return
		}
	}
	if publish && !t.Failed() {
		current, err := runrecord.VerifyingCommit(testutil.RepoRoot(t))
		if err != nil || current != revision {
			t.Fatalf("resource producer source changed: %v", err)
		}
		writer, err := overgodb.Open(roots.Store)
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Close()
		if err := modelintake.RegisterProjectionCandidate(ctx, writer, candidate); err != nil {
			t.Fatal(err)
		}
		record, err := runrecord.NewGateRecord(candidate.Definition.ID, environment.ID, revision, runrecord.OutcomeSucceeded, "", uint64(time.Since(startedAll)), steps)
		if err != nil {
			t.Fatal(err)
		}
		batch, err := record.Batch("validation/media-resources/" + record.Result.ID.String())
		if err != nil {
			t.Fatal(err)
		}
		content, err := environment.Content()
		if err != nil {
			t.Fatal(err)
		}
		batch.Contents = append(batch.Contents, content)
		for _, observation := range observations {
			content, err := observation.Content()
			if err != nil {
				t.Fatal(err)
			}
			batch.Contents = append(batch.Contents, content)
			batch.Lineage = append(batch.Lineage, observation.Lineage()...)
			batch.Lineage = append(batch.Lineage, artifact.DependencyLineage(record.Result.ID, observation.ID)...)
		}
		if _, err := artifact.CommitBatch(ctx, writer, batch); err != nil {
			t.Fatal(err)
		}
		t.Logf("published media resource proof: gate=%s run=%s recipe=%s observations=%d; no quality or promotion credit", record.Result.ID, record.Run.ID, candidate.Definition.ID, len(observations))
	}

	t.Log("6 modes x 3 reloads x 3 cycles: 162 complete native-matching generations, 54 cancellations, 54 consumer errors; projector and language allocation windows, stable retention and verified release; producer check only, no corpus-quality or promotion credit")
}
