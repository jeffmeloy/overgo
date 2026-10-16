package server

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/media"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/recipecontract"
	"overgo/internal/speechsynth"
	"overgo/internal/testutil"
)

type speechOutputFixture struct {
	*generationWorkspaceGenerator
	gate, cleanup       <-chan struct{}
	cancelled, released chan struct{}
	produced            atomic.Int32
	fail                bool
	content             artifact.Content
}

func newSpeechOutputFixture(t *testing.T) (*Handler, *speechOutputFixture) {
	t.Helper()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = store.Close() })
	var pcm []float32
	for index := range 6 {
		pcm = append(pcm, speechFixtureChunk(index).PCM...)
	}
	content, err := speechsynth.WAVContent(speechsynth.Audio{PCM: pcm, SampleRate: 24000, Channels: 1, Complete: true})
	if err != nil {
		t.Fatal(err)
	}
	fixture := &speechOutputFixture{generationWorkspaceGenerator: &generationWorkspaceGenerator{
		fakeGenerator: &fakeGenerator{}, repository: store, run: testutil.ArtifactID(t, artifact.KindRun, "speech-output"),
		capability: WorkflowCapability{Task: recipe.TaskSpeech, Recipe: testutil.ArtifactID(t, artifact.KindRecipe, "speech-output"), Name: "Speech fixture",
			Stages: []recipe.Stage{{Node: recipe.Node{ID: "generate", Module: "test.speech"}}}, Outputs: []recipe.Output{{Name: "audio", Data: recipe.DataAudio}},
			Controls: []workflowControl{{Name: "text", Type: workflowControlText, Required: true}},
		},
	}, cancelled: make(chan struct{}), released: make(chan struct{}), content: content}
	handler, err := New(Config{Repository: store, Workflows: fixture}, fixture)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = handler.Close() })
	return handler, fixture
}

func speechFixtureChunk(index int) speechsynth.Audio {
	pcm := make([]float32, 1920)
	for at := range pcm {
		pcm[at] = float32(index+1) / 10
	}
	pcm[1], pcm[2] = math.Float32frombits(1<<31), math.SmallestNonzeroFloat32
	return speechsynth.Audio{PCM: pcm, SampleRate: 24000, Channels: 1}
}

func (fixture *speechOutputFixture) ExecuteWorkflow(ctx context.Context, kind WorkflowKind, capability WorkflowCapability, raw json.RawMessage, reporter operation.Reporter) (operation.Completion, error) {
	if _, found, err := fixture.repository.Artifact(ctx, fixture.content.Descriptor.ID); err != nil {
		return operation.Completion{}, err
	} else if found {
		return operation.Completion{Run: fixture.run, Outputs: []artifact.ID{fixture.content.Descriptor.ID}}, nil
	}
	defer close(fixture.released)
	for index := range 6 {
		fixture.produced.Add(1)
		audio := speechFixtureChunk(index)
		err := media.EmitAudio(ctx, media.DecodedAudio{Samples: audio.PCM, Format: recipecontract.AudioFormat{SampleRate: uint64(audio.SampleRate), Channels: uint32(audio.Channels), Encoding: "pcm-f32le"}})
		if err == nil && index == 0 && fixture.gate != nil {
			select {
			case <-fixture.gate:
			case <-ctx.Done():
				err = context.Cause(ctx)
			}
		}
		if err != nil {
			close(fixture.cancelled)
			if fixture.cleanup != nil {
				<-fixture.cleanup
			}
			return operation.Completion{Run: fixture.run}, err
		}
		if fixture.fail {
			return operation.Completion{Run: fixture.run}, errors.New("speech fixture failed after its first chunk")
		}
	}
	completion, err := fixture.generationWorkspaceGenerator.ExecuteWorkflow(ctx, kind, capability, raw, reporter)
	if err != nil {
		return completion, err
	}
	_, err = fixture.repository.Commit(ctx, artifact.Batch{Key: "speech-output/wav", Contents: []artifact.Content{fixture.content}})
	completion.Outputs = []artifact.ID{fixture.content.Descriptor.ID}
	return completion, err
}

func speechOutputRequest(fixture *speechOutputFixture, native bool) *http.Request {
	path, body := "/generation/run", `{"task":"speech","recipe":"`+fixture.capability.Recipe.String()+`","input":{"text":"hello"},"request":"speech-output"}`
	if native {
		path, body = "/v1/audio/speech", `{"model":"`+fixture.capability.Recipe.String()+`","text":"hello"}`
	}
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	request.Header.Set("Accept", "text/event-stream")
	return request
}

type heldSpeechResponse struct {
	*httptest.ResponseRecorder
	entered, release chan struct{}
	receipt          chan struct{}
}

func (response *heldSpeechResponse) Write(data []byte) (int, error) {
	if response.receipt != nil && bytes.Contains(data, []byte(`"operation"`)) {
		close(response.receipt)
		response.receipt = nil
	}
	if bytes.Contains(data, []byte(`"pcm_f32le"`)) && response.entered != nil {
		close(response.entered)
		response.entered = nil
		<-response.release
	}
	return response.ResponseRecorder.Write(data)
}

func TestSpeechOutputStream(t *testing.T) {
	t.Parallel()
	t.Run("retry disconnect preserves owner", func(t *testing.T) {
		handler, fixture := newSpeechOutputFixture(t)
		entered, release := make(chan struct{}), make(chan struct{})
		owner := &heldSpeechResponse{ResponseRecorder: httptest.NewRecorder(), entered: entered, release: release}
		done := make(chan struct{})
		go func() { defer close(done); handler.ServeHTTP(owner, speechOutputRequest(fixture, false)) }()
		select {
		case <-entered:
		case <-done:
			t.Fatal("owner ended before chunk")
		}
		ctx, cancel := context.WithCancel(t.Context())
		defer cancel()
		receipt := make(chan struct{})
		retry := &heldSpeechResponse{ResponseRecorder: httptest.NewRecorder(), receipt: receipt}
		retried := make(chan struct{})
		go func() {
			defer close(retried)
			handler.ServeHTTP(retry, speechOutputRequest(fixture, false).WithContext(ctx))
		}()
		<-receipt
		cancel()
		<-retried
		select {
		case <-fixture.cancelled:
			t.Fatal("retry cancelled original execution")
		default:
		}
		if fixture.produced.Load() != 1 {
			t.Fatal("retry started another producer")
		}
		if owner.Header().Get("X-Overgo-Operation") != retry.Header().Get("X-Overgo-Operation") {
			t.Fatal("retry changed operation identity")
		}
		close(release)
		<-done
		if !strings.Contains(owner.Body.String(), "event: completed") {
			t.Fatal("owner did not complete")
		}
	})
	for _, native := range []bool{false, true} {
		name := "workflow"
		if native {
			name = "native"
		}
		t.Run(name, func(t *testing.T) {
			for _, failure := range []bool{false, true} {
				t.Run(map[bool]string{false: "complete", true: "failure"}[failure], func(t *testing.T) {
					handler, fixture := newSpeechOutputFixture(t)
					fixture.fail = failure
					entered, release := make(chan struct{}), make(chan struct{})
					response := &heldSpeechResponse{ResponseRecorder: httptest.NewRecorder(), entered: entered, release: release}
					done := make(chan struct{})
					go func() { defer close(done); handler.ServeHTTP(response, speechOutputRequest(fixture, native)) }()
					select {
					case <-entered:
					case <-done:
						t.Fatalf("no chunk before completion: %d %s", response.Code, response.Body.String())
					}
					if fixture.produced.Load() != 1 {
						t.Fatal("producer ran ahead of blocked write")
					}
					if _, found, err := fixture.repository.Artifact(t.Context(), fixture.content.Descriptor.ID); err != nil || found {
						t.Fatalf("partial recording published: %v %v", found, err)
					}
					close(release)
					<-done
					if response.Code != http.StatusOK || response.Header().Get("Content-Type") != "text/event-stream" {
						t.Fatalf("status=%d type=%s", response.Code, response.Header().Get("Content-Type"))
					}
					var samples []float32
					chunks := 0
					for block := range strings.SplitSeq(response.Body.String(), "\n\n") {
						if !strings.HasPrefix(block, "event: audio\n") {
							continue
						}
						var chunk struct {
							Sequence   int    `json:"sequence"`
							SampleRate int    `json:"sample_rate"`
							Channels   int    `json:"channels"`
							PCM        []byte `json:"pcm_f32le"`
						}
						if err := json.Unmarshal([]byte(strings.TrimPrefix(block, "event: audio\ndata: ")), &chunk); err != nil {
							t.Fatal(err)
						}
						decoded, err := media.DecodeFloat32LE(chunk.PCM)
						if err != nil || chunk.Sequence != chunks || chunk.SampleRate != 24000 || chunk.Channels != 1 || !slices.Equal(decoded, speechFixtureChunk(chunks).PCM) {
							t.Fatalf("chunk %d changed", chunks)
						}
						if math.Float32bits(decoded[1]) != 1<<31 || math.Float32bits(decoded[2]) != 1 {
							t.Fatal("transport changed signed zero or subnormal bits")
						}
						samples = append(samples, decoded...)
						chunks++
					}
					want := "event: completed"
					if failure {
						want = "event: failed"
					}
					if !strings.HasPrefix(response.Body.String(), "event: created") || !strings.Contains(response.Body.String(), want) {
						t.Fatalf("missing ordered terminal %s", want)
					}
					_, published, err := fixture.repository.Artifact(t.Context(), fixture.content.Descriptor.ID)
					if err != nil || published == failure {
						t.Fatalf("published=%v failure=%v err=%v", published, failure, err)
					}
					if !failure {
						wav, err := speechsynth.WAVContent(speechsynth.Audio{PCM: samples, Channels: 1, SampleRate: 24000})
						if err != nil || !bytes.Equal(wav.Data, fixture.content.Data) {
							t.Fatal("stream does not collect to published WAV", err)
						}
						if !native {
							replay := httptest.NewRecorder()
							handler.ServeHTTP(replay, speechOutputRequest(fixture, false))
							if strings.Contains(replay.Body.String(), "event: audio") || !strings.Contains(replay.Body.String(), "event: completed") || fixture.produced.Load() != 6 {
								t.Fatal("numbered replay regenerated speech")
							}
						}
					}
				})
			}
		})
	}
	for _, cancellation := range []string{"disconnect", "operation cancel"} {
		t.Run(cancellation+" waits for cleanup", func(t *testing.T) {
			handler, fixture := newSpeechOutputFixture(t)
			cleanup := make(chan struct{})
			finishCleanup := sync.OnceFunc(func() { close(cleanup) })
			t.Cleanup(finishCleanup)
			fixture.cleanup = cleanup
			interrupted := make(chan struct{})
			response := &stalledResponseWriter{ResponseRecorder: httptest.NewRecorder(), blocked: make(chan struct{}), interrupted: interrupted, event: "event: audio\n", unblock: sync.OnceFunc(func() { close(interrupted) })}
			t.Cleanup(response.unblock)
			ctx, cancel := context.WithCancel(t.Context())
			defer cancel()
			done := make(chan struct{})
			go func() {
				defer close(done)
				handler.ServeHTTP(response, speechOutputRequest(fixture, false).WithContext(ctx))
			}()
			select {
			case <-response.blocked:
			case <-done:
				t.Fatal("stream ended before chunk")
			}
			id, err := artifact.ParseID(response.Header().Get("X-Overgo-Operation"))
			if err != nil {
				t.Fatal(err)
			}
			if cancellation == "disconnect" {
				cancel()
			} else if !handler.operations.Cancel(id) {
				t.Fatal("operation refused cancellation")
			}
			<-fixture.cancelled
			select {
			case <-done:
				t.Fatal("handler released before executor cleanup")
			default:
			}
			finishCleanup()
			<-done
			<-fixture.released
			status, err := handler.operations.Wait(t.Context(), id)
			if err != nil || status.State != operation.StateCancelled {
				t.Fatalf("terminal cancellation: %+v %v", status, err)
			}
			if fixture.produced.Load() != 1 {
				t.Fatal("cancelled producer advanced")
			}
			if _, found, err := fixture.repository.Artifact(t.Context(), fixture.content.Descriptor.ID); err != nil || found {
				t.Fatalf("cancelled output published: %v %v", found, err)
			}
		})
	}
}
