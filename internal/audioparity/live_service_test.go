package audioparity

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/artifact"
	"overgo/internal/audiodsp"
	"overgo/internal/dataset"
	"overgo/internal/modelrecipe"
	"overgo/internal/modelrecipetest"
	"overgo/internal/modelswap"
	"overgo/internal/operation"
	"overgo/internal/overgodb"
	"overgo/internal/recipe"
	"overgo/internal/runrecord"
	"overgo/internal/server"
	"overgo/internal/speechrecognition"
	"overgo/internal/speechrecognitiontest"
	"overgo/internal/testevidence"
	"overgo/internal/testutil"
	"overgo/internal/workflowruntime"
)

// buildLiveASRServer builds the actual CLI from a clean, independent source
// fixture. Candidate verification worktrees intentionally differ from HEAD;
// attributing their executable to that parent would falsify run provenance.
// Only tracked and nonignored candidate files are copied, never model stores.
// The fixture commit changes no campaign refs and no campaign worktree bytes.
func buildLiveASRServer(t *testing.T) (string, string, string) {
	t.Helper()
	root := testutil.RepoRoot(t)
	directory := filepath.Join(t.TempDir(), "source")
	baselineCommand(t, root, "git", "clone", "--quiet", "--shared", "--no-checkout", root, directory)
	files := baselineCommand(t, root, "git", "ls-files", "-z", "--cached", "--others", "--exclude-standard")
	for name := range strings.SplitSeq(files, "\x00") {
		if name == "" {
			continue
		}
		if !filepath.IsLocal(name) {
			t.Fatalf("nonlocal source path %q", name)
		}
		data, err := os.ReadFile(filepath.Join(root, name))
		if errors.Is(err, os.ErrNotExist) {
			continue // A candidate deletion must stay deleted in the fixture.
		}
		if err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(directory, name)
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o600); err != nil {
			t.Fatal(err)
		}
	}
	// This independent test repository has no production commit admission hook.
	baselineCommand(t, directory, "git", "config", "core.hooksPath", t.TempDir())
	baselineCommand(t, directory, "git", "add", "--force", "--all")
	baselineCommand(t, directory, "git", "-c", "user.name=Audio source fixture", "-c", "user.email=audio-fixture@invalid", "-c", "commit.gpgsign=false", "commit", "--quiet", "--allow-empty", "-m", "Exact audio service candidate fixture")
	commit, err := runrecord.VerifyingCommit(directory)
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(t.TempDir(), "server.exe")
	baselineCommand(t, directory, "go", "build", "-buildvcs=true", "-trimpath", "-o", binary, "./cmd/server")
	return binary, directory, commit
}

type liveASRFixture struct {
	store              *overgodb.Store
	path, sourceCommit string
	definitions        []recipe.Definition
	inputs             [][]byte
	native             speechrecognitiontest.NativeFixture
	launcher           modelswap.ServerLauncher
	seenRuns           map[artifact.ID]bool
}

func newLiveASRFixture(t *testing.T) *liveASRFixture {
	t.Helper()
	if testing.Short() {
		t.Skip(testevidence.ShortIntegrationSkip + ": real ASR child startup requires registered models and corpus")
	}
	previousLog := log.Writer()
	log.SetOutput(testutil.UnexpectedLog{Test: t})
	t.Cleanup(func() { log.SetOutput(previousLog) })
	binary, directory, commit := buildLiveASRServer(t)
	t.Setenv("OVERGO_API_KEY", audioHTTPFixtureKey)
	f := &liveASRFixture{path: filepath.Join(t.TempDir(), "store"), sourceCommit: commit, seenRuns: make(map[artifact.ID]bool)}
	var err error
	f.store, err = overgodb.Open(f.path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { f.store.Close() })
	ctc := loadCTCTrainingFixture(t)
	publication := audioPublication{store: f.store}
	base, _ := publication.publishTranscriptionBase(t, ctc)
	f.native = speechrecognitiontest.PublishNative(t, f.store)
	f.definitions = []recipe.Definition{base, f.native.Definition}
	f.inputs = [][]byte{ctc.payload, speechrecognitiontest.NativeWave(t, f.native.Clips[0].PCM, f.native.Profile.Frontend.SampleRate)}
	for _, definition := range f.definitions {
		verification, err := modelrecipetest.PublishVerification(t.Context(), f.store, "live-service/fixture/"+definition.ID.String(), definition.ID)
		if err != nil {
			t.Fatal(err)
		}
		if err := modelrecipe.ActivateCapability(t.Context(), f.store, definition, verification, recipe.EvidenceParity, "isolated real-model serving acceptance, not production promotion"); err != nil {
			t.Fatal(err)
		}
	}
	policyBytes, err := os.ReadFile("../dataset/testdata/audio_inspection_policy.json")
	if err != nil {
		t.Fatal(err)
	}
	var inspection dataset.AudioInspectionPolicy
	if err := json.Unmarshal(policyBytes, &inspection); err != nil {
		t.Fatal(err)
	}
	inspection.MaximumSamples = max(ctc.record.Samples, uint64(len(f.native.Clips[0].PCM)))
	inspection.MaximumEncodedBytes = uint64(max(len(f.inputs[0]), len(f.inputs[1])))
	policyBytes, err = json.Marshal(server.TranscriptionPolicy{MemoryBytes: adapterAcceptanceMemory, Inspection: inspection})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(filepath.Dir(f.path), "transcription_policy.json"), policyBytes, 0o600); err != nil {
		t.Fatal(err)
	}
	f.launcher = modelswap.ServerLauncher{Binary: binary, Store: f.path, Dir: directory}
	return f
}

func (f *liveASRFixture) proxy(t *testing.T) (*modelswap.Supervisor, *httptest.Server) {
	t.Helper()
	reader, err := overgodb.OpenReadOnly(f.path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { reader.Close() })
	supervisor, err := modelswap.New(f.launcher, 0)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { supervisor.Close() })
	front := httptest.NewServer(&modelswap.Proxy{Supervisor: supervisor, Resolver: &modelswap.CatalogResolver{Store: reader, Limit: len(f.definitions)}})
	t.Cleanup(front.Close)
	return supervisor, front
}

func (f *liveASRFixture) transcribe(t *testing.T, front *httptest.Server, index int) string {
	t.Helper()
	definition := f.definitions[index]
	request := audioHTTPRequest(t, definition, f.inputs[index], true)
	request.RequestURI = ""
	request.URL.Scheme, request.URL.Host = "http", strings.TrimPrefix(front.URL, "http://")
	request.Host = request.URL.Host
	query := request.URL.Query()
	query.Set("swap", definition.Model.String())
	request.URL.RawQuery = query.Encode()
	ctx, cancel := context.WithTimeoutCause(t.Context(), 2*time.Minute, errors.New("live ASR transcription exceeded test budget"))
	defer cancel()
	response, err := front.Client().Do(request.WithContext(ctx))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	raw, err := io.ReadAll(response.Body)
	var output struct {
		Text string `json:"text"`
	}
	if err != nil || response.StatusCode != http.StatusOK || json.Unmarshal(raw, &output) != nil || output.Text == "" {
		t.Fatalf("transcription model=%s status=%d output=%s error=%v", definition.Model, response.StatusCode, raw, err)
	}
	return output.Text
}

func TestASRModelSwitchAcceptance(t *testing.T) {
	f := newLiveASRFixture(t)
	supervisor, front := f.proxy(t)
	baseline := make(map[int]string)
	for _, index := range []int{0, 1, 0} {
		text := f.transcribe(t, front, index)
		if previous, found := baseline[index]; found && text != previous {
			t.Fatalf("transcript changed after model restart: %q != %q", text, previous)
		}
		baseline[index] = text
		if current, ok := supervisor.Status(); !ok || current.Model != f.definitions[index].Model.String() {
			t.Fatal("proxy selected a different model identity")
		}
		f.assertRuns(t, front, f.definitions[index])
	}
	if baseline[1] != f.native.Clips[0].Offline {
		t.Fatalf("native transcript=%q want=%q", baseline[1], f.native.Clips[0].Offline)
	}
	if err := supervisor.Close(); err != nil {
		t.Fatal(err)
	}
	if _, running := supervisor.Status(); running {
		t.Fatal("closed supervisor retained a child")
	}
	f.rollback(t)
	f.refusedStartup(t)
	t.Logf("actual server CLI: 2 ASR models, exact switch/restart transcript, predecessor rollback, 2 refused startup policies and recovery; recipe/source attribution fixture=%s; no GPU, training, throughput or promotion claim", f.sourceCommit)
}

func (f *liveASRFixture) refusedStartup(t *testing.T) {
	t.Helper()
	path := filepath.Join(filepath.Dir(f.path), "transcription_policy.json")
	policy, err := server.LoadTranscriptionPolicy(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, test := range []struct {
		name   string
		mutate func(*server.TranscriptionPolicy)
	}{
		{"mismatched-recipe", func(policy *server.TranscriptionPolicy) { policy.Recipe = f.definitions[0].ID }},
		{"insufficient-memory", func(policy *server.TranscriptionPolicy) { policy.MemoryBytes = 1 }},
	} {
		t.Run(test.name, func(t *testing.T) {
			candidate := policy
			test.mutate(&candidate)
			data, err := json.Marshal(candidate)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(path, data, 0o600); err != nil {
				t.Fatal(err)
			}
			supervisor, front := f.proxy(t)
			ctx, cancel := context.WithTimeoutCause(t.Context(), time.Minute, errors.New("refused ASR startup exceeded test budget"))
			defer cancel()
			request, err := http.NewRequestWithContext(ctx, http.MethodGet, front.URL+"/health?swap="+f.native.Definition.Model.String(), nil)
			if err != nil {
				t.Fatal(err)
			}
			response, err := front.Client().Do(request)
			if err != nil {
				t.Fatal(err)
			}
			response.Body.Close()
			if _, running := supervisor.Status(); response.StatusCode != http.StatusServiceUnavailable || running {
				t.Fatalf("invalid startup policy admitted: HTTP=%d running=%t", response.StatusCode, running)
			}
		})
	}
	data, err := json.Marshal(policy)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
	supervisor, recovered := f.proxy(t)
	if text := f.transcribe(t, recovered, 1); text != f.native.Clips[0].Offline {
		t.Fatal("startup refusal leaked resources or changed recovered output")
	}
	f.assertRuns(t, recovered, f.native.Definition)
	if err := supervisor.Close(); err != nil {
		t.Fatal(err)
	}
}

func (f *liveASRFixture) rollback(t *testing.T) {
	t.Helper()
	if err := f.store.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	// The native capture requires masking the incomplete final hop. Publish a
	// valid but behaviorally incorrect successor to exercise retirement on an
	// actual preprocessing mismatch, not a fabricated model-load failure.
	broken := f.native.Profile
	broken.Frontend.MaskIncompleteHop = false
	broken, err := speechrecognition.NewExecutionProfile(broken)
	if err != nil {
		t.Fatal(err)
	}
	location, err := artifact.AvailablePath(t.Context(), f.store, f.native.Definition.Model, artifact.LocationDirectory)
	if err != nil {
		t.Fatal(err)
	}
	successor := speechrecognitiontest.PublishModel(t, f.store, location, broken)
	verification, err := modelrecipetest.PublishVerification(t.Context(), f.store, "live/rollback/successor", successor.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.ActivateCapability(t.Context(), f.store, successor, verification, recipe.EvidenceParity, "isolated successor for failed-boundary rollback; not production promotion"); err != nil {
		t.Fatal(err)
	}
	started := time.Now()
	clip := f.native.Clips[0]
	for _, profile := range []speechrecognition.ExecutionProfile{f.native.Profile, broken} {
		frontend, err := audiodsp.NewFrontend(profile.Frontend, adapterAcceptanceMemory)
		if err != nil {
			t.Fatal(err)
		}
		var workspace audiodsp.Workspace
		values, _, err := frontend.Process(t.Context(), [][]float32{clip.PCM}, profile.Frontend.SampleRate, &workspace, audiodsp.ProcessOptions{})
		if err != nil {
			t.Fatal(err)
		}
		begin := uint64(len(clip.PCM)) / profile.Frontend.Geometry.HopSamples * uint64(profile.Frontend.Geometry.FeatureBins)
		if begin >= uint64(len(values)) {
			t.Fatal("rollback fixture has no incomplete hop")
		}
		mismatch := false
		for _, value := range values[begin:] {
			mismatch = mismatch || value != 0
		}
		if mismatch == profile.Frontend.MaskIncompleteHop {
			t.Fatal("native capture boundary did not distinguish successor from predecessor")
		}
	}
	duration := uint64(time.Since(started).Nanoseconds())
	environment, err := runrecord.CurrentEnvironment("cpu", "go")
	if err != nil {
		t.Fatal(err)
	}
	publication := audioPublication{store: f.store}
	batch, err := environment.Batch("live/rollback/environment")
	publication.commit(t, batch, err)
	failed, err := runrecord.NewGateRecord(successor.ID, environment.ID, f.sourceCommit, runrecord.OutcomeFailed, "native-incomplete-hop-mismatch", duration,
		[]runrecord.GateStep{{Name: "native-incomplete-hop", Phase: runrecord.PhaseValidate, Outcome: runrecord.StepFailed, DurationNS: duration}})
	if err != nil {
		t.Fatal(err)
	}
	batch, err = failed.Batch("live/rollback/failed-boundary")
	publication.commit(t, batch, err)
	if err := modelrecipe.RetireActiveCapability(t.Context(), f.store, successor, modelrecipe.Verification{Gate: failed.Result.ID, Run: failed.Run.ID}, "native capture's incomplete-hop mask differs"); err != nil {
		t.Fatal(err)
	}
	fresh, err := modelrecipetest.PublishVerification(t.Context(), f.store, "live/rollback/predecessor-reverified", f.native.Definition.ID)
	if err != nil {
		t.Fatal(err)
	}
	if err := modelrecipe.RollbackActivation(t.Context(), f.store, successor.Model, recipe.TaskTranscription, fresh, recipe.EvidenceParity, "restore native-capture-bound predecessor"); err != nil {
		t.Fatal(err)
	}
	active, _, err := modelrecipe.ResolveActiveCapability(t.Context(), f.store, successor.Model, recipe.TaskTranscription)
	if err != nil || active.Definition.ID != f.native.Definition.ID {
		t.Fatalf("rollback active recipe=%s error=%v", active.Definition.ID, err)
	}
	if status, found, err := modelrecipe.Status(t.Context(), f.store, successor.ID); err != nil || !found || status != recipe.StatusSuperseded {
		t.Fatalf("retired successor history changed: %s %t %v", status, found, err)
	}
	supervisor, front := f.proxy(t)
	if text := f.transcribe(t, front, 1); text != clip.Offline {
		t.Fatal("rolled-back child did not restore native output")
	}
	f.assertRuns(t, front, f.native.Definition)
	if err := supervisor.Close(); err != nil {
		t.Fatal(err)
	}
}

func (f *liveASRFixture) assertRuns(t *testing.T, front *httptest.Server, definition recipe.Definition) {
	t.Helper()
	request, err := http.NewRequestWithContext(t.Context(), http.MethodGet, front.URL+"/operations", nil)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Authorization", "Bearer "+audioHTTPFixtureKey)
	response, err := front.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var statuses []operation.Status
	if err := json.NewDecoder(response.Body).Decode(&statuses); err != nil || response.StatusCode != http.StatusOK {
		t.Fatalf("operations HTTP=%d: %v", response.StatusCode, err)
	}
	if err := f.store.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	found := false
	for _, status := range statuses {
		if status.Run == nil || status.State != operation.StateCompleted || f.seenRuns[*status.Run] {
			continue
		}
		run, err := runrecord.RequireExactRun(t.Context(), f.store, *status.Run)
		if err != nil {
			t.Fatal(err)
		}
		if run.Recipe != definition.ID || status.Recipe != definition.ID || run.CodeCommit != f.sourceCommit || !run.Environment.Valid() || len(run.Outputs) == 0 {
			t.Fatalf("unbound served run: %+v", run)
		}
		transcript, err := speechrecognition.RequireTranscription(t.Context(), f.store, run.Outputs[0])
		if err != nil {
			t.Fatal(err)
		}
		for index, candidate := range f.definitions {
			if candidate.ID != definition.ID {
				continue
			}
			source, err := artifact.IdentifyBytes(artifact.KindFile, f.inputs[index])
			if err != nil || transcript.Source.Audio != source {
				t.Fatalf("served run changed source audio: %s != %s: %v", transcript.Source.Audio, source, err)
			}
		}
		f.seenRuns[run.ID] = true
		found = true
	}
	if !found {
		t.Fatalf("no completed source-bound run for %s", definition.ID)
	}
}

func TestASRLiveOperationsAcceptance(t *testing.T) {
	f := newLiveASRFixture(t)
	source, chunks := f.native.PublishStream(t, f.store, f.native.Clips[0], f.native.Profile.Frontend.SampleRate+1)
	supervisor, front := f.proxy(t)
	initial := struct {
		Model string `json:"model"`
		speechrecognition.StreamRequest
	}{Model: f.native.Definition.Model.String(), StreamRequest: speechrecognition.StreamRequest{Source: source}}
	// With no swap query, the proxy must route the opening envelope and return
	// created before this producer sends a chunk or closes its request body.
	client := openLiveASRStream(t, front, initial)
	defer client.close()
	first := client.send(t, chunks[0])
	current, running := supervisor.Status()
	if !running || current.Model != initial.Model {
		t.Fatal("live envelope started the wrong model")
	}
	other := f.definitions[0]
	location, err := artifact.AvailablePath(t.Context(), f.store, other.Model, artifact.LocationDirectory)
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancelCause(t.Context())
	swapped := make(chan error, 1)
	go func() {
		_, release, err := supervisor.Acquire(ctx, modelswap.Servable{Name: filepath.Base(location), Model: other.Model.String(), Location: location})
		if release != nil {
			release()
		}
		swapped <- err
	}()
	select {
	case err := <-swapped:
		t.Fatalf("swap did not wait for open audio stream: %v", err)
	case <-time.After(30 * time.Millisecond):
	}
	cancel(context.Canceled)
	select {
	case err := <-swapped:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("canceled live swap: %v", err)
		}
	case <-time.After(time.Second):
		t.Fatal("canceled swap retained an active waiter")
	}
	if after, running := supervisor.Status(); !running || after != current {
		t.Fatal("canceled swap replaced the streaming child")
	}
	client.close()
	if err := supervisor.Close(); err != nil {
		t.Fatal(err)
	}
	// A fresh supervisor and actual child restore the durable cursor. No old
	// Go object, predictor workspace or in-memory proxy state is reused.
	_, resumed := f.proxy(t)
	initial.Resume = first.Checkpoint
	client = openLiveASRStream(t, resumed, initial)
	defer client.close()
	var transcript strings.Builder
	transcript.WriteString(first.Transcript.Text)
	for _, chunk := range chunks[1:] {
		event := client.send(t, chunk)
		if event.Transcript.Recipe != f.native.Definition.ID || event.Transcript.Source != source {
			t.Fatal("resumed stream changed recipe or source")
		}
		transcript.WriteString(event.Transcript.Text)
	}
	client.close()
	if transcript.String() != f.native.Clips[0].Streaming {
		t.Fatalf("resumed child transcript=%q want=%q", transcript.String(), f.native.Clips[0].Streaming)
	}
	// Concurrent HTTP requests use the existing admission owner; their outputs
	// must be isolated even when recognizer execution is serialized by a lease.
	baseline := f.transcribe(t, resumed, 0)
	for range 2 {
		t.Run("concurrent-request", func(t *testing.T) {
			t.Parallel()
			if got := f.transcribe(t, resumed, 0); got != baseline {
				t.Fatalf("concurrent request changed transcript: %q != %q", got, baseline)
			}
		})
	}
	t.Logf("actual native child: %d live chunks, canceled pending swap, fresh-child checkpoint resume, 2 concurrent CTC requests; fixture source=%s; CPU only, no parallel-kernel or performance claim", len(chunks), f.sourceCommit)
}

type liveASREvent struct {
	Transcript speechrecognition.TranscriptionChunk `json:"transcript"`
	Checkpoint artifact.ID                          `json:"checkpoint"`
	Output     artifact.ID                          `json:"output"`
}

type liveASRStream struct {
	writer  *io.PipeWriter
	body    io.ReadCloser
	scanner *bufio.Scanner
	cancel  context.CancelFunc
}

func openLiveASRStream(t *testing.T, front *httptest.Server, initial any) *liveASRStream {
	t.Helper()
	ctx, cancel := context.WithTimeoutCause(t.Context(), 2*time.Minute, errors.New("live ASR stream exceeded test budget"))
	reader, writer := io.Pipe()
	client := &liveASRStream{writer: writer, cancel: cancel}
	t.Cleanup(client.close)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, front.URL+"/v1/audio/transcriptions", reader)
	if err != nil {
		t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/x-ndjson")
	request.Header.Set("Authorization", "Bearer "+audioHTTPFixtureKey)
	sent := make(chan error, 1)
	go func() { sent <- json.NewEncoder(writer).Encode(initial) }()
	response, err := front.Client().Do(request)
	if err != nil {
		t.Fatal(err)
	}
	client.body = response.Body
	if response.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(response.Body)
		t.Fatalf("live startup HTTP=%d: %s", response.StatusCode, body)
	}
	if err := <-sent; err != nil {
		t.Fatal(err)
	}
	client.scanner = bufio.NewScanner(client.body)
	if name, _ := client.event(t); name != "created" {
		t.Fatal("live child did not acknowledge source before chunks")
	}
	return client
}

func (client *liveASRStream) event(t *testing.T) (string, []byte) {
	t.Helper()
	var name string
	for client.scanner.Scan() {
		line := client.scanner.Text()
		if value, ok := strings.CutPrefix(line, "event: "); ok {
			name = value
		}
		if value, ok := strings.CutPrefix(line, "data: "); ok {
			return name, []byte(value)
		}
	}
	t.Fatalf("live stream ended without event: %v", client.scanner.Err())
	return "", nil
}

func (client *liveASRStream) send(t *testing.T, chunk workflowruntime.AudioStreamChunk) liveASREvent {
	t.Helper()
	if err := json.NewEncoder(client.writer).Encode(chunk); err != nil {
		t.Fatal(err)
	}
	name, data := client.event(t)
	var event liveASREvent
	if json.Unmarshal(data, &event) != nil || !event.Output.Valid() || !event.Checkpoint.Valid() || event.Transcript.Sequence != chunk.Sequence || event.Transcript.Final != chunk.Final ||
		chunk.Final && name != "done" || !chunk.Final && name != "token" {
		t.Fatalf("live chunk event %s: %s", name, data)
	}
	return event
}

func (client *liveASRStream) close() {
	client.cancel()
	client.writer.Close()
	if client.body != nil {
		client.body.Close()
	}
}
