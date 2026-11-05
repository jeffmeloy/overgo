package server

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelswap"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/remoteprovider"
	"overgo/internal/runrecord"
	"overgo/internal/testcheck"
	"overgo/internal/testutil"
)

func TestWorkbenchTransactionWriter(t *testing.T) {
	// Serial: a helper it calls sets the process environment.
	t.Run("served external evidence publication", testWorkbenchEvidencePublication)
	t.Run("catalog observes activation and retirement", testWorkbenchCatalogRefresh)
	root := t.TempDir()
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	defer store.Close()
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store, OvergoDBPath: root}, &fakeGenerator{})
	testcheck.NoError(t, err)
	defer handler.Close()
	external, err := overgodb.Create(root)
	if err != nil {
		t.Fatalf("serving handle blocks external writer: %v", err)
	}
	defer external.Close()
	descriptor := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "external gate publication")}
	head, err := artifact.CommitBatch(t.Context(), external, artifact.Batch{Key: "external/gate", Artifacts: []artifact.Descriptor{descriptor}})
	testcheck.NoError(t, err)
	_, sequence := external.Head()
	response := serveTestRequest(handler, http.MethodGet, "/store/deltas", "")
	if response.Code != http.StatusOK {
		t.Fatalf("browse status=%d body=%s", response.Code, response.Body)
	}
	var view storeDeltasResponse
	testcheck.NoError(t, json.Unmarshal(response.Body.Bytes(), &view))
	if view.Head != head.String() || view.Sequence != sequence {
		t.Fatalf("serving view is stale: %s@%d want %s@%d", view.Head, view.Sequence, head, sequence)
	}
}

type workbenchWriterMessage struct {
	Action string
	Key    string
}
type workbenchWriterReply struct {
	State      string
	Head       string
	Sequence   uint64
	Obligation artifact.ID `json:"obligation,omitzero"`
}

func TestWorkbenchEvidenceProcess(t *testing.T) {
	t.Parallel()
	root := os.Getenv("OVERGO_TEST_WORKBENCH_REPOSITORY")
	if root == "" {
		return
	}
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	decoder, encoder := json.NewDecoder(os.Stdin), json.NewEncoder(os.Stdout)
	testcheck.NoError(t, encoder.Encode(workbenchWriterReply{State: "ready"}))
	var held *processlock.Lock
	for {
		var message workbenchWriterMessage
		if err := decoder.Decode(&message); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if message.Action == "hold" {
			held, err = processlock.AcquireContext(t.Context(), root+".lock", 0o644)
			testcheck.NoError(t, err)
			testcheck.NoError(t, encoder.Encode(workbenchWriterReply{State: "held"}))
			continue
		}
		if held != nil {
			testcheck.NoError(t, held.Close())
			held = nil
		}
		task := testutil.ArtifactID(t, artifact.KindRecipe, "workbench gate invocation")
		source := testutil.ArtifactID(t, artifact.KindEvidence, message.Key)
		obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{Task: task, Name: "go-test", Scope: "complete:workbench", Sources: []artifact.ID{source}})
		testcheck.NoError(t, err)
		content, err := obligation.Content()
		testcheck.NoError(t, err)
		head, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "gate/obligation/" + obligation.ID.String(), Artifacts: []artifact.Descriptor{{ID: task}, {ID: source}}, Contents: []artifact.Content{content}, Lineage: obligation.Lineage()})
		testcheck.NoError(t, err)
		_, sequence := store.Head()
		testcheck.NoError(t, encoder.Encode(workbenchWriterReply{State: "published", Head: head.String(), Sequence: sequence, Obligation: obligation.ID}))
	}
	if held != nil {
		held.Close()
	}
	testcheck.NoError(t, store.Close())
	os.Exit(0)
}

func testWorkbenchEvidencePublication(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	defer store.Close()
	var wanted artifact.ID
	registered := 0
	intake := LibraryIntake{
		ModelFiles: func(path, projector string) (string, string, error) { return path, projector, nil },
		Register: func(ctx context.Context, repository *overgodb.Store, _, _ string) (map[string]any, error) {
			if repository != store {
				return nil, errors.New("library did not borrow serving store")
			}
			if _, err := runrecord.RequireAgentObligation(ctx, repository, wanted); err != nil {
				return nil, err
			}
			_, err := artifact.CommitBatch(ctx, repository, artifact.Batch{Key: "workspace/ack/" + wanted.String(), Aliases: []artifact.AliasBinding{{Name: "workbench/ack", Target: wanted}}})
			registered++
			return map[string]any{"obligation": wanted.String()}, err
		},
	}
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store, OvergoDBPath: root, LibraryIntake: intake}, &fakeGenerator{})
	testcheck.NoError(t, err)
	defer handler.Close()
	front := httptest.NewServer(handler)
	defer front.Close()
	ctx := t.Context()
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestWorkbenchEvidenceProcess$", "-test.timeout=45s")
	child.Env = append(os.Environ(), "OVERGO_TEST_WORKBENCH_REPOSITORY="+root)
	input, err := child.StdinPipe()
	testcheck.NoError(t, err)
	output, err := child.StdoutPipe()
	testcheck.NoError(t, err)
	child.Stderr = os.Stderr
	testcheck.NoError(t, child.Start())
	t.Cleanup(func() { input.Close(); child.Process.Kill(); child.Wait() })
	encoder, decoder := json.NewEncoder(input), json.NewDecoder(output)
	read := func(state string) workbenchWriterReply {
		var reply workbenchWriterReply
		if err := decoder.Decode(&reply); err != nil || reply.State != state {
			t.Fatalf("publisher = %+v, %v", reply, err)
		}
		return reply
	}
	send := func(action, key string) {
		testcheck.NoError(t, encoder.Encode(workbenchWriterMessage{Action: action, Key: key}))
	}
	read("ready")
	send("hold", "")
	read("held")
	health, err := http.NewRequestWithContext(ctx, http.MethodGet, front.URL+"/health", nil)
	testcheck.NoError(t, err)
	response, err := front.Client().Do(health)
	testcheck.NoError(t, err)
	response.Body.Close()
	if response.StatusCode != http.StatusOK {
		t.Fatalf("health under live writer = %d", response.StatusCode)
	}
	// A request whose caller has already left waits for no live writer.
	cancelled, stop := context.WithCancelCause(t.Context())
	stop(errors.New("the request's caller left"))
	blocked := httptest.NewRecorder()
	handler.ServeHTTP(blocked, httptest.NewRequestWithContext(cancelled, http.MethodPost, "/library/register", strings.NewReader(`{"kind":"model","path":"fixture"}`)))
	if blocked.Code != http.StatusServiceUnavailable || !strings.Contains(blocked.Body.String(), "store_busy") || registered != 0 {
		t.Fatalf("cancelled admission = %d %s registrations=%d", blocked.Code, blocked.Body, registered)
	}
	send("publish", "first evidence")
	first := read("published")
	view := serveTestRequest(handler, http.MethodGet, "/store/deltas", "")
	var delta storeDeltasResponse
	if err := json.Unmarshal(view.Body.Bytes(), &delta); err != nil || delta.Head != first.Head || delta.Sequence != first.Sequence {
		t.Fatalf("external evidence not visible: %+v, %v", delta, err)
	}
	send("publish", "second evidence")
	wanted = read("published").Obligation
	registeredResponse := serveTestRequest(handler, http.MethodPost, "/library/register", `{"kind":"model","path":"fixture"}`)
	if registeredResponse.Code != http.StatusOK || registered != 1 {
		t.Fatalf("fresh library intake = %d %s", registeredResponse.Code, registeredResponse.Body)
	}
	send("publish", "after workspace commit")
	read("published")
	testcheck.NoError(t, store.Refresh(t.Context()))
	if got, found, err := store.ResolveAlias(t.Context(), "workbench/ack"); err != nil || !found || got != wanted {
		t.Fatalf("workspace write lost: %s, %v, %v", got, found, err)
	}
	testcheck.NoError(t, input.Close())
	testcheck.NoError(t, child.Wait())
	t.Log("served health and cancellation coexist with an independent evidence writer; refreshed views and interleaved workspace publication preserved")
}

func testWorkbenchCatalogRefresh(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	defer store.Close()
	external, err := overgodb.Create(root)
	testcheck.NoError(t, err)
	defer external.Close()
	resolver := &modelswap.CatalogResolver{Store: store, Limit: 16}
	if entries, _, err := resolver.Catalog(t.Context()); err != nil || len(entries) != 0 {
		t.Fatalf("initial catalog: %v, %v", entries, err)
	}
	t.Setenv("OVERGO_TEST_WORKBENCH_KEY", "fixture-key")
	declaration, err := remoteprovider.Declare(t.Context(), external, remoteprovider.Provider{Name: "workbench", Endpoint: "https://fixture.example/v1", KeyEnvironment: "OVERGO_TEST_WORKBENCH_KEY", Model: "vendor/fixture"}, evaluationTestCommit)
	testcheck.NoError(t, err)
	if _, found, err := resolver.Resolve(t.Context(), declaration.Model.String()); err != nil || !found {
		t.Fatalf("new activation missing: %v, %v", found, err)
	}
	testcheck.NoError(t, remoteprovider.Retire(t.Context(), external, 16, declaration.Location, evaluationTestCommit, "fixture retirement"))
	if _, found, err := resolver.Resolve(t.Context(), declaration.Model.String()); err != nil || found {
		t.Fatalf("retired activation remains: %v, %v", found, err)
	}
}
