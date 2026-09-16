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
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/artifact"
	"overgo/internal/modelswap"
	"overgo/internal/overgodb"
	"overgo/internal/processlock"
	"overgo/internal/remoteprovider"
	"overgo/internal/runrecord"
	"overgo/internal/testutil"
)

func TestWorkbenchTransactionWriter(t *testing.T) {
	t.Run("served external evidence publication", testWorkbenchEvidencePublication)
	t.Run("catalog observes activation and retirement", testWorkbenchCatalogRefresh)
	t.Run("generation observes external activation", TestGenerationWorkspaceListsUnresolvableActivationWithRefusal)
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	handler, err := New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, Repository: store, OvergoDBPath: root}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	external, err := overgodb.Open(root)
	if err != nil {
		t.Fatalf("serving handle blocks external writer: %v", err)
	}
	defer external.Close()
	descriptor := artifact.Descriptor{ID: testutil.ArtifactID(t, artifact.KindEvidence, "external gate publication")}
	head, err := artifact.CommitBatch(t.Context(), external, artifact.Batch{Key: "external/gate", Artifacts: []artifact.Descriptor{descriptor}})
	if err != nil {
		t.Fatal(err)
	}
	_, sequence := external.Head()
	response := serveTestRequest(handler, http.MethodGet, "/store/deltas", "")
	if response.Code != http.StatusOK {
		t.Fatalf("browse status=%d body=%s", response.Code, response.Body)
	}
	var view storeDeltasResponse
	if err := json.Unmarshal(response.Body.Bytes(), &view); err != nil {
		t.Fatal(err)
	}
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
	root := os.Getenv("OVERGO_TEST_WORKBENCH_REPOSITORY")
	if root == "" {
		return
	}
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	decoder, encoder := json.NewDecoder(os.Stdin), json.NewEncoder(os.Stdout)
	if err := encoder.Encode(workbenchWriterReply{State: "ready"}); err != nil {
		t.Fatal(err)
	}
	var held *processlock.Lock
	for {
		var message workbenchWriterMessage
		if err := decoder.Decode(&message); errors.Is(err, io.EOF) {
			break
		} else if err != nil {
			t.Fatal(err)
		}
		if message.Action == "hold" {
			held, err = processlock.AcquireContext(t.Context(), filepath.Join(root, "overgodb.lock"), 0o644)
			if err != nil {
				t.Fatal(err)
			}
			if err := encoder.Encode(workbenchWriterReply{State: "held"}); err != nil {
				t.Fatal(err)
			}
			continue
		}
		if held != nil {
			if err := held.Close(); err != nil {
				t.Fatal(err)
			}
			held = nil
		}
		task := testutil.ArtifactID(t, artifact.KindRecipe, "workbench gate invocation")
		source := testutil.ArtifactID(t, artifact.KindEvidence, message.Key)
		obligation, err := runrecord.NewAgentObligation(runrecord.AgentObligation{Task: task, Name: "go-test", Scope: "complete:workbench", Sources: []artifact.ID{source}})
		if err != nil {
			t.Fatal(err)
		}
		content, err := obligation.Content()
		if err != nil {
			t.Fatal(err)
		}
		head, err := artifact.CommitBatch(t.Context(), store, artifact.Batch{Key: "gate/obligation/" + obligation.ID.String(), Artifacts: []artifact.Descriptor{{ID: task}, {ID: source}}, Contents: []artifact.Content{content}, Lineage: obligation.Lineage()})
		if err != nil {
			t.Fatal(err)
		}
		_, sequence := store.Head()
		if err := encoder.Encode(workbenchWriterReply{State: "published", Head: head.String(), Sequence: sequence, Obligation: obligation.ID}); err != nil {
			t.Fatal(err)
		}
	}
	if held != nil {
		held.Close()
	}
	if err := store.Close(); err != nil {
		t.Fatal(err)
	}
	os.Exit(0)
}

func testWorkbenchEvidencePublication(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
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
	if err != nil {
		t.Fatal(err)
	}
	defer handler.Close()
	front := httptest.NewServer(handler)
	defer front.Close()
	ctx := t.Context()
	child := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestWorkbenchEvidenceProcess$", "-test.timeout=45s")
	child.Env = append(os.Environ(), "OVERGO_TEST_WORKBENCH_REPOSITORY="+root)
	input, err := child.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	output, err := child.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	child.Stderr = os.Stderr
	if err := child.Start(); err != nil {
		t.Fatal(err)
	}
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
		if err := encoder.Encode(workbenchWriterMessage{Action: action, Key: key}); err != nil {
			t.Fatal(err)
		}
	}
	read("ready")
	send("hold", "")
	read("held")
	health, err := http.NewRequestWithContext(ctx, http.MethodGet, front.URL+"/health", nil)
	if err != nil {
		t.Fatal(err)
	}
	response, err := front.Client().Do(health)
	if err != nil {
		t.Fatal(err)
	}
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
	if err := store.Refresh(t.Context()); err != nil {
		t.Fatal(err)
	}
	if got, found, err := store.ResolveAlias(t.Context(), "workbench/ack"); err != nil || !found || got != wanted {
		t.Fatalf("workspace write lost: %s, %v, %v", got, found, err)
	}
	if err := input.Close(); err != nil {
		t.Fatal(err)
	}
	if err := child.Wait(); err != nil {
		t.Fatal(err)
	}
	t.Log("served health and cancellation coexist with an independent evidence writer; refreshed views and interleaved workspace publication preserved")
}

func testWorkbenchCatalogRefresh(t *testing.T) {
	root := t.TempDir()
	store, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	external, err := overgodb.Open(root)
	if err != nil {
		t.Fatal(err)
	}
	defer external.Close()
	resolver := &modelswap.CatalogResolver{Store: store, Limit: 16}
	if entries, _, err := resolver.Catalog(t.Context()); err != nil || len(entries) != 0 {
		t.Fatalf("initial catalog: %v, %v", entries, err)
	}
	t.Setenv("OVERGO_TEST_WORKBENCH_KEY", "fixture-key")
	declaration, err := remoteprovider.Declare(t.Context(), external, remoteprovider.Provider{Name: "workbench", Endpoint: "https://fixture.example/v1", KeyEnvironment: "OVERGO_TEST_WORKBENCH_KEY", Model: "vendor/fixture"}, evaluationTestCommit)
	if err != nil {
		t.Fatal(err)
	}
	if _, found, err := resolver.Resolve(t.Context(), declaration.Model.String()); err != nil || !found {
		t.Fatalf("new activation missing: %v, %v", found, err)
	}
	if err := remoteprovider.Retire(t.Context(), external, 16, declaration.Location, evaluationTestCommit, "fixture retirement"); err != nil {
		t.Fatal(err)
	}
	if _, found, err := resolver.Resolve(t.Context(), declaration.Model.String()); err != nil || found {
		t.Fatalf("retired activation remains: %v, %v", found, err)
	}
}
