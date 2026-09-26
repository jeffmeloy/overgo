package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/agenttool"
	"overgo/internal/artifact"
	"overgo/internal/overgodb"
)

// TestAgentRecoveryToolProcess is the streaming tool's child process; a
// normal test run does nothing. It prints a first chunk, blocks on the
// test's loopback listener until the test releases it, then prints a second.
func TestAgentRecoveryToolProcess(t *testing.T) {
	// Serial: it is the child process's entry, which returns at once in a normal run.
	separator := slices.Index(os.Args, "--")
	if separator < 0 || separator+1 >= len(os.Args) {
		return
	}
	fmt.Println("first chunk")
	connection, err := net.Dial("tcp", os.Args[separator+1])
	if err != nil {
		os.Exit(1)
	}
	if _, err := connection.Read(make([]byte, 1)); err != nil {
		os.Exit(1)
	}
	fmt.Println("second chunk")
	os.Exit(0)
}

// streamingAgentFixture activates research-agent over the standard store
// tools and tool.stream, an argv tool whose output arrives in two pieces
// around the returned release. The argv program is this test binary, found
// on PATH, so a caller runs serially.
func streamingAgentFixture(t *testing.T, store *overgodb.Store) (agentWorkspaceFixture, func()) {
	t.Helper()
	executable, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", filepath.Dir(executable)+string(os.PathListSeparator)+os.Getenv("PATH"))
	fixture := newAgentWorkspaceFixture(t, store, nil, nil)
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = listener.Close() })
	release := func() {
		t.Helper()
		connection, err := listener.Accept()
		if err != nil {
			t.Fatal(err)
		}
		defer connection.Close()
		if _, err := connection.Write([]byte{1}); err != nil {
			t.Fatal(err)
		}
	}
	manuals, err := agenttool.StandardManuals()
	if err != nil {
		t.Fatal(err)
	}
	stream, err := agenttool.NewManual(agenttool.Manual{
		Name: "tool.stream", Description: "Prints its output in two pieces.", Effect: agenttool.EffectInspection,
		Transport: agenttool.Transport{Kind: agenttool.TransportArgv, Program: filepath.Base(executable),
			Args: []string{"-test.run=^TestAgentRecoveryToolProcess$", "--", listener.Addr().String()}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishArgvPolicy(t.Context(), store, []string{filepath.Base(executable)}); err != nil {
		t.Fatal(err)
	}
	if _, err := agenttool.PublishManualCatalog(t.Context(), store, append(manuals, stream)); err != nil && !errors.Is(err, artifact.ErrNoChange) {
		t.Fatal(err)
	}
	definition := AgentDefinitionInput{
		Name: "research-agent", Prompt: fixture.prompt, ModelRecipe: fixture.generator.description.Identity.Recipe,
		ToolManuals: []artifact.ID{fixture.manual, stream.ID}, Policies: []artifact.ID{fixture.policy},
	}
	published := serveTestRequest(fixture.handler, http.MethodPost, "/agents/definitions", marshalAutomationJSON(t, definition))
	var created struct {
		ID artifact.ID `json:"id"`
	}
	if err := json.Unmarshal(published.Body.Bytes(), &created); published.Code != http.StatusCreated || err != nil {
		t.Fatalf("agent publish status=%d body=%s", published.Code, published.Body.String())
	}
	activateDefinitionFromAPI(t, fixture.handler, created.ID, "/agents/activate")
	return fixture, release
}

// TestAgentSessionRecoveryAndStreaming: a running tool's output reaches the
// event hub piece by piece before its step returns, the step's result is the
// whole output, and the session's thread lists the step beside its turns.
func TestAgentSessionRecoveryAndStreaming(t *testing.T) {
	// Serial: the fixture sets PATH.
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	fixture, release := streamingAgentFixture(t, store)
	events, unsubscribe, err := fixture.handler.events.subscribe(64)
	if err != nil {
		t.Fatal(err)
	}
	defer unsubscribe()
	stepped := make(chan string, 1)
	go func() {
		step := serveTestRequest(fixture.handler, http.MethodPost, "/agents/step", `{"agent":"research-agent","session":"live","tool":"tool.stream"}`)
		stepped <- fmt.Sprint(step.Code, " ", step.Body.String())
	}()
	var streamed []agentOutputEvent
	next := func() agentOutputEvent {
		t.Helper()
		for {
			select {
			case event := <-events:
				if output, ok := event.value.(agentOutputEvent); ok {
					return output
				}
			case <-t.Context().Done():
				t.Fatal("no output reached the hub")
			}
		}
	}
	streamed = append(streamed, next())
	select {
	case result := <-stepped:
		t.Fatalf("the step returned before its output finished: %s", result)
	default:
	}
	if streamed[0].Session != "research-agent:live" || streamed[0].Text != "first chunk\n" {
		t.Fatalf("first piece = %+v", streamed[0])
	}
	release()
	streamed = append(streamed, next())
	if streamed[1].Text != "second chunk\n" || streamed[1].Sequence <= streamed[0].Sequence {
		t.Fatalf("second piece = %+v after %+v", streamed[1], streamed[0])
	}
	if result := <-stepped; !strings.HasPrefix(result, "200 ") || !strings.Contains(result, `"result":"first chunk\nsecond chunk"`) {
		t.Fatalf("step = %s", result)
	}
	read := serveTestRequest(fixture.handler, http.MethodGet, "/agents/thread?agent=research-agent&session=live", "")
	var thread agentThreadResponse
	if err := json.Unmarshal(read.Body.Bytes(), &thread); err != nil || len(thread.Entries) != 1 || thread.Entries[0].Tool != "tool.stream" {
		t.Fatalf("thread = %+v, %v", thread, err)
	}
}

// TestAgentThreadSurvivesRestart: a session's answered chat turns are kept
// beside its tool steps, and a restarted server reads the session's thread
// back in order, its step count resumed; a session whose name extends this
// one's stays out of it.
func TestAgentThreadSurvivesRestart(t *testing.T) {
	t.Parallel()
	store, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer store.Close()
	first := newAgentWorkspaceFixture(t, store, nil, nil)
	activateDefinitionFromAPI(t, first.handler, publishAgentFromAPI(t, first, nil, nil), "/agents/activate")
	chat := func(fixture agentWorkspaceFixture, session string, messages ...string) {
		t.Helper()
		turns := make([]map[string]string, len(messages))
		for index, message := range messages {
			turns[index] = map[string]string{"role": "user", "content": message}
		}
		body, err := json.Marshal(map[string]any{"agent": "research-agent", "session": session, "messages": turns})
		if err != nil {
			t.Fatal(err)
		}
		if answered := serveTestRequest(fixture.handler, http.MethodPost, "/agents/chat", string(body)); answered.Code != http.StatusOK || !strings.Contains(answered.Body.String(), `"choices"`) {
			t.Fatalf("agent chat status=%d body=%s", answered.Code, answered.Body.String())
		}
	}
	chat(first, "s1", "hello")
	if step := serveTestRequest(first.handler, http.MethodPost, "/agents/step", `{"agent":"research-agent","session":"s1","tool":"store.head"}`); step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":1`) {
		t.Fatalf("step status=%d body=%s", step.Code, step.Body.String())
	}
	chat(first, "s1", "hello", "again")
	chat(first, "s1-2", "another session")
	if err := first.handler.Close(); err != nil {
		t.Fatal(err)
	}

	// A restarted server over the same store reads the thread back.
	second := newAgentWorkspaceFixture(t, store, nil, nil)
	read := serveTestRequest(second.handler, http.MethodGet, "/agents/thread?agent=research-agent&session=s1", "")
	var thread agentThreadResponse
	if err := json.Unmarshal(read.Body.Bytes(), &thread); read.Code != http.StatusOK || err != nil {
		t.Fatalf("thread status=%d body=%s", read.Code, read.Body.String())
	}
	kinds := make([]string, len(thread.Entries))
	for index, entry := range thread.Entries {
		kinds[index] = entry.Kind
	}
	if strings.Join(kinds, ",") != "turn,step,turn" || thread.Steps != 1 || thread.Session != "research-agent:s1" {
		t.Fatalf("thread = %+v", thread)
	}
	if thread.Entries[0].User != "hello" || thread.Entries[0].Assistant == "" || thread.Entries[2].User != "again" ||
		thread.Entries[1].Tool != "store.head" || thread.Entries[1].Result == "" || thread.Entries[1].Error {
		t.Fatalf("thread entries = %+v", thread.Entries)
	}
	if step := serveTestRequest(second.handler, http.MethodPost, "/agents/step", `{"agent":"research-agent","session":"s1","tool":"store.head"}`); step.Code != http.StatusOK || !strings.Contains(step.Body.String(), `"steps":2`) {
		t.Fatalf("resumed step status=%d body=%s", step.Code, step.Body.String())
	}
}
