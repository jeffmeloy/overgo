//go:build modeltest

package evaluation

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"overgo/internal/dataroot"
	"overgo/internal/discovery"
	"overgo/internal/modelswap"
	"overgo/internal/overgodb"
	"overgo/internal/processcontrol"
	"overgo/internal/recipe"
	"overgo/internal/testskip"
	"overgo/internal/testutil"
)

// The long-context acceptance's shape: the characters of articles the first request reads, the fewest prompt tokens
// that read may count to stand for the 262,144-token context, the chunks the cancelled generation streams before its
// cancellation, the tokens an answer may take, and the frozen case that puts an image beside the retained context.
const (
	longContextChars        = 1_000_000
	longContextFewestTokens = 196_608
	longContextCancelChunks = 64
	longContextAnswerTokens = 16
	longContextImageCase    = "screenshot"
	longContextModel        = "Qwen3.8-Flash-Next"
	longContextCatalogBound = 256
	// The cancelled summary's token allowance, more than its chunks, and the longest streamed line read.
	longContextSummaryTokens = 800
	longContextLineBytes     = 1 << 20
)

// longContextCodes are the archive codes placed through the articles, each at its share of the characters read.
var longContextCodes = []struct {
	room, code string
	at         float64
}{{"Amber", "418273", 0.05}, {"Cobalt", "590614", 0.25}, {"Juniper", "273958", 0.50}, {"Saffron", "846102", 0.75}, {"Indigo", "631795", 0.95}}

// TestFlashNextLongContextAcceptance serves the registered Flash-Next model and holds it to the campaign's
// long-context acceptance: real articles with five archive codes placed through them, read in one request that asks
// for the middle code; follow-ups in the same conversation asking for the first and the last; a long generation
// cancelled part-way and a follow-up after it; an image request beside the retained context and a last follow-up.
// Each answer holds its code, the read's prompt counts as the long context it is, and the card's peak over the run,
// which the server reports, stays under the device's memory. It builds and serves the server itself, so it runs in
// the model lane.
func TestFlashNextLongContextAcceptance(t *testing.T) {
	t.Parallel()
	roots, err := dataroot.Resolve(testutil.RepoRoot(t))
	if err != nil {
		t.Fatal(err)
	}
	ctx := t.Context()
	name, location := flashNextServable(t, ctx, roots.Store)
	articles := filepath.Join(roots.Datasets, "wikitext2_train_articles")
	if _, err := os.Stat(articles); err != nil {
		testskip.NotApplicable(t, "no article directory at "+articles)
	}
	frozen := filepath.Join(filepath.Dir(roots.Store), "references", "strata", "baseline-20261016", "representative.json")
	image, imageTokens, err := frozenCase(frozen, longContextImageCase)
	if err != nil {
		testskip.NotApplicable(t, err.Error())
	}
	binary := filepath.Join(t.TempDir(), "overgo-server.exe")
	receipt, err := processcontrol.Run(ctx, processcontrol.Command{
		Path: "go", Args: []string{"build", "-o", binary, "overgo/cmd/server"}, Stdout: os.Stderr, Stderr: os.Stderr,
	})
	if err != nil || receipt.ExitCode != 0 {
		t.Fatalf("the server binary did not build: %v (exit %d)", err, receipt.ExitCode)
	}
	supervisor, err := modelswap.New(modelswap.ServerLauncher{Binary: binary, Store: roots.Store, Dir: testutil.RepoRoot(t)}, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer supervisor.Close()
	lease, err := supervisor.Acquire(ctx, modelswap.Servable{Name: name, Location: location})
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	server := &longContextServer{t: t, ctx: ctx, url: lease.Target, model: name}

	question := func(room string) string {
		return "What is the archive code for the " + room + " Room? Reply with the six digits only."
	}
	var conversation []map[string]any
	ask := func(label, content string, want string) int {
		t.Helper()
		turn := append(append([]map[string]any(nil), conversation...), map[string]any{"role": "user", "content": content})
		answer, prompt, took := server.answer(turn, longContextAnswerTokens)
		if !strings.Contains(answer, want) {
			t.Fatalf("%s answered %q in %v, want the code %s", label, answer, took, want)
		}
		conversation = append(turn, map[string]any{"role": "assistant", "content": answer})
		t.Logf("%s: %q, a prompt of %d tokens, in %v", label, answer, prompt, took)
		return prompt
	}
	corpus := longContextCorpus(t, articles)
	middle := longContextCodes[2]
	if prompt := ask("first read", corpus+"\n\nQuestion: "+question(middle.room), middle.code); prompt < longContextFewestTokens {
		t.Fatalf("the first read counted %d prompt tokens, fewer than the %d a long context takes", prompt, longContextFewestTokens)
	}
	for _, index := range []int{0, 4} {
		ask("follow-up "+longContextCodes[index].room, question(longContextCodes[index].room), longContextCodes[index].code)
	}
	chunks, took := server.cancelled(append(append([]map[string]any(nil), conversation...), map[string]any{"role": "user", "content": "Summarize every article above in detail."}))
	if chunks < longContextCancelChunks {
		t.Fatalf("the cancelled generation streamed %d chunks in %v, fewer than the %d it is cancelled at", chunks, took, longContextCancelChunks)
	}
	t.Logf("cancelled generation: %d chunks in %v", chunks, took)
	ask("after the cancellation "+longContextCodes[1].room, question(longContextCodes[1].room), longContextCodes[1].code)
	if answer, _, took := server.answer(image, imageTokens); answer == "" {
		t.Fatalf("the image beside the context answered nothing in %v", took)
	} else {
		t.Logf("image beside the context: %.120q in %v", answer, took)
	}
	ask("after the image "+longContextCodes[3].room, question(longContextCodes[3].room), longContextCodes[3].code)
	peak, total := server.memory()
	if peak == 0 || total == 0 || peak > total {
		t.Fatalf("the card's peak %d bytes against its %d total, want a peak reported under the total", peak, total)
	}
	t.Logf("card peak %d MiB of %d MiB", peak>>20, total>>20)
}

// flashNextServable is the registered Flash-Next model the store serves for inference, by the discovery the server's
// own catalog reads: its name and its on-disk location. A store that registers none cannot apply the test and says so.
func flashNextServable(t *testing.T, ctx context.Context, store string) (name, location string) {
	t.Helper()
	if _, err := os.Stat(store); err != nil {
		testskip.NotApplicable(t, "no store at "+store)
	}
	db, err := overgodb.OpenReadOnly(store)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	entries, _, err := discovery.CapabilityCatalog(ctx, db, longContextCatalogBound, discovery.LoadMemo(ctx, db))
	if err != nil {
		t.Fatal(err)
	}
	for _, entry := range entries {
		inference := false
		for _, capability := range entry.Capabilities {
			inference = inference || capability.Task == recipe.TaskInference && capability.Stale == ""
		}
		if inference && entry.Present && strings.Contains(filepath.Base(entry.Location), longContextModel) {
			return filepath.Base(entry.Location), entry.Location
		}
	}
	testskip.NotApplicable(t, "no servable "+longContextModel+" model is registered here")
	return "", ""
}

// longContextCorpus reads the articles in order until the characters are read, placing each code at its share.
func longContextCorpus(t *testing.T, articles string) string {
	t.Helper()
	var corpus strings.Builder
	placed := 0
	for index := 0; corpus.Len() < longContextChars; index++ {
		data, err := os.ReadFile(filepath.Join(articles, fmt.Sprintf("%05d.txt", index)))
		if err != nil {
			t.Fatal(err)
		}
		for placed < len(longContextCodes) && float64(corpus.Len()) >= longContextCodes[placed].at*longContextChars {
			fmt.Fprintf(&corpus, "\nThe archive code for the %s Room is %s.\n\n", longContextCodes[placed].room, longContextCodes[placed].code)
			placed++
		}
		corpus.Write(data)
		corpus.WriteString("\n\n")
	}
	for ; placed < len(longContextCodes); placed++ {
		fmt.Fprintf(&corpus, "\nThe archive code for the %s Room is %s.\n\n", longContextCodes[placed].room, longContextCodes[placed].code)
	}
	return corpus.String()
}

// frozenCase returns the messages and the answer tokens of one case of a frozen comparison file.
func frozenCase(path, id string) ([]map[string]any, int, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, 0, err
	}
	var document struct {
		Cases []struct {
			ID        string           `json:"id"`
			Messages  []map[string]any `json:"messages"`
			MaxTokens int              `json:"max_tokens"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &document); err != nil {
		return nil, 0, fmt.Errorf("%s: %w", path, err)
	}
	for _, frozen := range document.Cases {
		if frozen.ID == id {
			return frozen.Messages, frozen.MaxTokens, nil
		}
	}
	return nil, 0, fmt.Errorf("%s holds no case %q", path, id)
}

// longContextServer talks to the served model's chat completions.
type longContextServer struct {
	t     *testing.T
	ctx   context.Context
	url   string
	model string
}

func (s *longContextServer) request(ctx context.Context, messages []map[string]any, maxTokens int, stream bool) *http.Response {
	s.t.Helper()
	body, err := json.Marshal(map[string]any{
		"model": s.model, "messages": messages, "temperature": 0, "max_tokens": maxTokens, "stream": stream,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	if err != nil {
		s.t.Fatal(err)
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, s.url+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		s.t.Fatal(err)
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		s.t.Fatalf("the chat request did not answer: %v", err)
	}
	return response
}

// answer asks and returns the answer's text, its prompt's tokens and the request's wall time.
func (s *longContextServer) answer(messages []map[string]any, maxTokens int) (string, int, time.Duration) {
	s.t.Helper()
	started := time.Now()
	response := s.request(s.ctx, messages, maxTokens, false)
	defer response.Body.Close()
	body, err := io.ReadAll(response.Body)
	if err != nil || response.StatusCode != http.StatusOK {
		s.t.Fatalf("the chat request answered %d (%v): %.300s", response.StatusCode, err, body)
	}
	var parsed struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens int `json:"prompt_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &parsed); err != nil || len(parsed.Choices) != 1 {
		s.t.Fatalf("the chat answer holds %d choices (%v): %.300s", len(parsed.Choices), err, body)
	}
	return strings.TrimSpace(parsed.Choices[0].Message.Content), parsed.Usage.PromptTokens, time.Since(started)
}

// cancelled streams a generation, cancels it once it has streamed its chunks and returns the chunks streamed in all.
func (s *longContextServer) cancelled(messages []map[string]any) (int, time.Duration) {
	s.t.Helper()
	ctx, cancel := context.WithCancel(s.ctx)
	defer cancel()
	started := time.Now()
	response := s.request(ctx, messages, longContextSummaryTokens, true)
	defer response.Body.Close()
	chunks, lines := 0, bufio.NewScanner(response.Body)
	lines.Buffer(make([]byte, longContextLineBytes), longContextLineBytes)
	for lines.Scan() {
		if strings.HasPrefix(lines.Text(), "data: {") {
			if chunks++; chunks == longContextCancelChunks {
				cancel()
			}
		}
	}
	if err := lines.Err(); err != nil && !errors.Is(err, context.Canceled) && !errors.Is(ctx.Err(), context.Canceled) {
		s.t.Fatalf("the cancelled stream ended with %v", err)
	}
	return chunks, time.Since(started)
}

// memory reads the server's peak card bytes and the device's total from its metrics.
func (s *longContextServer) memory() (peak, total uint64) {
	s.t.Helper()
	request, err := http.NewRequestWithContext(s.ctx, http.MethodGet, s.url+"/metrics", nil)
	if err != nil {
		s.t.Fatal(err)
	}
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		s.t.Fatal(err)
	}
	defer response.Body.Close()
	lines := bufio.NewScanner(response.Body)
	for lines.Scan() {
		name, value, found := strings.Cut(lines.Text(), " ")
		if !found {
			continue
		}
		parsed, err := strconv.ParseUint(strings.TrimSpace(value), 10, 64)
		if err != nil {
			continue
		}
		switch name {
		case "overgo_cuda_memory_peak_bytes":
			peak = parsed
		case "overgo_device_memory_total_bytes":
			total = parsed
		}
	}
	return peak, total
}
