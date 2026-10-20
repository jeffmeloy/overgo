package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
)

// strataServed is a served model off the device as the suite meets it over
// HTTP: it counts a word as a token beside a template of templateTokens,
// reuses the tokens of the messages its last request shares with this one,
// as a conversation cache does, and streams a reply of a token a word.
type strataServed struct {
	mu       sync.Mutex
	requests []strataServedRequest
}

type strataServedRequest struct {
	Model    string            `json:"model"`
	Stream   bool              `json:"stream"`
	Tokens   int               `json:"max_tokens"`
	Greedy   *float64          `json:"temperature"`
	Messages []strataMessage   `json:"messages"`
	Tools    []json.RawMessage `json:"tools"`
	Template map[string]bool   `json:"chat_template_kwargs"`
	reply    string
}

const (
	templateTokens = 10
	servedPrefill  = 1000.0
	servedDecode   = 50.0
)

func wordsOf(messages []strataMessage) int {
	words := templateTokens
	for _, message := range messages {
		words += len(strings.Fields(message["content"]))
	}
	return words
}

func (served *strataServed) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	var body strataServedRequest
	if request.URL.Path != "/v1/chat/completions" || json.NewDecoder(request.Body).Decode(&body) != nil {
		http.Error(response, "unknown request", http.StatusBadRequest)
		return
	}
	served.mu.Lock()
	defer served.mu.Unlock()
	prompt, cached := wordsOf(body.Messages), 0
	if len(served.requests) > 0 {
		last := served.requests[len(served.requests)-1]
		held := append(slices.Clone(last.Messages), strataMessage{"role": "assistant", "content": last.reply})
		shared := 0
		for shared < min(len(held), len(body.Messages)) && fmt.Sprint(held[shared]) == fmt.Sprint(body.Messages[shared]) {
			shared++
		}
		if shared > 0 {
			cached = wordsOf(held[:shared])
		}
	}
	words := make([]string, min(body.Tokens, len(served.requests)%7+3))
	for index := range words {
		words[index] = fmt.Sprintf("w%d", len(served.requests)*100+index)
	}
	body.reply = strings.Join(words, " ")
	served.requests = append(served.requests, body)
	response.Header().Set("Content-Type", "text/event-stream")
	write := func(value any) {
		data, _ := json.Marshal(value)
		fmt.Fprintf(response, "data: %s\n\n", data)
	}
	write(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"role": "assistant"}}}})
	for index, word := range words {
		if index > 0 {
			word = " " + word
		}
		write(map[string]any{"choices": []any{map[string]any{"delta": map[string]string{"content": word}}}})
	}
	write(map[string]any{
		"choices": []any{map[string]any{"delta": map[string]string{}, "finish_reason": "length"}},
		"usage":   map[string]any{"prompt_tokens": prompt, "completion_tokens": len(words), "prompt_tokens_details": map[string]int{"cached_tokens": cached}},
		"timings": map[string]any{"cache_n": cached, "prompt_n": prompt - cached, "prompt_per_second": servedPrefill, "predicted_per_second": servedDecode},
	})
	fmt.Fprint(response, "data: [DONE]\n\n")
}

// TestStrataSuiteAgainstAServedModel holds benchmark strata to Strata's frozen
// suite against a served model off the device: the warm-up and calibration,
// three runs of the three short questions at 100 new tokens, three fresh
// prompts sized to 4K and 16K tokens at 24, and three agent sessions of five
// file turns at 64 with the four tools, each turn resending the conversation
// with the model's own answers; every request streamed, greedy and with
// thinking off. The medians come from the engine's timings and the reuse from
// its usage, as the reference reads them.
func TestStrataSuiteAgainstAServedModel(t *testing.T) {
	served := &strataServed{}
	server := httptest.NewServer(served)
	defer server.Close()
	docs := t.TempDir()
	write := func(name string, words int) {
		path := filepath.Join(docs, filepath.FromSlash(name))
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(strings.Repeat(name+" word\n", words/2)), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	write("README.md", 2400)
	write("docs/HOW_IT_WORKS.md", 800)
	write("docs/DETAILS.md", 100)
	var output bytes.Buffer
	if err := runStrata([]string{server.URL + "/", docs}, &output); err != nil {
		t.Fatal(err)
	}

	// Four warm-up requests, nine short questions, six fresh prompts and three sessions of six turns.
	if len(served.requests) != 4+9+6+18 {
		t.Fatalf("%d requests, want 37", len(served.requests))
	}
	for index, request := range served.requests {
		if !request.Stream || request.Greedy == nil || *request.Greedy != 0 || request.Template["enable_thinking"] || request.Model != "" {
			t.Fatalf("request %d %+v is not a streamed greedy request with thinking off", index, request)
		}
	}
	chat, prompts, agent := served.requests[4:13], served.requests[13:19], served.requests[19:]
	for index, request := range chat {
		if request.Tokens != strataChatTokens || !strings.HasSuffix(request.Messages[0]["content"], strataQuestions[index%3]) {
			t.Fatalf("short chat %d = %+v", index, request)
		}
	}
	for index, request := range prompts {
		want := []float64{4096, 16384}[index/3]
		if got := float64(wordsOf(request.Messages)); request.Tokens != strataPromptTokens || math.Abs(got-want) > 4 ||
			!strings.Contains(request.Messages[0]["content"], "Here is a list of words:\n") {
			t.Fatalf("fresh prompt %d holds %g tokens for %d new, want about %g for %d", index, got, request.Tokens, want, strataPromptTokens)
		}
	}
	files := []string{"File README.md:\n", "File docs/HOW_IT_WORKS.md:\n", "File generated/part0.txt:\n", "File generated/part1.txt:\n", "File generated/part2.txt:\n"}
	for index, request := range agent {
		turn := index % 6
		if request.Tokens != strataAgentTokens || len(request.Tools) != 4 || len(request.Messages) != 2+2*turn ||
			request.Messages[0]["role"] != "system" || !strings.HasPrefix(request.Messages[0]["content"], strataSystem) {
			t.Fatalf("agent turn %d = %d messages, %d tools", index, len(request.Messages), len(request.Tools))
		}
		if turn > 0 {
			added := request.Messages[len(request.Messages)-1]["content"]
			if !strings.HasPrefix(added, files[turn-1]) || turn == 1 && len(strings.Fields(added)) != strataAgentWords+9 {
				t.Fatalf("agent turn %d adds %.40q", index, added)
			}
			if answer := request.Messages[len(request.Messages)-2]; answer["content"] != agent[index-1].reply || answer["role"] != "assistant" {
				t.Fatalf("agent turn %d does not continue from the model's answer %q", index, agent[index-1].reply)
			}
		}
	}

	var result struct {
		Chat    map[string]json.RawMessage
		Prompts map[string]map[string]json.RawMessage
		Agent   map[string]json.RawMessage
	}
	if err := json.Unmarshal(output.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	median := func(raw json.RawMessage) strataStat {
		t.Helper()
		var stat strataStat
		if err := json.Unmarshal(raw, &stat); err != nil {
			t.Fatal(err)
		}
		return stat
	}
	var turns map[string]json.RawMessage
	if err := json.Unmarshal(result.Agent["turns"], &turns); err != nil {
		t.Fatal(err)
	}
	if decode, first := median(result.Chat["decode_tps"]), median(result.Chat["ttft_s"]); decode.Median != servedDecode || decode.N != 9 || first.Median <= 0 {
		t.Fatalf("short chat decode %+v, first token %+v", decode, first)
	}
	if prefill, reused := median(result.Prompts["16K"]["prompt_tps"]), median(result.Prompts["4K"]["reused_tokens"]); prefill.Median != servedPrefill || prefill.N != 3 || reused.Max != 0 {
		t.Fatalf("fresh prompts prefill %+v, reused %+v", prefill, reused)
	}
	hit, reread, last := median(result.Agent["hit_pct"]), median(result.Agent["reread_tokens"]), median(result.Agent["last_turn_prompt_tokens"])
	if hit.N != 3 || hit.Median <= 0 || hit.Median >= strataPercent || reread.Max != 0 || last.Median <= 0 || median(turns["decode_tps"]).N != 15 {
		t.Fatalf("agent hit %+v, read again %+v, last turn %+v", hit, reread, last)
	}
}
