package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
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
// as a conversation cache does, and streams a reply of a token a word. The
// request at an index breaks names breaks its stream after the reply's first
// word: "eof" ends it there, "malformed" sends data that is no JSON and then
// finishes, "oversize" sends a line longer than a reader holds, and "error"
// sends an error event.
type strataServed struct {
	mu       sync.Mutex
	requests []strataServedRequest
	breaks   map[int]string
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
	breaks := served.breaks[len(served.requests)]
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
		if index > 0 {
			continue
		}
		switch breaks {
		case "eof":
			return
		case "malformed":
			fmt.Fprint(response, "data: {\"choices\": [\n\n")
		case "oversize":
			fmt.Fprintf(response, "data: %s\n\n", strings.Repeat("x", bufio.MaxScanTokenSize))
		case "error":
			write(map[string]any{"error": map[string]string{"message": "the server stopped"}})
		}
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
// its usage, as the reference reads them. Each measured request is kept as it
// completes, a line of JSON naming its part, in the order it was sent.
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
	var output, kept bytes.Buffer
	if err := runStrata([]string{server.URL + "/", docs}, &output, &kept); err != nil {
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

	var parts []string
	for line := range strings.Lines(kept.String()) {
		var request struct {
			Part      string  `json:"part"`
			OK        bool    `json:"ok"`
			PromptTPS float64 `json:"prompt_tps"`
		}
		if err := json.Unmarshal([]byte(line), &request); err != nil || !request.OK || request.PromptTPS != servedPrefill {
			t.Fatalf("kept request %q = %+v, %v; want an answered request with the engine's prefill rate", line, request, err)
		}
		parts = append(parts, request.Part)
	}
	want := slices.Concat(slices.Repeat([]string{"chat"}, 9), slices.Repeat([]string{"4K"}, 3), slices.Repeat([]string{"16K"}, 3), slices.Repeat([]string{"agent"}, 18))
	if !slices.Equal(parts, want) {
		t.Fatalf("kept parts %q, want %q", parts, want)
	}
}

// TestStrataRefusesUnfinishedStreams holds a request to the stream that
// answered it: after the reply's first word, a stream cut short, data that is
// no JSON, a line longer than the reader holds and an error event each leave
// the request unanswered with why, and the finished request after it is
// answered. An agent session stops at its unanswered turn, which is kept as
// such and joins neither the medians nor a next turn.
func TestStrataRefusesUnfinishedStreams(t *testing.T) {
	question := []strataMessage{{"role": "user", "content": "q"}}
	for _, mode := range []string{"eof", "malformed", "oversize", "error"} {
		served := &strataServed{breaks: map[int]string{0: mode}}
		server := httptest.NewServer(served)
		suite := &strataSuite{url: server.URL, nonce: mode, kept: io.Discard}
		broken, finished := suite.chat(question, strataChatTokens, false), suite.chat(question, strataChatTokens, false)
		server.Close()
		if broken.OK || broken.Err == "" || !finished.OK {
			t.Fatalf("%s: requests %+v then %+v, want the first unanswered with why and the second answered", mode, broken, finished)
		}
	}
	// The third request, the first session's second file turn, is cut short.
	served := &strataServed{breaks: map[int]string{2: "eof"}}
	server := httptest.NewServer(served)
	defer server.Close()
	var kept bytes.Buffer
	suite := &strataSuite{url: server.URL, nonce: "agent", kept: &kept}
	result := suite.agent(strataFiles(""))
	if sent := len(served.requests); sent != 3+2*(strataAgentTurns+1) {
		t.Fatalf("%d requests, want the cut session's three and two sessions of %d", sent, strataAgentTurns+1)
	}
	if next := served.requests[3].Messages; len(next) != 2 {
		t.Fatalf("the request after the cut turn holds %d messages, want a new session's two", len(next))
	}
	if decode := result["turns"].(map[string]any)["decode_tps"].(*strataStat); decode.N != 1+2*strataAgentTurns {
		t.Fatalf("the follow-up turns' decode median is of %d, want the cut session's one answered and two sessions' %d", decode.N, strataAgentTurns)
	}
	if unanswered := strings.Count(kept.String(), `"ok":false`); unanswered != 1 {
		t.Fatalf("%d unanswered requests kept, want the cut turn", unanswered)
	}
}
