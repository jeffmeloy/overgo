package main

import (
	"bufio"
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"time"

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/processmeasure"
)

// Strata's frozen bench, suite 1 of its tools/strata_bench.py at 296b59d2:
// the runs of each part and the new tokens of each request; the warm-up's
// short answer and 1K prompt; the two sizes of fresh prompt (word lists,
// evaluation.WordList); the word counts whose prompt tokens calibrate
// the tokens per word, the fewest words of a prompt and the fallbacks where
// calibration measures nothing; the agent session's turns and the words of
// the file each adds, of which a document must hold half to be read; a
// session's share of reused tokens in percent; and the command's arguments,
// the URL and the documents' folder.
const (
	strataRuns          = 3
	strataChatTokens    = 100
	strataPromptTokens  = 24
	strataAgentTokens   = 64
	strataWarmTokens    = 24
	strataPrimeTokens   = 8
	strataPrimeSize     = 1024
	strataShortPrompt   = 4096
	strataLongPrompt    = 16384
	strataCalibrateLow  = 200
	strataCalibrateHigh = 800
	strataMinimumWords  = 50
	strataPerWord       = 1.1
	strataOverhead      = 40
	strataAgentTurns    = 5
	strataAgentWords    = 1500
	strataHalfTurn      = strataAgentWords / 2
	strataPercent       = 100
	strataArguments     = 2
)

// The reference's three short questions (ab_engine.PROMPTS), and its agent
// session's system prompt, opening request, four tools (agent_turns_bench)
// and the documents of its own folder the session reads.
var (
	strataQuestions = []string{
		"Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
		"Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
		"List ten practical tips for learning a new language as an adult, one sentence each.",
	}
	strataDocs = []string{"README.md", "docs/HOW_IT_WORKS.md", "docs/DETAILS.md", "docs/INSTALL.md", "docs/MODELS.md", "docs/COMMUNITY_BENCHMARKS.md"}
)

const (
	strataSystem = "You are an autonomous coding agent working in the user's repository. You can read files, search the code, " +
		"edit files and run commands with the tools below. Work step by step: read the code you need before changing " +
		"it, make small and correct edits, keep the existing style, and explain briefly what you did. Never invent " +
		"file contents: read them. When the task is done, give a short summary of the changes and how to test them."
	strataIntro = "I will show you files from a project one at a time. After each file, answer in one or two sentences what it is for. Do not call tools."
	strataTools = `[{"type": "function", "function": {"name": "read_file", "description": "Read a file of the repository (UTF-8 text).", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path from the repository root"}}, "required": ["path"]}}}, ` +
		`{"type": "function", "function": {"name": "grep", "description": "Search the repository for a regular expression; returns matching lines.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}}, ` +
		`{"type": "function", "function": {"name": "edit_file", "description": "Replace an exact string in a file with a new one.", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}}, "required": ["path", "old", "new"]}}}, ` +
		`{"type": "function", "function": {"name": "run", "description": "Run a shell command in the repository and return its output.", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]`
)

type strataMessage = map[string]string

// strataFile is a file an agent turn adds: its name and its words.
type strataFile struct{ name, text string }

// strataRequest is one request of the suite and what it measured, as the
// reference digests it: the engine's own counts and speeds, which an Overgo
// answer always reports, and the first token as the client sees it. Index is
// the question or turn.
type strataRequest struct {
	OK        bool    `json:"ok"`
	Err       string  `json:"err,omitzero"`
	TTFT      float64 `json:"ttft_s"`
	Total     float64 `json:"total_s"`
	Prompt    int     `json:"prompt_tokens"`
	Reused    int     `json:"reused_tokens"`
	Read      int     `json:"read_tokens"`
	Generated int     `json:"generated_tokens"`
	PromptTPS float64 `json:"prompt_tps"`
	DecodeTPS float64 `json:"decode_tps"`
	Run       int     `json:"run"`
	Index     int     `json:"index"`
	text      string
}

// strataChunk is what the suite reads of a streamed chunk; usage and timings
// come with the last.
type strataChunk struct {
	Error   json.RawMessage
	Choices []struct {
		Delta struct {
			Content   string
			Reasoning string `json:"reasoning_content"`
		}
	}
	Usage struct {
		Prompt     int `json:"prompt_tokens"`
		Completion int `json:"completion_tokens"`
		Details    struct {
			Cached int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	}
	Timings struct {
		Cached    int     `json:"cache_n"`
		Read      int     `json:"prompt_n"`
		PromptTPS float64 `json:"prompt_per_second"`
		DecodeTPS float64 `json:"predicted_per_second"`
	}
}

// strataStat is a measure's median and range over the requests that
// measured it, as the reference reports them.
type strataStat struct {
	Median float64 `json:"median"`
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	N      int     `json:"n"`
}

// strataSuite sends the suite to one served model: every prompt opens with a
// tag holding the nonce, so no conversation cache answers a fresh prompt.
type strataSuite struct {
	url, nonce        string
	perWord, overhead float64
}

// runStrata sends Strata's frozen bench to the HTTP API at the URL and writes
// every request with the medians, under the reference's names. The agent
// session reads the documents of the folder named after it, as the reference
// reads its own folder's, and generated word lists without one.
func runStrata(args []string, output io.Writer) error {
	if len(args) == 0 || len(args) > strataArguments {
		return errors.New("usage: benchmark strata URL [FOLDER]")
	}
	suite := &strataSuite{url: strings.TrimSuffix(args[0], "/"), nonce: fmt.Sprint(time.Now().Unix())}
	return clioptions.WritePrettyJSON(output, suite.run(append(args, "")[1]))
}

// run warms the model up unmeasured (a short answer, the calibration and a
// 1K prompt), then measures the short chat, the fresh 4K and 16K prompts and
// the agent sessions.
func (s *strataSuite) run(docs string) map[string]any {
	user := func(content string) []strataMessage { return []strataMessage{{"role": "user", "content": content}} }
	s.chat(user(strataQuestions[0]), strataWarmTokens, false)
	low := s.chat(user(fmt.Sprintf("[cal-%d-%s]\n%s", strataCalibrateLow, s.nonce, evaluation.WordList(strataCalibrateLow))), 1, false).Prompt
	high := s.chat(user(fmt.Sprintf("[cal-%d-%s]\n%s", strataCalibrateHigh, s.nonce, evaluation.WordList(strataCalibrateHigh))), 1, false).Prompt
	s.perWord, s.overhead = strataPerWord, strataOverhead
	if low > 0 && high > low {
		s.perWord = float64(high-low) / (strataCalibrateHigh - strataCalibrateLow)
		s.overhead = max(0, float64(low)-strataCalibrateLow*s.perWord)
	}
	s.chat(user(s.prompt(strataPrimeSize, -1)), strataPrimeTokens, false)
	var chat []strataRequest
	for run := range strataRuns {
		for index, question := range strataQuestions {
			request := s.chat(user(fmt.Sprintf("[%d-%d-c-%s] %s", run, index, s.nonce, question)), strataChatTokens, false)
			request.Run, request.Index = run, index
			chat = append(chat, request)
		}
	}
	prompts := map[string]any{}
	for _, size := range []int{strataShortPrompt, strataLongPrompt} {
		var requests []strataRequest
		for run := range strataRuns {
			request := s.chat(user(s.prompt(size, run)), strataPromptTokens, false)
			request.Run = run
			requests = append(requests, request)
		}
		prompts[strconv.Itoa(size/strataPrimeSize)+"K"] = strataSummary(requests)
	}
	return map[string]any{"chat": strataSummary(chat), "prompts": prompts, "agent": s.agent(strataFiles(docs))}
}

// prompt is a fresh prompt of about the given tokens: a tagged word list of
// the calibrated words.
func (s *strataSuite) prompt(tokens, run int) string {
	words := max(strataMinimumWords, int((float64(tokens)-s.overhead)/s.perWord))
	return fmt.Sprintf("[bench %d-%d-p-%s]\n%s", run, tokens, s.nonce, evaluation.WordList(words))
}

// agent runs the agent sessions: the system prompt and four tools, then a
// turn per file continuing from the model's own answers, as an agent client
// resends a conversation. Of the follow-up turns it gives the medians, the
// share of prompt tokens the server reused, the earlier tokens it read again
// and the files' tokens it read once.
func (s *strataSuite) agent(files []strataFile) map[string]any {
	var sessions [][]strataRequest
	var follow []strataRequest
	var hit, again, fresh, last, first []float64
	for run := range strataRuns {
		messages := []strataMessage{
			{"role": "system", "content": fmt.Sprintf("%s\n(session %d-a-%s)", strataSystem, run, s.nonce)},
			{"role": "user", "content": strataIntro},
		}
		var turns []strataRequest
		for turn := range strataAgentTurns + 1 {
			if turn > 0 {
				messages = append(messages, strataMessage{"role": "user", "content": fmt.Sprintf("File %s:\n%s\n\nWhat is this file for? Two sentences.", files[turn-1].name, files[turn-1].text)})
			}
			request := s.chat(messages, strataAgentTokens, true)
			request.Run, request.Index = run, turn
			if turns = append(turns, request); !request.OK {
				break
			}
			messages = append(messages, strataMessage{"role": "assistant", "content": cmp.Or(request.text, "Understood.")})
		}
		sessions = append(sessions, turns)
		if turns[0].OK {
			first = append(first, turns[0].TTFT)
		}
		// A session stops at its first unanswered turn.
		if !turns[len(turns)-1].OK {
			turns = turns[:len(turns)-1]
		}
		if len(turns) <= 1 {
			continue
		}
		reused, prompted, read, reread := 0, 0, 0, 0
		for index, turn := range turns[1:] {
			reused, prompted, read = reused+turn.Reused, prompted+turn.Prompt, read+turn.Read
			reread += max(0, turns[index].Prompt+turns[index].Generated-turn.Reused)
		}
		follow = append(follow, turns[1:]...)
		hit = append(hit, float64(reused)/float64(max(1, prompted))*strataPercent)
		again, fresh = append(again, float64(reread)), append(fresh, float64(read-reread))
		last = append(last, float64(turns[len(turns)-1].Prompt))
	}
	return map[string]any{
		"sessions": sessions, "turns": strataSummary(follow), "hit_pct": strataStats(hit), "reread_tokens": strataStats(again),
		"new_tokens": strataStats(fresh), "last_turn_prompt_tokens": strataStats(last), "first_turn_ttft_s": strataStats(first),
	}
}

// chat sends one streamed, greedy request with thinking off and digests its
// answer.
func (s *strataSuite) chat(messages []strataMessage, maxTokens int, tools bool) strataRequest {
	body := map[string]any{
		"stream": true, "max_tokens": maxTokens, "temperature": 0, "messages": messages,
		"chat_template_kwargs": map[string]bool{"enable_thinking": false},
	}
	if tools {
		body["tools"] = json.RawMessage(strataTools)
	}
	data, _ := json.Marshal(body)
	var request strataRequest
	var chunk strataChunk
	var text strings.Builder
	pieces, clock := 0, processmeasure.NewStopwatch()
	var walls processmeasure.Walls
	response, err := http.Post(s.url+"/v1/chat/completions", "application/json", bytes.NewReader(data))
	if err == nil {
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			err = errors.New(response.Status)
		}
		for lines := bufio.NewScanner(response.Body); err == nil && lines.Scan(); {
			payload, found := strings.CutPrefix(lines.Text(), "data:")
			chunk.Choices = nil
			if !found || json.Unmarshal([]byte(strings.TrimSpace(payload)), &chunk) != nil {
				continue
			}
			for _, choice := range chunk.Choices {
				if choice.Delta.Content+choice.Delta.Reasoning != "" {
					if pieces++; pieces == 1 {
						request.TTFT = time.Duration(walls.Elapsed(clock)).Seconds()
					}
					text.WriteString(choice.Delta.Content)
				}
			}
			if chunk.Error != nil {
				err = errors.New(string(chunk.Error))
			}
		}
	}
	request.Total = time.Duration(walls.Elapsed(clock)).Seconds()
	request.Prompt, request.Generated = chunk.Usage.Prompt, cmp.Or(chunk.Usage.Completion, pieces)
	request.Reused = cmp.Or(chunk.Usage.Details.Cached, chunk.Timings.Cached)
	request.Read = cmp.Or(chunk.Timings.Read, request.Prompt-request.Reused)
	request.PromptTPS, request.DecodeTPS = chunk.Timings.PromptTPS, chunk.Timings.DecodeTPS
	request.OK, request.text = err == nil && pieces > 0, text.String()
	if err != nil {
		request.Err = err.Error()
	}
	return request
}

// strataSummary gives a part's requests with the medians of those that
// answered; a speed of zero is one the engine did not measure.
func strataSummary(requests []strataRequest) map[string]any {
	var decode, prompt, ttft, tokens, reused, generated []float64
	for _, request := range requests {
		if !request.OK {
			continue
		}
		ttft, tokens = append(ttft, request.TTFT), append(tokens, float64(request.Prompt))
		reused, generated = append(reused, float64(request.Reused)), append(generated, float64(request.Generated))
		if request.DecodeTPS > 0 {
			decode = append(decode, request.DecodeTPS)
		}
		if request.PromptTPS > 0 {
			prompt = append(prompt, request.PromptTPS)
		}
	}
	return map[string]any{
		"requests": requests, "decode_tps": strataStats(decode), "prompt_tps": strataStats(prompt), "ttft_s": strataStats(ttft),
		"prompt_tokens": strataStats(tokens), "reused_tokens": strataStats(reused), "generated_tokens": strataStats(generated),
	}
}

// strataStats is the median and range of the values, none of none.
func strataStats(values []float64) *strataStat {
	if len(values) == 0 {
		return nil
	}
	slices.Sort(values)
	return &strataStat{Median: median(values), Min: values[0], Max: values[len(values)-1], N: len(values)}
}

// strataFiles are the files the agent session adds: the first words of each
// document under docs that holds half a turn's words, then generated word
// lists, as the reference reads its own folder.
func strataFiles(docs string) []strataFile {
	var files []strataFile
	for _, name := range strataDocs {
		if docs == "" {
			break
		}
		data, err := os.ReadFile(filepath.Join(docs, filepath.FromSlash(name)))
		if words := strings.Fields(string(data)); err == nil && len(words) >= strataHalfTurn {
			files = append(files, strataFile{name, strings.Join(words[:min(len(words), strataAgentWords)], " ")})
		}
	}
	for index := 0; len(files) < strataAgentTurns; index++ {
		files = append(files, strataFile{fmt.Sprintf("generated/part%d.txt", index), evaluation.WordList(strataAgentWords)})
	}
	return files[:strataAgentTurns]
}
