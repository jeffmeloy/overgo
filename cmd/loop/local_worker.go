package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
)

// A local worker's budgets: what it may spend on one step before the driver hands the step on.
const (
	// localWorkerTurns is the most replies the model gives one step: a step it has not passed in that many is the next
	// worker's.
	localWorkerTurns = 40
	// localReplyTokens bounds one reply: a tool call or a short answer, the model's thinking left off.
	localReplyTokens = 4096
	// localThinkingTokens bounds one reply the model thinks before: room to reason, then the call or the answer.
	localThinkingTokens = 16384
	// localResultChars is the most of a tool's result the model is shown; the rest is cut and said to be.
	localResultChars = 24000
	// localRepeatedReplies is how many replies in a row of repeated readings alone end an attempt.
	localRepeatedReplies = 3
)

// readingTools are the tools whose result an identical call repeats while nothing is edited.
var readingTools = []string{"read_file", "read_lines", "search_files", "show_step"}

// localInstruction opens a local worker's conversation in place of a prompt written for a client: the nine tools are
// its only hands and the step's check its end.
const localInstruction = "You work one step of a plan in a Go checkout, through the tools alone: there is no shell. " +
	"Read the step with show_step and run its check with run_check to see what fails. Find what the failure names with " +
	"search_files, and read only the lines you need with read_lines; read a whole file with read_file only when it is short. " +
	"Your replies are few, so change what the step asks with edit_file, or remove a file with delete_file, as soon as you know where, then run run_check again, " +
	"until it passes. Then answer, with no tool call, in one line saying what you changed."

// chatMessage is one message of an OpenAI chat conversation, as overgo's chat route reads and writes it.
type chatMessage struct {
	Role       string     `json:"role"`
	Content    string     `json:"content"`
	ToolCalls  []chatCall `json:"tool_calls,omitempty"`
	ToolCallID string     `json:"tool_call_id,omitempty"`
}

// chatCall is a tool call a reply makes, its arguments a JSON object written as text.
type chatCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

// localWork is what a local worker did with a step: its replies and the tool calls it made.
type localWork struct {
	Turns, Calls int
	Answer       string
}

// workerModel is the model a local worker is served: its name, and whether it thinks before each reply. Thinking is off
// unless asked: replies are fewer tokens then, and the owner has it switched on for a run to see what it buys
// (2026-11-10).
type workerModel struct {
	name     string
	thinking bool
}

// workLocally works prompt with the model served at server under model, through tools: each tool call the model makes
// is run and its result sent back, until the model answers with no call or spends localWorkerTurns. Each call is
// written to log as it is made.
func workLocally(ctx context.Context, server string, model workerModel, prompt string, tools []workerTool, log io.Writer) (localWork, error) {
	listing := make([]map[string]any, len(tools))
	for index, tool := range tools {
		listing[index] = map[string]any{"type": "function", "function": map[string]any{"name": tool.name, "description": tool.description, "parameters": tool.schema()}}
	}
	messages := []chatMessage{{Role: "system", Content: localInstruction}, {Role: "user", Content: prompt}}
	var work localWork
	// A reading call the model made before, its name and arguments alike, is not run again while no edit has changed
	// the checkout: its result stands above. A model drawn greedily that repeats a call repeats it until its replies run
	// out (Qwen3.8-27B searched the same four texts over 50 times on one step, 2026-11-09), so localRepeatedReplies
	// replies of repeats alone end the attempt, and the next attempt or worker takes the step.
	asked, repeating := map[string]bool{}, 0
	for work.Turns < localWorkerTurns {
		reply, err := askChat(ctx, server, model, messages, listing)
		if err != nil {
			return work, err
		}
		work.Turns++
		messages = append(messages, reply)
		if len(reply.ToolCalls) == 0 {
			work.Answer = strings.TrimSpace(reply.Content)
			fmt.Fprintf(log, "local worker: answered after %d replies and %d tool calls: %s\n", work.Turns, work.Calls, work.Answer)
			return work, nil
		}
		repeats := 0
		for _, call := range reply.ToolCalls {
			work.Calls++
			var arguments map[string]any
			result := ""
			key := call.Function.Name + " " + call.Function.Arguments
			switch {
			case slices.Contains(readingTools, call.Function.Name) && asked[key]:
				repeats++
				result = "this exact call was made already and nothing has been edited since: its result is above. Edit, or answer."
			case json.Unmarshal([]byte(call.Function.Arguments), &arguments) != nil:
				result = "the arguments are not one JSON object"
			default:
				text, err := callTool(tools, call.Function.Name, arguments)
				result = text
				if err != nil {
					result = errors.Join(errors.New(text), err).Error()
				}
				asked[key] = true
				if (call.Function.Name == "edit_file" || call.Function.Name == "delete_file") && err == nil {
					// The checkout changed, so every reading may differ now.
					asked = map[string]bool{}
				}
			}
			fmt.Fprintf(log, "local worker: %s %.200s\n", call.Function.Name, call.Function.Arguments)
			if len(result) > localResultChars {
				result = result[:localResultChars] + fmt.Sprintf("\n[%d more characters cut]", len(result)-localResultChars)
			}
			messages = append(messages, chatMessage{Role: "tool", ToolCallID: call.ID, Content: result})
		}
		repeating++
		if repeats < len(reply.ToolCalls) {
			repeating = 0
		}
		if repeating == localRepeatedReplies {
			return work, fmt.Errorf("loop: the local worker repeated readings it had made for %d replies in a row", localRepeatedReplies)
		}
	}
	return work, fmt.Errorf("loop: the local worker spent its %d replies on the step", localWorkerTurns)
}

// askChat sends the conversation to overgo's chat route and returns the model's reply.
func askChat(ctx context.Context, server string, model workerModel, messages []chatMessage, tools []map[string]any) (chatMessage, error) {
	tokens := localReplyTokens
	if model.thinking {
		tokens = localThinkingTokens
	}
	// Greedy with no penalty: a model's recipe may add one (Qwen3.8's presence penalty of 1.5), which turns a model away
	// from tokens it has written, as code repeats its names, and keeps the server off its greedy draw, which asks a tool
	// call's grammar about the model's likeliest token alone: under the penalty every token asked it about the whole
	// vocabulary, a second a token on 2026-11-09 with the GPU idle.
	body, err := json.Marshal(map[string]any{
		"model": model.name, "messages": messages, "tools": tools, "temperature": 0, "presence_penalty": 0, "frequency_penalty": 0,
		"repeat_penalty": 1, "max_tokens": tokens, "chat_template_kwargs": map[string]any{"enable_thinking": model.thinking},
	})
	if err != nil {
		return chatMessage{}, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimSuffix(server, "/")+"/v1/chat/completions", bytes.NewReader(body))
	if err != nil {
		return chatMessage{}, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return chatMessage{}, fmt.Errorf("loop: the chat route did not answer: %w", err)
	}
	defer response.Body.Close()
	answer, err := io.ReadAll(response.Body)
	if err != nil || response.StatusCode != http.StatusOK {
		return chatMessage{}, fmt.Errorf("loop: the chat route answered %s (%v): %.500s", response.Status, err, answer)
	}
	var parsed struct {
		Choices []struct {
			Message chatMessage `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(answer, &parsed); err != nil || len(parsed.Choices) == 0 {
		return chatMessage{}, fmt.Errorf("loop: the chat route's answer holds no reply (%v): %.500s", err, answer)
	}
	return parsed.Choices[0].Message, nil
}
