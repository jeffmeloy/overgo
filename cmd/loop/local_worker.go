package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// A local worker's budgets: what it may spend on one step before the driver hands the step on.
const (
	// localWorkerTurns is the most replies the model gives one step: a step it has not passed in that many is the next
	// worker's.
	localWorkerTurns = 40
	// localReplyTokens bounds one reply: a tool call or a short answer, the model's thinking left off.
	localReplyTokens = 4096
	// localResultChars is the most of a tool's result the model is shown; the rest is cut and said to be.
	localResultChars = 24000
)

// localInstruction opens a local worker's conversation in place of a prompt written for a client: the eight tools are
// its only hands and the step's check its end.
const localInstruction = "You work one step of a plan in a Go checkout, through the tools alone: there is no shell. " +
	"Read the step with show_step and run its check with run_check to see what fails. Find what the failure names with " +
	"search_files, and read only the lines you need with read_lines; read a whole file with read_file only when it is short. " +
	"Your replies are few, so change what the step asks with edit_file as soon as you know where, then run run_check again, " +
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

// workLocally works prompt with the model served at server under model, through tools: each tool call the model makes
// is run and its result sent back, until the model answers with no call or spends localWorkerTurns. Each call is
// written to log as it is made.
func workLocally(ctx context.Context, server, model, prompt string, tools []workerTool, log io.Writer) (localWork, error) {
	listing := make([]map[string]any, len(tools))
	for index, tool := range tools {
		listing[index] = map[string]any{"type": "function", "function": map[string]any{"name": tool.name, "description": tool.description, "parameters": tool.schema()}}
	}
	messages := []chatMessage{{Role: "system", Content: localInstruction}, {Role: "user", Content: prompt}}
	var work localWork
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
		for _, call := range reply.ToolCalls {
			work.Calls++
			var arguments map[string]any
			result := ""
			if err := json.Unmarshal([]byte(call.Function.Arguments), &arguments); err != nil {
				result = "the arguments are not one JSON object: " + err.Error()
			} else if text, err := callTool(tools, call.Function.Name, arguments); err != nil {
				result = errors.Join(errors.New(text), err).Error()
			} else {
				result = text
			}
			fmt.Fprintf(log, "local worker: %s %.200s\n", call.Function.Name, call.Function.Arguments)
			if len(result) > localResultChars {
				result = result[:localResultChars] + fmt.Sprintf("\n[%d more characters cut]", len(result)-localResultChars)
			}
			messages = append(messages, chatMessage{Role: "tool", ToolCallID: call.ID, Content: result})
		}
	}
	return work, fmt.Errorf("loop: the local worker spent its %d replies on the step", localWorkerTurns)
}

// askChat sends the conversation to overgo's chat route and returns the model's reply.
func askChat(ctx context.Context, server, model string, messages []chatMessage, tools []map[string]any) (chatMessage, error) {
	body, err := json.Marshal(map[string]any{
		"model": model, "messages": messages, "tools": tools, "temperature": 0, "max_tokens": localReplyTokens,
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
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
