package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"

	"overgo/internal/clioptions"
	"overgo/internal/evaluation"
	"overgo/internal/inference"
	"overgo/internal/tokenizer"
)

// readPrompt is one multiple-choice case as a model is given it: the text,
// and the tokens the text reaches the model as.
type readPrompt struct {
	// Name, Prompt and Candidates are the case as its suite holds it.
	Name       string   `json:"name"`
	Prompt     string   `json:"prompt"`
	Candidates []string `json:"candidates"`
	// User and Text are, under the chat-template protocol, what the evaluation
	// says in the user's turn before any template shapes it, the question
	// with its instruction, and the whole text a generation starts from.
	User string `json:"user,omitempty"`
	Text string `json:"text,omitempty"`
	// IDs are the tokens a generation starts from under the chat-template
	// protocol, and under raw completion the context every candidate is
	// scored after.
	IDs []tokenizer.TokenID `json:"ids"`
	// Continuations are, under raw completion, each candidate's tokens after
	// that context.
	Continuations [][]tokenizer.TokenID `json:"continuations,omitempty"`
}

// templateShaper shapes a user turn from the template a model file carries,
// as chatShapedRuntime shapes it from a loaded model, and keeps what it was
// asked to shape.
type templateShaper struct {
	reader *inference.PromptReader
	user   *string
}

// ShapeChatPrompt renders the prompt as the user turn the evaluation asks
// every chat-template question in.
func (s templateShaper) ShapeChatPrompt(prompt string) (string, error) {
	*s.user = prompt
	return userTurn(s.reader, prompt)
}

// readPrompts reads each multiple-choice case of a file as the model file at
// path is given it by an evaluation, from the file's template and vocabulary
// and none of its weights. Under the chat-template protocol a case is the
// question shaped as one user turn with the model's turn opened for its
// answer, tokenized as a generation tokenizes it. Under raw completion it is
// the prompt and each candidate's continuation of it, tokenized as a scoring
// tokenizes them. The texts and tokens come from the evaluation's own code,
// so they can be set beside what the model's own tokenizer gives.
func readPrompts(path, casesPath string, chat bool) ([]readPrompt, error) {
	data, err := os.ReadFile(casesPath)
	if err != nil {
		return nil, err
	}
	var suite struct {
		Cases []evaluation.MultipleChoiceCase `json:"cases"`
	}
	if err := json.Unmarshal(data, &suite); err != nil {
		return nil, fmt.Errorf("evaluate: %s: %w", casesPath, err)
	}
	if len(suite.Cases) == 0 {
		return nil, errors.New("evaluate: " + casesPath + " holds no multiple-choice case")
	}
	reader, err := inference.OpenPromptReader(path)
	if err != nil {
		return nil, err
	}
	read := make([]readPrompt, 0, len(suite.Cases))
	for _, testCase := range suite.Cases {
		prompt := readPrompt{Name: testCase.Name, Prompt: testCase.Prompt, Candidates: testCase.Candidates}
		if chat {
			if prompt.Text, err = evaluation.ChatChoicePrompt(templateShaper{reader, &prompt.User}, testCase); err == nil {
				prompt.IDs, err = reader.GenerationTokens(prompt.Text, true)
			}
		} else {
			prompt.IDs, prompt.Continuations, err = reader.ContinuationTokens(testCase.Prompt, testCase.Candidates, false)
		}
		if err != nil {
			return nil, errors.Join(fmt.Errorf("evaluate: case %s: %w", testCase.Name, err), reader.Close())
		}
		read = append(read, prompt)
	}
	return read, reader.Close()
}

// writePrompts prints, as JSON, each case of a file as the model is given it.
func writePrompts(path, casesPath string, chat bool) error {
	read, err := readPrompts(path, casesPath, chat)
	if err != nil {
		return err
	}
	return clioptions.WriteJSON(os.Stdout, read)
}
