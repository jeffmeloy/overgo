package inference

import (
	"errors"

	"overgo/internal/gguf"
	"overgo/internal/tokenizer"
)

// PromptReader reads prompts as a model is given them, without the model: the
// chat template and the vocabulary its file carries, and none of its weights.
// It shapes a conversation and tokenizes a prompt through the runner's own
// code, so what it reads is what a served request's model reads, and it opens
// in the time it takes to read a file's header. A prompt that reaches a model
// as other tokens than the model's own tokenizer gives is a fault no score
// shows as one; this is how the tokens are had to compare.
type PromptReader struct{ runner *Runner }

// OpenPromptReader opens the template and vocabulary of the model file at
// path.
func OpenPromptReader(path string) (*PromptReader, error) {
	file, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	vocab, err := tokenizer.Load(file)
	if err != nil {
		return nil, errors.Join(err, file.Close())
	}
	return &PromptReader{runner: &Runner{preparedModel: preparedModel{file: file, path: path, vocab: vocab}}}, nil
}

// FormatChatWithOptions renders a conversation through the file's chat
// template, as a runner of the model renders it.
func (p *PromptReader) FormatChatWithOptions(messages []ChatMessage, options ChatFormatOptions) (string, error) {
	return p.runner.FormatChatWithOptions(messages, options)
}

// GenerationTokens are the tokens a generation from the prompt starts from.
// parseSpecial reads a template's markers in the prompt as the tokens they
// name, as a generation from a shaped prompt does.
func (p *PromptReader) GenerationTokens(prompt string, parseSpecial bool) ([]tokenizer.TokenID, error) {
	return p.runner.promptTokenIDs(prompt, GenerateOptions{ParseSpecial: parseSpecial})
}

// ContinuationTokens are the tokens a scoring of candidates after a prompt
// reads: the context every candidate shares, and each candidate's own
// continuation of it.
func (p *PromptReader) ContinuationTokens(prompt string, candidates []string, parseSpecial bool) ([]tokenizer.TokenID, [][]tokenizer.TokenID, error) {
	return p.runner.continuationTokens(prompt, candidates, parseSpecial)
}

// Close releases the model file.
func (p *PromptReader) Close() error { return p.runner.file.Close() }
