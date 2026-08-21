package inference

import (
	"errors"
	"fmt"
	"slices"

	"overgo/internal/sampling"
	"overgo/internal/tokenizer"
)

// TokenizeGrammarChoices: compiles exact textual completion alternatives into
// token-level grammar for this runner's vocabulary
func (r *Runner) TokenizeGrammarChoices(choices []string) (*sampling.TokenGrammar, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	if len(choices) == 0 {
		return nil, errors.New("inference: grammar choice list is empty")
	}
	tokenChoices := make([][]int, len(choices))
	for choiceIndex, choice := range choices {
		if choice == "" {
			return nil, fmt.Errorf("inference: grammar choice %d is empty", choiceIndex)
		}
		ids, err := r.vocab.Encode(choice, tokenizer.EncodeOptions{})
		if err != nil {
			return nil, fmt.Errorf("inference: tokenize grammar choice %d: %w", choiceIndex, err)
		}
		if len(ids) == 0 {
			return nil, fmt.Errorf("inference: grammar choice %d tokenizes to nothing", choiceIndex)
		}
		tokenChoices[choiceIndex] = make([]int, len(ids))
		for tokenIndex, id := range ids {
			tokenChoices[choiceIndex][tokenIndex] = int(id)
		}
	}
	terminalIDs := r.vocab.EOGTokens()
	terminalTokens := make([]int, 0, len(terminalIDs))
	for _, id := range terminalIDs {
		terminalTokens = append(terminalTokens, int(id))
	}
	return sampling.NewChoiceGrammar(tokenChoices, terminalTokens, r.vocab.Len())
}

// CompileGBNF: parses llama.cpp-style character grammar and binds it to
// runner vocabulary's decoded token bytes; empty root name selects "root"
func (r *Runner) CompileGBNF(source, root string) (*sampling.GBNFGrammar, error) {
	return r.compileGBNF(source, root, sampling.GBNFLazyOptions{})
}

// CompileLazyGBNF defers grammar filtering until configured regex or token
// trigger: accepted; Trigger-token pieces are included in grammar replay
func (r *Runner) CompileLazyGBNF(
	source, root string,
	patterns []string,
	triggerTokens []tokenizer.TokenID,
) (*sampling.GBNFGrammar, error) {
	tokens := make([]int, len(triggerTokens))
	for index, token := range triggerTokens {
		tokens[index] = int(token)
	}
	return r.compileGBNF(source, root, sampling.GBNFLazyOptions{
		Enabled:  true,
		Patterns: slices.Clone(patterns),
		Tokens:   tokens,
	})
}

func (r *Runner) compileGBNF(
	source, root string,
	lazy sampling.GBNFLazyOptions,
) (*sampling.GBNFGrammar, error) {
	if r == nil || r.vocab == nil {
		return nil, errRunnerNil
	}
	pieces := make([][]byte, r.vocab.Len())
	tokenIDs := make(map[string]int, r.vocab.Len())
	for token := range pieces {
		piece, err := r.vocab.DecodePiece(tokenizer.TokenID(token), true)
		if err != nil {
			return nil, fmt.Errorf(
				"inference: decode GBNF token %d: %w",
				token,
				err,
			)
		}
		pieces[token] = []byte(piece)
		if value, ok := r.vocab.Token(tokenizer.TokenID(token)); ok {
			tokenIDs[value.Text] = token
		}
	}
	terminalIDs := r.vocab.EOGTokens()
	terminalTokens := make([]int, len(terminalIDs))
	for index, id := range terminalIDs {
		terminalTokens[index] = int(id)
	}
	grammar, err := sampling.NewGBNFGrammarWithOptions(
		source,
		root,
		pieces,
		terminalTokens,
		tokenIDs,
		lazy,
	)
	if err != nil {
		return nil, fmt.Errorf("inference: compile GBNF: %w", err)
	}
	return grammar, nil
}
