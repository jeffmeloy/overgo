package inference

import (
	"errors"
	"fmt"

	"llamacpp2go/internal/sampling"
	"llamacpp2go/internal/tokenizer"
)

const (
	maxGrammarChoices      = 128
	maxGrammarChoiceTokens = 256
	maxGrammarTotalTokens  = 4096
)

// TokenizeGrammarChoices compiles exact textual completion alternatives into
// a token-level grammar for this runner's vocabulary.
func (r *Runner) TokenizeGrammarChoices(choices []string) (*sampling.TokenGrammar, error) {
	if r == nil || r.vocab == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if len(choices) == 0 {
		return nil, errors.New("inference: grammar choice list is empty")
	}
	if len(choices) > maxGrammarChoices {
		return nil, fmt.Errorf("inference: grammar choice count exceeds %d", maxGrammarChoices)
	}
	tokenChoices := make([][]int, len(choices))
	totalTokens := 0
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
		if len(ids) > maxGrammarChoiceTokens {
			return nil, fmt.Errorf(
				"inference: grammar choice %d exceeds %d tokens",
				choiceIndex,
				maxGrammarChoiceTokens,
			)
		}
		totalTokens += len(ids)
		if totalTokens > maxGrammarTotalTokens {
			return nil, fmt.Errorf(
				"inference: grammar choices exceed %d total tokens",
				maxGrammarTotalTokens,
			)
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

// CompileGBNF parses a llama.cpp-style character grammar and binds it to the
// runner vocabulary's decoded token bytes. An empty root name selects "root".
func (r *Runner) CompileGBNF(source, root string) (*sampling.GBNFGrammar, error) {
	return r.compileGBNF(source, root, sampling.GBNFLazyOptions{})
}

// CompileLazyGBNF defers grammar filtering until a configured regex or token
// trigger is accepted. Trigger-token pieces are included in grammar replay.
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
		Patterns: append([]string(nil), patterns...),
		Tokens:   tokens,
	})
}

func (r *Runner) compileGBNF(
	source, root string,
	lazy sampling.GBNFLazyOptions,
) (*sampling.GBNFGrammar, error) {
	if r == nil || r.vocab == nil {
		return nil, errors.New("inference: runner is nil")
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
