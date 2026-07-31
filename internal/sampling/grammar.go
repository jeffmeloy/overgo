package sampling

import (
	"errors"
	"fmt"
)

// TokenGrammar is a deterministic token-level grammar. A transition value of
// -1 rejects that token. Accepting states may still have outgoing transitions,
// which permits one choice to be a prefix of another.
type TokenGrammar struct {
	Start          int
	VocabularySize int
	Transitions    []map[int]int
	Accepting      []bool
}

// NewChoiceGrammar compiles exact token-sequence alternatives into a trie DFA.
// eosTokens are accepted only after a complete choice.
func NewChoiceGrammar(
	choices [][]int,
	eosTokens []int,
	vocabularySize int,
) (*TokenGrammar, error) {
	if vocabularySize <= 0 {
		return nil, errors.New("grammar vocabulary size must be positive")
	}
	if len(choices) == 0 {
		return nil, errors.New("grammar choice list is empty")
	}
	type trieState struct {
		next      map[int]int
		accepting bool
	}
	states := []trieState{{next: make(map[int]int)}}
	for choiceIndex, choice := range choices {
		if len(choice) == 0 {
			return nil, fmt.Errorf("grammar choice %d is empty", choiceIndex)
		}
		state := 0
		for _, token := range choice {
			if token < 0 || token >= vocabularySize {
				return nil, fmt.Errorf(
					"grammar choice %d token %d is outside vocabulary",
					choiceIndex,
					token,
				)
			}
			next, ok := states[state].next[token]
			if !ok {
				next = len(states)
				states[state].next[token] = next
				states = append(states, trieState{next: make(map[int]int)})
			}
			state = next
		}
		states[state].accepting = true
	}
	terminal := len(states)
	states = append(states, trieState{next: make(map[int]int), accepting: true})
	for state := range terminal {
		if !states[state].accepting {
			continue
		}
		for _, token := range eosTokens {
			if token < 0 || token >= vocabularySize {
				continue
			}
			states[state].next[token] = terminal
		}
	}
	result := &TokenGrammar{
		Start:          0,
		VocabularySize: vocabularySize,
		Transitions:    make([]map[int]int, len(states)),
		Accepting:      make([]bool, len(states)),
	}
	for state, item := range states {
		result.Transitions[state] = make(map[int]int, len(item.next))
		for token, next := range item.next {
			result.Transitions[state][token] = next
		}
		result.Accepting[state] = item.accepting
	}
	return result, nil
}

func validateGrammar(grammar *TokenGrammar) error {
	if grammar == nil {
		return nil
	}
	if len(grammar.Transitions) == 0 ||
		grammar.Start < 0 || grammar.Start >= len(grammar.Transitions) ||
		len(grammar.Accepting) != len(grammar.Transitions) {
		return errors.New("grammar state table is invalid")
	}
	if grammar.VocabularySize <= 0 {
		return errors.New("grammar vocabulary is empty")
	}
	for state, transitions := range grammar.Transitions {
		for token, next := range transitions {
			if token < 0 || token >= grammar.VocabularySize {
				return fmt.Errorf(
					"grammar transition state %d has invalid token %d",
					state,
					token,
				)
			}
			if next < 0 || next >= len(grammar.Transitions) {
				return fmt.Errorf(
					"grammar transition state %d token %d has invalid target %d",
					state,
					token,
					next,
				)
			}
		}
	}
	return nil
}

func cloneGrammar(grammar *TokenGrammar) *TokenGrammar {
	if grammar == nil {
		return nil
	}
	result := &TokenGrammar{
		Start:          grammar.Start,
		VocabularySize: grammar.VocabularySize,
		Transitions:    make([]map[int]int, len(grammar.Transitions)),
		Accepting:      append([]bool(nil), grammar.Accepting...),
	}
	for state := range grammar.Transitions {
		result.Transitions[state] = make(map[int]int, len(grammar.Transitions[state]))
		for token, next := range grammar.Transitions[state] {
			result.Transitions[state][token] = next
		}
	}
	return result
}
