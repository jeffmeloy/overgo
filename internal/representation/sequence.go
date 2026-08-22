package representation

import (
	"errors"

	"overgo/internal/checked"
)

// PaddedSequence is a fixed-budget token sequence and its explicit key mask.
type PaddedSequence struct {
	IDs        []int
	Mask       []bool
	PromptRows int
}

// ActivePrefixRows returns the materialized prefix size for a sequence that
// keeps one explicit zero-padding row unless it already fills capacity.
func ActivePrefixRows(tokens, capacity int) (int, error) {
	if tokens <= 0 || capacity <= 0 || tokens > capacity {
		return 0, errors.New("representation: active token rows exceed capacity")
	}
	if tokens == capacity {
		return tokens, nil
	}
	return tokens + 1, nil
}

// AssemblePaddedSequence lays out prefix, truncated prompt, suffix padding, and
// suffix tokens under a fixed prompt-row budget.
func AssemblePaddedSequence(prefix, prompt, suffix []int, maximumPromptTokens, paddingToken int) (PaddedSequence, error) {
	if maximumPromptTokens <= 0 || len(prefix) == 0 || len(suffix) == 0 || len(suffix) > maximumPromptTokens {
		return PaddedSequence{}, errors.New("representation: invalid padded sequence contract")
	}
	if len(prompt) > maximumPromptTokens {
		prompt = prompt[:maximumPromptTokens]
	}
	padded, ok := checked.AddInt(maximumPromptTokens, len(prefix))
	if !ok || padded < len(suffix) {
		return PaddedSequence{}, errors.New("representation: padded sequence extent overflows")
	}
	padded -= len(suffix)
	total, ok := checked.AddInt(padded, len(suffix))
	if !ok {
		return PaddedSequence{}, errors.New("representation: padded sequence extent overflows")
	}
	result := PaddedSequence{IDs: make([]int, total), Mask: make([]bool, total), PromptRows: maximumPromptTokens}
	for index := range result.IDs {
		result.IDs[index] = paddingToken
	}
	written := copy(result.IDs, prefix)
	written += copy(result.IDs[written:], prompt)
	for index := 0; index < written; index++ {
		result.Mask[index] = true
	}
	copy(result.IDs[padded:], suffix)
	for index := padded; index < total; index++ {
		result.Mask[index] = true
	}
	return result, nil
}
