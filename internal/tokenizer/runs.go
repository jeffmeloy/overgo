package tokenizer

import (
	"errors"
	"fmt"
	"strings"
)

// TextEncoder: vocabulary-compatible text encoder.
type TextEncoder func(string, EncodeOptions) ([]TokenID, error)

// EncodeRuns inserts verified single-token placeholder runs without encoding
// their repeated source spelling.
func EncodeRuns(
	encode TextEncoder,
	text, placeholder string,
	counts []int,
	options EncodeOptions,
) ([]TokenID, []int, error) {
	return encodePlaceholderRuns(encode, text, placeholder, counts, options, true)
}

// EncodeMarkers expands one placeholder marker per token run.
func EncodeMarkers(
	encode TextEncoder,
	text, placeholder string,
	counts []int,
	options EncodeOptions,
) ([]TokenID, []int, error) {
	return encodePlaceholderRuns(encode, text, placeholder, counts, options, false)
}

func encodePlaceholderRuns(
	encode TextEncoder,
	text, placeholder string,
	counts []int,
	options EncodeOptions,
	expanded bool,
) ([]TokenID, []int, error) {
	if encode == nil || placeholder == "" || len(counts) == 0 {
		return nil, nil, errors.New("tokenizer: token-run plan is invalid")
	}
	placeholderIDs, err := encode(placeholder, EncodeOptions{ParseSpecial: true})
	if err != nil {
		return nil, nil, err
	}
	if len(placeholderIDs) != 1 {
		return nil, nil, fmt.Errorf("tokenizer: placeholder maps to %d tokens", len(placeholderIDs))
	}
	ids := make([]TokenID, 0, len(text)/2)
	starts := make([]int, len(counts))
	remaining := text
	addSpecial := options.AddSpecial
	appendText := func(part string) error {
		encoded, encodeErr := encode(part, EncodeOptions{
			AddSpecial: addSpecial, ParseSpecial: options.ParseSpecial,
		})
		if encodeErr != nil {
			return encodeErr
		}
		ids = append(ids, encoded...)
		addSpecial = false
		return nil
	}
	for run, count := range counts {
		if count <= 0 {
			return nil, nil, fmt.Errorf("tokenizer: placeholder run %d is empty", run)
		}
		start := strings.Index(remaining, placeholder)
		if start < 0 {
			return nil, nil, fmt.Errorf("tokenizer: placeholder run %d is absent", run)
		}
		if err := appendText(remaining[:start]); err != nil {
			return nil, nil, err
		}
		remaining = remaining[start:]
		starts[run] = len(ids)
		sourceCount := 1
		if expanded {
			sourceCount = count
		}
		for index := range sourceCount {
			if !strings.HasPrefix(remaining, placeholder) {
				return nil, nil, fmt.Errorf("tokenizer: placeholder run %d has %d tokens, need %d", run, index, count)
			}
			remaining = remaining[len(placeholder):]
		}
		for range count {
			ids = append(ids, placeholderIDs[0])
		}
		if expanded && strings.HasPrefix(remaining, placeholder) {
			return nil, nil, fmt.Errorf("tokenizer: placeholder run %d exceeds %d tokens", run, count)
		}
	}
	if strings.Contains(remaining, placeholder) {
		return nil, nil, errors.New("tokenizer: prompt has an unexpected placeholder run")
	}
	if err := appendText(remaining); err != nil {
		return nil, nil, err
	}
	return ids, starts, nil
}
