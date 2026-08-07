package inference

import (
	"errors"
	"fmt"

	"overgo/internal/tokenizer"
)

const DefaultInfillBatchSize = 2048

type InfillExtra struct {
	Filename string
	Tokens   []tokenizer.TokenID
}

type InfillFormatOptions struct {
	BatchSize    int
	MaxNewTokens int
	SuffixPrefix bool
}

// FormatInfillTokens implements pinned server's FIM repository-context
// layout and its 3:1 prefix/suffix truncation policy
func (r *Runner) FormatInfillTokens(
	prefix, suffix, prompt []tokenizer.TokenID,
	extra []InfillExtra,
	options InfillFormatOptions,
) ([]tokenizer.TokenID, error) {
	if r == nil || r.vocab == nil {
		return nil, errors.New("inference: runner is nil")
	}
	if r.vocab.FIMPre == tokenizer.NullToken {
		return nil, errors.New("inference: FIM prefix token is unavailable")
	}
	if r.vocab.FIMSuf == tokenizer.NullToken {
		return nil, errors.New("inference: FIM suffix token is unavailable")
	}
	if r.vocab.FIMMid == tokenizer.NullToken {
		return nil, errors.New("inference: FIM middle token is unavailable")
	}
	if options.BatchSize == 0 {
		options.BatchSize = DefaultInfillBatchSize
	}
	if options.BatchSize < 1 ||
		options.BatchSize > int(r.spec.ContextLength) {
		return nil, fmt.Errorf(
			"inference: FIM batch size must be in [1,%d]",
			r.spec.ContextLength,
		)
	}
	if options.MaxNewTokens < 0 {
		return nil, errors.New("inference: FIM maximum new tokens is negative")
	}
	for name, tokens := range map[string][]tokenizer.TokenID{
		"prefix": prefix,
		"suffix": suffix,
		"prompt": prompt,
	} {
		if err := r.validateInfillTokens(name, tokens); err != nil {
			return nil, err
		}
	}

	extraTokens := make([]tokenizer.TokenID, 0)
	if r.vocab.FIMRep != tokenizer.NullToken {
		project, err := r.vocab.Encode("myproject\n", tokenizer.EncodeOptions{})
		if err != nil {
			return nil, fmt.Errorf("inference: tokenize FIM project: %w", err)
		}
		extraTokens = append(extraTokens, r.vocab.FIMRep)
		extraTokens = append(extraTokens, project...)
	}
	for index, chunk := range extra {
		if err := r.validateInfillTokens(
			fmt.Sprintf("extra chunk %d", index),
			chunk.Tokens,
		); err != nil {
			return nil, err
		}
		filename := chunk.Filename
		if filename == "" {
			filename = "tmp"
		}
		if r.vocab.FIMSep != tokenizer.NullToken {
			filenameTokens, err := r.vocab.Encode(
				filename+"\n",
				tokenizer.EncodeOptions{},
			)
			if err != nil {
				return nil, fmt.Errorf(
					"inference: tokenize FIM extra filename %d: %w",
					index,
					err,
				)
			}
			extraTokens = append(extraTokens, r.vocab.FIMSep)
			extraTokens = append(extraTokens, filenameTokens...)
		} else {
			separator, err := r.vocab.Encode(
				"\n\n--- snippet ---\n\n",
				tokenizer.EncodeOptions{},
			)
			if err != nil {
				return nil, fmt.Errorf(
					"inference: tokenize FIM chunk separator: %w",
					err,
				)
			}
			extraTokens = append(extraTokens, separator...)
		}
		extraTokens = append(extraTokens, chunk.Tokens...)
	}
	if r.vocab.FIMSep != tokenizer.NullToken {
		filename, err := r.vocab.Encode("filename\n", tokenizer.EncodeOptions{})
		if err != nil {
			return nil, fmt.Errorf("inference: tokenize FIM filename: %w", err)
		}
		extraTokens = append(extraTokens, r.vocab.FIMSep)
		extraTokens = append(extraTokens, filename...)
	}

	quarter := options.BatchSize / 4
	prefixTake := min(len(prefix), 3*quarter)
	suffixTake := min(len(suffix), max(0, quarter-(2+len(prompt))))
	prefix = prefix[len(prefix)-prefixTake:]
	suffix = suffix[:suffixTake]

	prefixPart := make([]tokenizer.TokenID, 0, 1+len(prefix)+len(prompt))
	prefixPart = append(prefixPart, r.vocab.FIMPre)
	prefixPart = append(prefixPart, prefix...)
	prefixPart = append(prefixPart, prompt...)
	suffixPart := make([]tokenizer.TokenID, 0, 1+len(suffix))
	suffixPart = append(suffixPart, r.vocab.FIMSuf)
	suffixPart = append(suffixPart, suffix...)

	first, last := prefixPart, suffixPart
	if options.SuffixPrefix {
		first, last = suffixPart, prefixPart
	}
	if r.vocab.AddBOS {
		first = append([]tokenizer.TokenID{r.vocab.BOS}, first...)
	}

	extraTake := min(
		max(
			0,
			int(r.spec.ContextLength)-
				options.BatchSize-
				2*options.MaxNewTokens,
		),
		len(extraTokens),
	)
	result := make([]tokenizer.TokenID, 0, extraTake+len(first)+len(last)+1)
	result = append(result, extraTokens[len(extraTokens)-extraTake:]...)
	result = append(result, first...)
	result = append(result, last...)
	result = append(result, r.vocab.FIMMid)
	if len(result) > int(r.spec.ContextLength) {
		return nil, fmt.Errorf(
			"inference: formatted FIM prompt has %d tokens, exceeds context length %d",
			len(result),
			r.spec.ContextLength,
		)
	}
	return result, nil
}

func (r *Runner) validateInfillTokens(
	name string,
	tokens []tokenizer.TokenID,
) error {
	for index, token := range tokens {
		if token < 0 || int(token) >= r.vocab.Len() {
			return fmt.Errorf(
				"inference: FIM %s token %d ID %d is out of range",
				name,
				index,
				token,
			)
		}
	}
	return nil
}
