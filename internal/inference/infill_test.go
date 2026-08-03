package inference

import (
	"slices"
	"testing"

	"llamacpp2go/internal/model"
	"llamacpp2go/internal/tokenizer"
)

func TestFormatInfillTokensPSMAndSPM(t *testing.T) {
	tokens := make([]tokenizer.Token, 24)
	for index := range tokens {
		tokens[index] = tokenizer.Token{Text: string(rune('a' + index))}
	}
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{ContextLength: 32}},
		vocab: &tokenizer.Vocab{
			Model:  "gpt2",
			Tokens: tokens,
			BOS:    1,
			EOS:    tokenizer.NullToken,
			EOT:    tokenizer.NullToken,
			EOM:    tokenizer.NullToken,
			UNK:    tokenizer.NullToken,
			SEP:    tokenizer.NullToken,
			PAD:    tokenizer.NullToken,
			Mask:   tokenizer.NullToken,
			FIMPre: 2,
			FIMSuf: 3,
			FIMMid: 4,
			FIMPad: tokenizer.NullToken,
			FIMRep: tokenizer.NullToken,
			FIMSep: tokenizer.NullToken,
			AddBOS: true,
		}},
	}
	prefix := []tokenizer.TokenID{5, 6, 7, 8, 9}
	suffix := []tokenizer.TokenID{10, 11, 12}
	prompt := []tokenizer.TokenID{13}
	psm, err := runner.FormatInfillTokens(
		prefix,
		suffix,
		prompt,
		nil,
		InfillFormatOptions{BatchSize: 16, MaxNewTokens: 2},
	)
	if err != nil {
		t.Fatal(err)
	}
	wantPSM := []tokenizer.TokenID{1, 2, 5, 6, 7, 8, 9, 13, 3, 10, 4}
	if !slices.Equal(psm, wantPSM) {
		t.Fatalf("PSM tokens = %v, want %v", psm, wantPSM)
	}
	spm, err := runner.FormatInfillTokens(
		prefix,
		suffix,
		prompt,
		nil,
		InfillFormatOptions{
			BatchSize:    16,
			MaxNewTokens: 2,
			SuffixPrefix: true,
		},
	)
	if err != nil {
		t.Fatal(err)
	}
	wantSPM := []tokenizer.TokenID{1, 3, 10, 2, 5, 6, 7, 8, 9, 13, 4}
	if !slices.Equal(spm, wantSPM) {
		t.Fatalf("SPM tokens = %v, want %v", spm, wantSPM)
	}
}

func TestFormatInfillTokensTruncatesPrefixTailAndSuffixHead(t *testing.T) {
	tokens := make([]tokenizer.Token, 32)
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{ContextLength: 32}},
		vocab: &tokenizer.Vocab{
			Tokens: tokens,
			BOS:    tokenizer.NullToken,
			FIMPre: 1,
			FIMSuf: 2,
			FIMMid: 3,
			FIMPad: tokenizer.NullToken,
			FIMRep: tokenizer.NullToken,
			FIMSep: tokenizer.NullToken,
		}},
	}
	prefix := []tokenizer.TokenID{4, 5, 6, 7, 8, 9, 10, 11}
	suffix := []tokenizer.TokenID{12, 13, 14, 15}
	got, err := runner.FormatInfillTokens(
		prefix,
		suffix,
		nil,
		nil,
		InfillFormatOptions{BatchSize: 8},
	)
	if err != nil {
		t.Fatal(err)
	}
	want := []tokenizer.TokenID{1, 6, 7, 8, 9, 10, 11, 2, 3}
	if !slices.Equal(got, want) {
		t.Fatalf("truncated FIM tokens = %v, want %v", got, want)
	}
}

func TestFormatInfillTokensRequiresControlTokens(t *testing.T) {
	runner := &Runner{preparedModel: preparedModel{spec: model.Spec{CommonSpec: model.CommonSpec{ContextLength: 8}},
		vocab: &tokenizer.Vocab{
			Tokens: make([]tokenizer.Token, 4),
			FIMPre: tokenizer.NullToken,
			FIMSuf: tokenizer.NullToken,
			FIMMid: tokenizer.NullToken,
		}},
	}
	if _, err := runner.FormatInfillTokens(
		nil,
		nil,
		nil,
		nil,
		InfillFormatOptions{BatchSize: 4},
	); err == nil {
		t.Fatal("FIM formatter accepted a vocabulary without control tokens")
	}
}
