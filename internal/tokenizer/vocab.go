package tokenizer

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"llamacpp2go/internal/gguf"
)

const NullToken TokenID = -1

// TokenID is compatible with llama_token.
type TokenID int32

// TokenType is the legacy tokenizer.ggml.token_type value.
type TokenType int32

const (
	TokenUndefined TokenType = iota
	TokenNormal
	TokenUnknown
	TokenControl
	TokenUserDefined
	TokenUnused
	TokenByte
)

// Token describes one vocabulary entry.
type Token struct {
	Text  string
	Score float32
	Type  TokenType
}

type pair struct {
	left  string
	right string
}

// Vocab is an immutable supported vocabulary loaded from GGUF.
type Vocab struct {
	Model string
	Pre   string

	Tokens []Token

	BOS  TokenID
	EOS  TokenID
	EOT  TokenID
	EOM  TokenID
	UNK  TokenID
	SEP  TokenID
	PAD  TokenID
	Mask TokenID

	FIMPre TokenID
	FIMSuf TokenID
	FIMMid TokenID
	FIMPad TokenID
	FIMRep TokenID
	FIMSep TokenID

	AddBOS       bool
	AddEOS       bool
	AddSEP       bool
	AddPrefix    bool
	IgnoreMerges bool
	Lowercase    bool
	StripAccents bool

	tokenToID     map[string]TokenID
	mergeRank     map[pair]int
	special       []TokenID
	ugmMaxLen     int
	maxTokenLen   int
	fimConfigured bool
}

// Load reads and validates a supported vocabulary profile.
func Load(file *gguf.File) (*Vocab, error) {
	if file == nil {
		return nil, errors.New("tokenizer: GGUF file is nil")
	}
	values := make(map[string]gguf.Value, len(file.Metadata))
	for _, item := range file.Metadata {
		values[item.Key] = item.Value
	}

	model, err := requiredScalar[string](values, "tokenizer.ggml.model", gguf.ValueTypeString)
	if err != nil {
		return nil, err
	}
	if model != "gpt2" && model != "llama" && model != "t5" && model != "bert" {
		return nil, fmt.Errorf("tokenizer: model %q is unsupported; need bert, gpt2, llama, or t5", model)
	}
	pre, err := optionalScalar[string](values, "tokenizer.ggml.pre", gguf.ValueTypeString, "")
	if err != nil {
		return nil, err
	}
	if model == "gpt2" && !supportedPreTokenizer(pre) {
		return nil, fmt.Errorf("tokenizer: GPT-2 pre-tokenizer %q is unsupported", pre)
	}

	tokenTexts, err := requiredArray[string](values, "tokenizer.ggml.tokens", gguf.ValueTypeString)
	if err != nil {
		return nil, err
	}
	if len(tokenTexts) == 0 {
		return nil, errors.New("tokenizer: vocabulary is empty")
	}
	if uint64(len(tokenTexts)) > math.MaxInt32 {
		return nil, errors.New("tokenizer: vocabulary exceeds llama_token range")
	}
	scores, err := optionalArray[float32](values, "tokenizer.ggml.scores", gguf.ValueTypeFloat32)
	if err != nil {
		return nil, err
	}
	if scores != nil && len(scores) < len(tokenTexts) {
		return nil, fmt.Errorf("tokenizer: score count %d is smaller than token count %d", len(scores), len(tokenTexts))
	}
	types, err := optionalArray[int32](values, "tokenizer.ggml.token_type", gguf.ValueTypeInt32)
	if err != nil {
		return nil, err
	}
	if types != nil && len(types) < len(tokenTexts) {
		return nil, fmt.Errorf("tokenizer: token type count %d is smaller than token count %d", len(types), len(tokenTexts))
	}

	vocab := &Vocab{
		Model:         model,
		Pre:           pre,
		Tokens:        make([]Token, len(tokenTexts)),
		BOS:           11,
		EOS:           11,
		EOT:           NullToken,
		EOM:           NullToken,
		UNK:           NullToken,
		SEP:           NullToken,
		PAD:           NullToken,
		Mask:          NullToken,
		FIMPre:        NullToken,
		FIMSuf:        NullToken,
		FIMMid:        NullToken,
		FIMPad:        NullToken,
		FIMRep:        NullToken,
		FIMSep:        NullToken,
		fimConfigured: true,
		tokenToID:     make(map[string]TokenID, len(tokenTexts)),
	}
	if model == "llama" {
		vocab.BOS = 1
		vocab.EOS = 2
		vocab.UNK = 0
		vocab.AddBOS = true
		vocab.AddPrefix = true
	} else if model == "t5" {
		vocab.BOS = NullToken
		vocab.EOS = 1
		vocab.UNK = 2
		vocab.AddPrefix = true
	} else if model == "bert" {
		vocab.BOS = 101
		vocab.EOS = NullToken
		vocab.UNK = 100
		vocab.SEP = 102
		vocab.PAD = 0
		vocab.Mask = 103
		vocab.AddBOS = true
		vocab.AddSEP = true
	} else if isLlama3Pre(pre) {
		vocab.AddBOS = true
		vocab.IgnoreMerges = true
	}
	for i, text := range tokenTexts {
		if text == "" {
			text = "[EMPTY_" + strconv.Itoa(i) + "]"
		}
		if previous, exists := vocab.tokenToID[text]; exists {
			return nil, fmt.Errorf("tokenizer: duplicate token %q at IDs %d and %d", text, previous, i)
		}
		token := Token{Text: text, Type: TokenNormal}
		if scores != nil {
			token.Score = scores[i]
		}
		if types != nil {
			token.Type = TokenType(types[i])
			if token.Type < TokenUndefined || token.Type > TokenByte {
				token.Type = TokenUndefined
			}
		}
		id := TokenID(i)
		vocab.Tokens[i] = token
		vocab.tokenToID[text] = id
		if len(text) > vocab.maxTokenLen {
			vocab.maxTokenLen = len(text)
		}
		if model == "t5" &&
			(token.Type == TokenNormal || token.Type == TokenUserDefined ||
				token.Type == TokenUnused) &&
			len(text) > vocab.ugmMaxLen {
			vocab.ugmMaxLen = len(text)
		}
		if token.Type == TokenControl || token.Type == TokenUnknown || token.Type == TokenUserDefined {
			vocab.special = append(vocab.special, id)
		}
	}
	sort.SliceStable(vocab.special, func(i, j int) bool {
		return len(vocab.Tokens[vocab.special[i]].Text) > len(vocab.Tokens[vocab.special[j]].Text)
	})

	if model == "gpt2" {
		merges, mergeErr := requiredArray[string](values, "tokenizer.ggml.merges", gguf.ValueTypeString)
		if mergeErr != nil {
			return nil, mergeErr
		}
		vocab.mergeRank = make(map[pair]int, len(merges))
		for rank, merge := range merges {
			separator := strings.Index(merge[1:], " ")
			if separator < 0 {
				return nil, fmt.Errorf("tokenizer: malformed BPE merge at rank %d: %q", rank, merge)
			}
			separator++
			key := pair{left: merge[:separator], right: merge[separator+1:]}
			if key.left == "" || key.right == "" {
				return nil, fmt.Errorf("tokenizer: malformed BPE merge at rank %d: %q", rank, merge)
			}
			if previous, exists := vocab.mergeRank[key]; exists {
				return nil, fmt.Errorf("tokenizer: duplicate BPE merge %q at ranks %d and %d", merge, previous, rank)
			}
			vocab.mergeRank[key] = rank
		}
	}

	specialKeys := []struct {
		key         string
		destination *TokenID
	}{
		{"tokenizer.ggml.bos_token_id", &vocab.BOS},
		{"tokenizer.ggml.cls_token_id", &vocab.BOS},
		{"tokenizer.ggml.eos_token_id", &vocab.EOS},
		{"tokenizer.ggml.eot_token_id", &vocab.EOT},
		{"tokenizer.ggml.eom_token_id", &vocab.EOM},
		{"tokenizer.ggml.unknown_token_id", &vocab.UNK},
		// "seperator" is the spelling in the GGUF schema.
		{"tokenizer.ggml.seperator_token_id", &vocab.SEP},
		{"tokenizer.ggml.padding_token_id", &vocab.PAD},
		{"tokenizer.ggml.mask_token_id", &vocab.Mask},
		{"tokenizer.ggml.fim_pre_token_id", &vocab.FIMPre},
		{"tokenizer.ggml.fim_suf_token_id", &vocab.FIMSuf},
		{"tokenizer.ggml.fim_mid_token_id", &vocab.FIMMid},
		{"tokenizer.ggml.fim_pad_token_id", &vocab.FIMPad},
		{"tokenizer.ggml.fim_rep_token_id", &vocab.FIMRep},
		{"tokenizer.ggml.fim_sep_token_id", &vocab.FIMSep},
		// Deprecated aliases are read after the current keys, matching the
		// pinned vocabulary loader's precedence.
		{"tokenizer.ggml.prefix_token_id", &vocab.FIMPre},
		{"tokenizer.ggml.suffix_token_id", &vocab.FIMSuf},
		{"tokenizer.ggml.middle_token_id", &vocab.FIMMid},
	}
	for _, item := range specialKeys {
		id, found, readErr := optionalTokenID(values, item.key)
		if readErr != nil {
			return nil, readErr
		}
		if found {
			if id < 0 || int(id) >= len(vocab.Tokens) {
				return nil, fmt.Errorf("tokenizer: metadata %q has out-of-range token ID %d", item.key, id)
			}
			*item.destination = id
		}
	}
	if vocab.BOS >= TokenID(len(vocab.Tokens)) {
		vocab.BOS = NullToken
	}
	if vocab.EOS >= TokenID(len(vocab.Tokens)) {
		vocab.EOS = NullToken
	}
	vocab.detectFIMTokens()
	if vocab.AddBOS, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_bos_token",
		gguf.ValueTypeBool,
		vocab.AddBOS,
	); err != nil {
		return nil, err
	}
	if vocab.AddEOS, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_eos_token",
		gguf.ValueTypeBool,
		vocab.AddEOS,
	); err != nil {
		return nil, err
	}
	if vocab.AddSEP, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_sep_token",
		gguf.ValueTypeBool,
		vocab.AddSEP,
	); err != nil {
		return nil, err
	}
	if vocab.AddPrefix, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_space_prefix",
		gguf.ValueTypeBool,
		vocab.AddPrefix,
	); err != nil {
		return nil, err
	}
	if vocab.AddBOS && vocab.BOS == NullToken {
		return nil, errors.New("tokenizer: add_bos_token is true but BOS token is unavailable")
	}
	if vocab.AddEOS && vocab.EOS == NullToken {
		return nil, errors.New("tokenizer: add_eos_token is true but EOS token is unavailable")
	}
	if vocab.AddSEP && vocab.SEP == NullToken {
		return nil, errors.New("tokenizer: add_sep_token is true but separator token is unavailable")
	}
	defaultLowercase := model == "bert"
	if vocab.Lowercase, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.normalizer.lowercase",
		gguf.ValueTypeBool,
		defaultLowercase,
	); err != nil {
		return nil, err
	}
	if vocab.StripAccents, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.normalizer.strip_accents",
		gguf.ValueTypeBool,
		vocab.Lowercase,
	); err != nil {
		return nil, err
	}
	return vocab, nil
}

func supportedPreTokenizer(pre string) bool {
	switch pre {
	case "", "default", "gpt-2", "phi-2", "qwen2", "deepseek-r1-qwen", "kormo", "f2llmv2", "megrez",
		"qwen35", "llama3", "llama-v3", "llama-bpe":
		return true
	default:
		return false
	}
}

func (v *Vocab) Len() int {
	if v == nil {
		return 0
	}
	return len(v.Tokens)
}

func (v *Vocab) Token(id TokenID) (Token, bool) {
	if v == nil || id < 0 || int(id) >= len(v.Tokens) {
		return Token{}, false
	}
	return v.Tokens[id], true
}

// IsEOG reports the explicit GGUF terminal IDs and common terminal control
// tokens that llama.cpp promotes to end-of-generation markers.
func (v *Vocab) IsEOG(id TokenID) bool {
	if v == nil || id < 0 || int(id) >= len(v.Tokens) {
		return false
	}
	if id == v.EOS || id == v.EOT || id == v.EOM {
		return true
	}
	if v.fimConfigured &&
		(id == v.FIMPad || id == v.FIMRep || id == v.FIMSep) {
		return true
	}
	switch v.Tokens[id].Text {
	case "<eos>", "</s>", "<|end_of_text|>", "<|eot_id|>", "<end_of_turn>",
		"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
		"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
		"<|file_sep|>":
		return true
	default:
		return false
	}
}

func (v *Vocab) detectFIMTokens() {
	if v == nil {
		return
	}
	promoted := false
	detect := func(destination *TokenID, names ...string) {
		if *destination != NullToken {
			return
		}
		for _, name := range names {
			if id, ok := v.tokenToID[name]; ok {
				*destination = id
				if v.Tokens[id].Type != TokenControl {
					v.Tokens[id].Type = TokenControl
					promoted = true
				}
				return
			}
		}
	}
	detect(
		&v.FIMPre,
		"<|fim_prefix|>", "<fim-prefix>", "<fim_prefix>",
		"<｜fim▁begin｜>", "<PRE>", "▁<PRE>", "<|code_prefix|>", "<|prefix|>",
	)
	detect(
		&v.FIMSuf,
		"<|fim_suffix|>", "<fim-suffix>", "<fim_suffix>",
		"<｜fim▁hole｜>", "<SUF>", "▁<SUF>", "<|code_suffix|>", "<|suffix|>",
	)
	detect(
		&v.FIMMid,
		"<|fim_middle|>", "<fim-middle>", "<fim_middle>",
		"<｜fim▁end｜>", "<MID>", "▁<MID>", "<|code_middle|>", "<|middle|>",
	)
	detect(
		&v.FIMPad,
		"<|fim_pad|>", "<fim-pad>", "<fim_pad>", "<PAD>", "[PAD]",
	)
	detect(
		&v.FIMRep,
		"<|fim_repo|>", "<|repo_name|>", "<fim-repo>", "<REPO>", "<reponame>",
	)
	detect(&v.FIMSep, "<|file_sep|>")
	if promoted {
		v.special = v.special[:0]
		for id, token := range v.Tokens {
			if token.Type == TokenControl ||
				token.Type == TokenUnknown ||
				token.Type == TokenUserDefined {
				v.special = append(v.special, TokenID(id))
			}
		}
		sort.SliceStable(v.special, func(i, j int) bool {
			return len(v.Tokens[v.special[i]].Text) >
				len(v.Tokens[v.special[j]].Text)
		})
	}
}

func (v *Vocab) EOGTokens() []TokenID {
	if v == nil {
		return nil
	}
	result := make([]TokenID, 0, 5)
	for id := range v.Tokens {
		if v.IsEOG(TokenID(id)) {
			result = append(result, TokenID(id))
		}
	}
	return result
}

func (v *Vocab) ID(text string) (TokenID, bool) {
	if v == nil {
		return NullToken, false
	}
	id, ok := v.tokenToID[text]
	return id, ok
}

func (v *Vocab) MergeCount() int {
	if v == nil {
		return 0
	}
	return len(v.mergeRank)
}

func requiredScalar[T any](values map[string]gguf.Value, key string, kind gguf.ValueType) (T, error) {
	var zero T
	value, ok := values[key]
	if !ok {
		return zero, fmt.Errorf("tokenizer: required metadata %q is missing", key)
	}
	if value.Type != kind {
		return zero, fmt.Errorf("tokenizer: metadata %q has type %s, need %s", key, value.Type, kind)
	}
	result, ok := value.Data.(T)
	if !ok {
		return zero, fmt.Errorf("tokenizer: metadata %q has an invalid Go representation", key)
	}
	return result, nil
}

func optionalScalar[T any](values map[string]gguf.Value, key string, kind gguf.ValueType, fallback T) (T, error) {
	value, ok := values[key]
	if !ok {
		return fallback, nil
	}
	if value.Type != kind {
		return fallback, fmt.Errorf("tokenizer: metadata %q has type %s, need %s", key, value.Type, kind)
	}
	result, ok := value.Data.(T)
	if !ok {
		return fallback, fmt.Errorf("tokenizer: metadata %q has an invalid Go representation", key)
	}
	return result, nil
}

func requiredArray[T any](values map[string]gguf.Value, key string, kind gguf.ValueType) ([]T, error) {
	result, err := optionalArray[T](values, key, kind)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("tokenizer: required metadata %q is missing", key)
	}
	return result, nil
}

func optionalArray[T any](values map[string]gguf.Value, key string, kind gguf.ValueType) ([]T, error) {
	value, ok := values[key]
	if !ok {
		return nil, nil
	}
	if value.Type != gguf.ValueTypeArray || value.ArrayType != kind {
		return nil, fmt.Errorf("tokenizer: metadata %q must be a %s array", key, kind)
	}
	result, ok := value.Data.([]T)
	if !ok {
		return nil, fmt.Errorf("tokenizer: metadata %q has an invalid Go representation", key)
	}
	return result, nil
}

func optionalTokenID(values map[string]gguf.Value, key string) (TokenID, bool, error) {
	value, ok := values[key]
	if !ok {
		return NullToken, false, nil
	}
	switch value.Type {
	case gguf.ValueTypeUint32:
		raw, valid := value.Data.(uint32)
		if !valid || raw > math.MaxInt32 {
			return NullToken, false, fmt.Errorf("tokenizer: metadata %q has an invalid token ID", key)
		}
		return TokenID(raw), true, nil
	case gguf.ValueTypeInt32:
		raw, valid := value.Data.(int32)
		if !valid || raw < 0 {
			return NullToken, false, fmt.Errorf("tokenizer: metadata %q has an invalid token ID", key)
		}
		return TokenID(raw), true, nil
	default:
		return NullToken, false, fmt.Errorf("tokenizer: metadata %q must be uint32 or int32", key)
	}
}
