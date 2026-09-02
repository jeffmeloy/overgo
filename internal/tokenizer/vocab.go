package tokenizer

import (
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"

	"overgo/internal/gguf"
)

const NullToken TokenID = -1

// TokenID: compatible with llama_token
type TokenID int32

// ParseTokenID parses a decimal identifier within the serialized token range.
func ParseTokenID(text string) (TokenID, error) {
	value, err := strconv.Atoi(text)
	if err != nil || int(TokenID(value)) != value {
		return NullToken, fmt.Errorf("token ID %q is invalid", text)
	}
	return TokenID(value), nil
}

// TokenType: tokenizer.ggml.token_type value.
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

// Token: describes one vocabulary entry
type Token struct {
	Text  string
	Score float32
	Type  TokenType
}

// TokenRange: half-open vocabulary interval.
type TokenRange struct {
	Start TokenID
	End   TokenID
}

type pair struct {
	left  string
	right string
}

// Vocab: immutable supported vocabulary loaded from GGUF
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

	tokenToID   map[string]TokenID
	eogByName   map[TokenID]bool
	mergeRank   map[pair]int
	special     []TokenID
	ugmMaxLen   int
	maxTokenLen int
	fimDeclared bool
	dna         *dnaExtension
}

// Load: reads and validates supported vocabulary profile
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
	if model != "gpt2" && model != "gemma4" && model != "llama" && model != "t5" && model != "bert" {
		return nil, fmt.Errorf("tokenizer: model %q is unsupported; need bert, gemma4, gpt2, llama, or t5", model)
	}
	pre := ""
	if model == "gpt2" {
		pre, err = requiredScalar[string](values, "tokenizer.ggml.pre", gguf.ValueTypeString)
		if err != nil {
			return nil, err
		}
		if !supportedPreTokenizer(pre) {
			return nil, fmt.Errorf("tokenizer: GPT-2 pre-tokenizer %q is unsupported", pre)
		}
	}
	if model == "gemma4" {
		pre = "gemma4"
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
		Model:     model,
		Pre:       pre,
		Tokens:    make([]Token, len(tokenTexts)),
		BOS:       NullToken,
		EOS:       NullToken,
		EOT:       NullToken,
		EOM:       NullToken,
		UNK:       NullToken,
		SEP:       NullToken,
		PAD:       NullToken,
		Mask:      NullToken,
		FIMPre:    NullToken,
		FIMSuf:    NullToken,
		FIMMid:    NullToken,
		FIMPad:    NullToken,
		FIMRep:    NullToken,
		FIMSep:    NullToken,
		tokenToID: make(map[string]TokenID, len(tokenTexts)),
	}
	if policy := preTokenizers[pre]; model == "gpt2" && policy.ignoreMerges {
		vocab.AddBOS = policy.addBOS
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

	if model == "gpt2" || model == "gemma4" {
		merges, mergeErr := requiredArray[string](values, "tokenizer.ggml.merges", gguf.ValueTypeString)
		if mergeErr != nil {
			return nil, mergeErr
		}
		vocab.mergeRank = make(map[pair]int, len(merges))
		for rank, merge := range merges {
			if len(merge) < 3 {
				return nil, fmt.Errorf("tokenizer: malformed BPE merge at rank %d: %q", rank, merge)
			}
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
		// "seperator" is spelling in GGUF schema
		{"tokenizer.ggml.seperator_token_id", &vocab.SEP},
		{"tokenizer.ggml.padding_token_id", &vocab.PAD},
		{"tokenizer.ggml.mask_token_id", &vocab.Mask},
		{"tokenizer.ggml.fim_pre_token_id", &vocab.FIMPre},
		{"tokenizer.ggml.fim_suf_token_id", &vocab.FIMSuf},
		{"tokenizer.ggml.fim_mid_token_id", &vocab.FIMMid},
		{"tokenizer.ggml.fim_pad_token_id", &vocab.FIMPad},
		{"tokenizer.ggml.fim_rep_token_id", &vocab.FIMRep},
		{"tokenizer.ggml.fim_sep_token_id", &vocab.FIMSep},
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
	vocab.fimDeclared = vocab.FIMPad != NullToken || vocab.FIMRep != NullToken || vocab.FIMSep != NullToken
	vocab.markEndOfGenerationByName()
	if vocab.BOS >= TokenID(len(vocab.Tokens)) {
		vocab.BOS = NullToken
	}
	if vocab.EOS >= TokenID(len(vocab.Tokens)) {
		vocab.EOS = NullToken
	}
	if vocab.AddBOS, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_bos_token",
		gguf.ValueTypeBool,
		false,
	); err != nil {
		return nil, err
	}
	if vocab.AddEOS, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_eos_token",
		gguf.ValueTypeBool,
		false,
	); err != nil {
		return nil, err
	}
	if vocab.AddSEP, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_sep_token",
		gguf.ValueTypeBool,
		false,
	); err != nil {
		return nil, err
	}
	if vocab.AddPrefix, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.add_space_prefix",
		gguf.ValueTypeBool,
		false,
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
	if vocab.Lowercase, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.normalizer.lowercase",
		gguf.ValueTypeBool,
		false,
	); err != nil {
		return nil, err
	}
	if vocab.StripAccents, err = optionalScalar[bool](
		values,
		"tokenizer.ggml.normalizer.strip_accents",
		gguf.ValueTypeBool,
		false,
	); err != nil {
		return nil, err
	}
	dnaK, err := optionalScalar[uint32](values, MetadataDNAK, gguf.ValueTypeUint32, 0)
	if err != nil {
		return nil, err
	}
	if dnaK != 0 {
		start, err := requiredScalar[uint32](values, MetadataDNAStartID, gguf.ValueTypeUint32)
		if err != nil {
			return nil, err
		}
		vocabulary, err := requiredScalar[uint32](values, MetadataDNAVocabulary, gguf.ValueTypeUint32)
		if err != nil {
			return nil, err
		}
		specialTokens, err := requiredArray[string](values, MetadataDNASpecialTokens, gguf.ValueTypeString)
		if err != nil {
			return nil, err
		}
		autoTags, err := optionalScalar[bool](values, MetadataDNAAutoTags, gguf.ValueTypeBool, false)
		if err != nil {
			return nil, err
		}
		if err := ValidateDNAExtension(dnaK, start, vocabulary, specialTokens, uint32(len(vocab.Tokens))); err != nil {
			return nil, fmt.Errorf("tokenizer: DNA extension: %w", err)
		}
		vocab.dna = &dnaExtension{
			k: dnaK, start: start, vocabulary: vocabulary,
			specialTokens: specialTokens, autoTags: autoTags,
		}
	}
	return vocab, nil
}

type preTokenizerPolicy struct {
	split        func(string) []string
	addBOS       bool
	ignoreMerges bool
}

var preTokenizers = map[string]preTokenizerPolicy{
	"default":          {split: preTokenizeGPT2},
	"gpt-2":            {split: preTokenizeGPT2},
	"phi-2":            {split: preTokenizeGPT2},
	"qwen2":            {split: preTokenizeQwen2},
	"deepseek-r1-qwen": {split: preTokenizeQwen2},
	"kormo":            {split: preTokenizeQwen2},
	"f2llmv2":          {split: preTokenizeQwen2},
	"megrez":           {split: preTokenizeQwen2},
	"bailingmoe":       {split: preTokenizeQwen2},
	"bailingmoe2":      {split: preTokenizeQwen2},
	"deepseek-llm":     {split: preTokenizeDeepSeekLLM},
	"qwen35":           {split: preTokenizeQwen35},
	"dbrx":             {split: preTokenizeLlama3, addBOS: true, ignoreMerges: true},
	"llama3":           {split: preTokenizeLlama3, addBOS: true, ignoreMerges: true},
	"llama-v3":         {split: preTokenizeLlama3, addBOS: true, ignoreMerges: true},
	"llama-bpe":        {split: preTokenizeLlama3, addBOS: true, ignoreMerges: true},
	"gpt-4o":           {split: preTokenizeGPT4O},
	"llama4":           {split: preTokenizeGPT4O},
	"kanana2":          {split: preTokenizeGPT4O},
	"talkie":           {split: preTokenizeGPT4O},
}

func supportedPreTokenizer(pre string) bool {
	_, ok := preTokenizers[pre]
	return ok
}

func preTokenizeFor(pre, text string) []string {
	return preTokenizers[pre].split(text)
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

// TensorIndices validates token IDs for embedding selection.
func (v *Vocab) TensorIndices(ids []TokenID) ([]uint32, error) {
	rows := make([]uint32, len(ids))
	for index, id := range ids {
		if id < 0 || int(id) >= len(v.Tokens) {
			return nil, fmt.Errorf("tokenizer: token ID %d is out of range", id)
		}
		rows[index] = uint32(id)
	}
	return rows, nil
}

// NonTextGenerationRanges compiles serialized codebook-token intervals.
func (v *Vocab) NonTextGenerationRanges() []TokenRange {
	if v == nil {
		return nil
	}
	var ranges []TokenRange
	for index, token := range v.Tokens {
		if !isSerializedCodebookToken(token.Text) {
			continue
		}
		id := TokenID(index)
		if len(ranges) != 0 && ranges[len(ranges)-1].End == id {
			ranges[len(ranges)-1].End++
			continue
		}
		ranges = append(ranges, TokenRange{Start: id, End: id + 1})
	}
	return ranges
}

func isSerializedCodebookToken(text string) bool {
	const prefix = "IMGIMG"
	if !strings.HasPrefix(text, prefix) || len(text) <= len(prefix)+1 || text[len(text)-1] != 'Z' {
		return false
	}
	for _, value := range text[len(prefix) : len(text)-1] {
		if value < 'A' || value > 'I' {
			return false
		}
	}
	return true
}

// endOfGenerationNames are the control tokens upstream (llama.cpp
// llama-vocab.cpp, pinned commit) marks end-of-generation by name
// whatever the metadata declares: a converted GGUF whose eos points at
// <|endoftext|> still ends a turn at <|im_end|>, and gemma4 ends one at
// <turn|>. A model that emits one of these has stopped.
var endOfGenerationNames = map[string]bool{
	"<|eot_id|>": true, "<|im_end|>": true, "<|end|>": true,
	"<|return|>": true, "<|call|>": true, "<|flush|>": true, "<|calls|>": true,
	"<end_of_turn>": true, "<|endoftext|>": true, "</s>": true, "<|eom_id|>": true,
	"<EOT>": true, "_<EOT>": true, "[EOT]": true, "[EOS]": true,
	"<|end_of_text|>": true, "<end_of_utterance>": true,
	"<eos>": true, "<turn|>": true, "<|tool_response>": true,
	"<｜end▁of▁sentence｜>": true,
}

// endOfTurnNames are the names upstream promotes to the end-of-turn
// token when the metadata declares none, in the same precedence.
var endOfTurnNames = []string{
	"<|eot_id|>", "<|im_end|>", "<|end|>", "<end_of_turn>", "<|endoftext|>",
	"<|end_of_text|>", "<EOT>", "_<EOT>", "[EOT]", "<｜end▁of▁sentence｜>", "<end_of_utterance>",
}

// markEndOfGenerationByName ports the upstream name rule: every token
// whose text is a known end marker is end-of-generation, and an
// undeclared end-of-turn token is discovered by name.
func (v *Vocab) markEndOfGenerationByName() {
	if v == nil {
		return
	}
	v.eogByName = make(map[TokenID]bool)
	for id, token := range v.Tokens {
		if endOfGenerationNames[token.Text] {
			v.eogByName[TokenID(id)] = true
		}
	}
	if v.EOT != NullToken {
		return
	}
	for _, name := range endOfTurnNames {
		if id, ok := v.tokenToID[name]; ok {
			v.EOT = id
			return
		}
	}
}

// IsEOG reports declared terminal IDs.
func (v *Vocab) IsEOG(id TokenID) bool {
	if v == nil || id < 0 || int(id) >= len(v.Tokens) {
		return false
	}
	for _, terminal := range [...]TokenID{v.EOS, v.EOT, v.EOM} {
		if terminal != NullToken && id == terminal {
			return true
		}
	}
	if v.eogByName[id] {
		return true
	}
	return v.fimDeclared && (id == v.FIMPad || id == v.FIMRep || id == v.FIMSep)
}

func (v *Vocab) EOGTokens() []TokenID {
	if v == nil {
		return nil
	}
	var result []TokenID
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
