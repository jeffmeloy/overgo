package tokenizer

import (
	"bytes"
	"container/heap"
	"errors"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"overgo/internal/binaryschema"
)

// EncodeOptions: controls special-token handling
type EncodeOptions struct {
	AddSpecial   bool
	ParseSpecial bool
}

type segment struct {
	text    string
	tokenID TokenID
}

// Encode: tokenizes text using GGUF vocabulary
func (v *Vocab) Encode(text string, options EncodeOptions) ([]TokenID, error) {
	if v == nil {
		return nil, errors.New("tokenizer: vocabulary is nil")
	}
	// A declared sequence extension owns encoding whenever it applies: always
	// under auto-tags, otherwise when the text carries the declared tags.
	// Non-tagged segments recurse into the base path tag-free, so this
	// cannot loop.
	if v.dna != nil && len(v.dna.specialTokens) >= dnaSpecialCount &&
		(v.dna.autoTags || strings.Contains(text, v.dna.specialTokens[dnaSpecialBegin]) ||
			strings.Contains(text, v.dna.specialTokens[dnaSpecialEnd])) {
		return v.encodeWithDNA(text, options)
	}
	output := make([]TokenID, 0, len(text)/3+2)
	if options.AddSpecial && (v.Model == "bert" || v.AddBOS) {
		output = append(output, v.BOS)
	}
	previousSpecial := true
	for _, item := range v.partitionSpecial(text, options.ParseSpecial) {
		if item.tokenID != NullToken {
			output = append(output, item.tokenID)
			previousSpecial = true
			continue
		}
		raw := item.text
		if v.Model == "llama" && v.AddPrefix && previousSpecial {
			raw = " " + raw
		}
		ids, err := v.encodeText(raw)
		if err != nil {
			return nil, err
		}
		output = append(output, ids...)
		previousSpecial = false
	}
	if options.AddSpecial {
		if v.Model == "bert" {
			output = append(output, v.SEP)
		} else if v.AddEOS {
			output = append(output, v.EOS)
		}
	}
	return output, nil
}

func (v *Vocab) encodeText(text string) ([]TokenID, error) {
	if text == "" {
		return nil, nil
	}
	if v.Model == "llama" {
		return v.encodeSPM(strings.ReplaceAll(text, " ", "▁"))
	}
	if v.Model == "t5" {
		return v.encodeUGM(text)
	}
	if v.Model == "bert" {
		return v.encodeWPM(text)
	}
	if v.Model == "gemma4" {
		return v.encodeGemma4(text)
	}
	words := preTokenizeFor(v.Pre, text)
	output := make([]TokenID, 0, len(words))
	for _, word := range words {
		encoded := encodeBytes([]byte(word))
		pieces := v.applyBPE(encoded)
		for _, piece := range pieces {
			if id, ok := v.tokenToID[piece]; ok {
				output = append(output, id)
				continue
			}
			// Match llama.cpp's last-resort lookup over encoded UTF-8 bytes
			for _, value := range []byte(piece) {
				id, ok := v.tokenToID[string([]byte{value})]
				if !ok {
					return nil, fmt.Errorf("tokenizer: no token for BPE piece %q (byte 0x%02x)", piece, value)
				}
				output = append(output, id)
			}
		}
	}
	return output, nil
}

func (v *Vocab) encodeGemma4(text string) ([]TokenID, error) {
	text = strings.ReplaceAll(text, " ", "\u2581")
	words := splitGemma4Newlines(text)
	output := make([]TokenID, 0, len(words))
	for _, word := range words {
		if strings.Trim(word, "\n") == "" {
			if id, ok := v.tokenToID[word]; ok {
				output = append(output, id)
				continue
			}
		}
		for _, piece := range v.applyBPE(word) {
			if id, ok := v.tokenToID[piece]; ok {
				output = append(output, id)
				continue
			}
			for _, value := range []byte(piece) {
				byteToken := fmt.Sprintf("<0x%02X>", value)
				id, ok := v.tokenToID[byteToken]
				if !ok {
					return nil, fmt.Errorf("tokenizer: no token for Gemma 4 BPE piece %q (byte 0x%02X)", piece, value)
				}
				output = append(output, id)
			}
		}
	}
	return output, nil
}

func splitGemma4Newlines(text string) []string {
	if text == "" {
		return nil
	}
	var result []string
	for start := 0; start < len(text); {
		newline := text[start] == '\n'
		end := start + 1
		for end < len(text) && (text[end] == '\n') == newline {
			end++
		}
		result = append(result, text[start:end])
		start = end
	}
	return result
}

// applyBPE merges a word's runes by ascending merge rank, the leftmost
// pair first among equal ranks. Candidate merges wait in a heap keyed
// by (rank, left position) and the symbols form a linked list, so each
// merge costs a heap operation instead of a scan of every adjacent
// pair: a Gemma 4 "word" is a whole line, and a 12,000-character
// narrative line took two seconds per encode under the pair scan (the
// MuSR scoring pass encoded each prompt four times per case, eight
// seconds of tokenization against half a second of model work).
func (v *Vocab) applyBPE(word string) []string {
	if v.IgnoreMerges {
		if _, ok := v.tokenToID[word]; ok {
			return []string{word}
		}
	}
	// The symbol list and bigram queue are the SPM encoder's: a merge
	// rank becomes a descending score, so the queue's highest-score,
	// leftmost order is the scan's lowest-rank, leftmost order, and a
	// bigram whose symbols no longer sum to its recorded size is stale.
	symbols := make([]spmSymbol, 0, utf8.RuneCountInString(word))
	for _, symbol := range word {
		index := len(symbols)
		symbols = append(symbols, spmSymbol{previous: index - 1, next: index + 1, text: string(symbol)})
	}
	if len(symbols) == 0 {
		return nil
	}
	symbols[len(symbols)-1].next = -1
	queue := make(spmQueue, 0, len(symbols))
	push := func(left, right int) {
		if left < 0 || right < 0 {
			return
		}
		rank, ok := v.mergeRank[pair{left: symbols[left].text, right: symbols[right].text}]
		if !ok {
			return
		}
		heap.Push(&queue, spmBigram{
			left: left, right: right, score: -float32(rank),
			size: len(symbols[left].text) + len(symbols[right].text),
		})
	}
	for index := 1; index < len(symbols); index++ {
		push(index-1, index)
	}
	for queue.Len() > 0 {
		bigram := heap.Pop(&queue).(spmBigram)
		left, right := &symbols[bigram.left], &symbols[bigram.right]
		if left.text == "" || right.text == "" || len(left.text)+len(right.text) != bigram.size {
			continue
		}
		left.text += right.text
		right.text = ""
		left.next = right.next
		if right.next >= 0 {
			symbols[right.next].previous = bigram.left
		}
		push(left.previous, bigram.left)
		push(bigram.left, left.next)
	}
	result := make([]string, 0, len(symbols))
	for index := 0; index >= 0; index = symbols[index].next {
		result = append(result, symbols[index].text)
	}
	return result
}

func (v *Vocab) partitionSpecial(text string, parseSpecial bool) []segment {
	if text == "" {
		return nil
	}
	eligible := make([]TokenID, 0, len(v.special))
	for _, id := range v.special {
		kind := v.Tokens[id].Type
		if parseSpecial || kind == TokenUserDefined {
			eligible = append(eligible, id)
		}
	}
	if len(eligible) == 0 {
		return []segment{{text: text, tokenID: NullToken}}
	}

	var result []segment
	for offset := 0; offset < len(text); {
		matchAt := -1
		matchID := NullToken
		for _, id := range eligible {
			index := strings.Index(text[offset:], v.Tokens[id].Text)
			if index < 0 {
				continue
			}
			index += offset
			if matchAt < 0 || index < matchAt ||
				(index == matchAt && len(v.Tokens[id].Text) > len(v.Tokens[matchID].Text)) {
				matchAt = index
				matchID = id
			}
		}
		if matchAt < 0 {
			result = append(result, segment{text: text[offset:], tokenID: NullToken})
			break
		}
		if matchAt > offset {
			result = append(result, segment{text: text[offset:matchAt], tokenID: NullToken})
		}
		result = append(result, segment{tokenID: matchID})
		offset = matchAt + len(v.Tokens[matchID].Text)
	}
	return result
}

// Decode: converts token IDs back to UTF-8; Control and unknown tokens are
// omitted unless includeSpecial is true
func (v *Vocab) Decode(ids []TokenID, includeSpecial bool) (string, error) {
	if v == nil {
		return "", errors.New("tokenizer: vocabulary is nil")
	}
	var output bytes.Buffer
	for _, id := range ids {
		piece, err := v.DecodePiece(id, includeSpecial)
		if err != nil {
			return "", err
		}
		output.WriteString(piece)
	}
	result := output.String()
	if (v.Model == "llama" || v.Model == "t5") && v.AddPrefix {
		result = strings.TrimPrefix(result, " ")
	}
	if v.Model == "bert" {
		result = cleanWPMSpaces(result)
	}
	return result, nil
}

// DecodePiece decodes one token without sequence-initial whitespace removal
// Streaming callers should use this instead of Decode on singleton slice
func (v *Vocab) DecodePiece(id TokenID, includeSpecial bool) (string, error) {
	if v == nil {
		return "", errors.New("tokenizer: vocabulary is nil")
	}
	if id < 0 || int(id) >= len(v.Tokens) {
		return "", fmt.Errorf("tokenizer: token ID %d is out of range", id)
	}
	if piece, ok := v.dna.piece(id); ok {
		return piece, nil
	}
	token := v.Tokens[id]
	switch token.Type {
	case TokenControl, TokenUnknown:
		if includeSpecial {
			return token.Text, nil
		}
		return "", nil
	case TokenByte:
		value, err := parseByteToken(token.Text)
		if err != nil {
			return "", fmt.Errorf("tokenizer: token ID %d: %w", id, err)
		}
		return string([]byte{value}), nil
	case TokenUnused, TokenUndefined:
		return "", nil
	default:
		if v.Model == "llama" || v.Model == "t5" || v.Model == "bert" || v.Model == "gemma4" {
			return strings.ReplaceAll(token.Text, "▁", " "), nil
		}
		decoded, err := decodeBytes(token.Text)
		if err != nil {
			return "", fmt.Errorf("tokenizer: token ID %d: %w", id, err)
		}
		return string(decoded), nil
	}
}

func parseByteToken(text string) (byte, error) {
	if len(text) != 6 || !strings.HasPrefix(text, "<0x") || text[5] != '>' {
		return 0, fmt.Errorf("invalid byte token %q", text)
	}
	const hex = "0123456789abcdef"
	high := strings.IndexByte(hex, byte(unicode.ToLower(rune(text[3]))))
	low := strings.IndexByte(hex, byte(unicode.ToLower(rune(text[4]))))
	if high < 0 || low < 0 {
		return 0, fmt.Errorf("invalid byte token %q", text)
	}
	return byte(high<<4 | low), nil
}

func encodeBytes(data []byte) string {
	var output strings.Builder
	output.Grow(len(data) * 2)
	for _, value := range data {
		output.WriteRune(byteEncoder[value])
	}
	return output.String()
}

func decodeBytes(text string) ([]byte, error) {
	output := make([]byte, 0, len(text))
	for _, value := range text {
		decoded, ok := byteDecoder[value]
		if !ok {
			return nil, fmt.Errorf("BPE token contains unmapped rune U+%04X", value)
		}
		output = append(output, decoded)
	}
	return output, nil
}

var byteEncoder, byteDecoder = makeByteCodec()

func makeByteCodec() ([binaryschema.ByteValueCount]rune, map[rune]byte) {
	var encoder [binaryschema.ByteValueCount]rune
	used := make(map[int]bool, binaryschema.ByteValueCount)
	for value := 0x21; value <= 0x7e; value++ {
		encoder[value] = rune(value)
		used[value] = true
	}
	for value := 0xa1; value <= 0xac; value++ {
		encoder[value] = rune(value)
		used[value] = true
	}
	for value := 0xae; value <= 0xff; value++ {
		encoder[value] = rune(value)
		used[value] = true
	}
	next := 0
	for value := range binaryschema.ByteValueCount {
		if !used[value] {
			encoder[value] = rune(binaryschema.ByteValueCount + next)
			next++
		}
	}
	decoder := make(map[rune]byte, binaryschema.ByteValueCount)
	for value, encoded := range encoder {
		decoder[encoded] = byte(value)
	}
	return encoder, decoder
}

func preTokenizeGPT2(text string) []string {
	return preTokenize(text, false)
}

func preTokenizeQwen2(text string) []string {
	return preTokenize(text, true)
}

const deepSeekLLMLetterClass = "A-Za-zµÀ-ÖØ-öø-ƺƼ-ƿǄ-ʓʕ-ʯͰ-ͳͶͷͻ-ͽͿΆΈ-ΊΌΎ-ΡΣ-ϵϷ-ҁҊ-ԯԱ-ՖႠ-ჅᎠ-Ᏽᏸ-ᏽᲐ-ᲺᲽ-Ჿᴀ-ᴫᵫ-ᵷᵹ-ᶚḀ-ἕἘ-Ἕἠ-ὅὈ-Ὅὐ-ὗὙὛὝὟ-ώᾀ-ᾴᾶ-ᾼιῂ-ῄῆ-ῌῐ-ΐῖ-Ίῠ-Ῥῲ-ῴῶ-ῼℂℇℊ-ℓℕℙ-ℝℤΩℨK-ℭℯ-ℴℹℼ-ℿⅅ-ⅉⅎↃↄⰀ-ⱻⱾ-ⳤⳫ-ⳮⳲⳳꙀ-ꙭꚀ-ꚛꜢ-ꝯꝱ-ꞇꞋ-ꞎꭰ-ꮿﬀ-ﬆﬓ-ﬗＡ-Ｚａ-ｚ𐐀-𐑏𐒰-𐓓𐓘-𐓻𐲀-𐲲𐳀-𐳲𑢠-𑣟𞤀-𞥃"

var deepSeekLLMLetterRanges = parseRuneRanges(deepSeekLLMLetterClass)

func preTokenizeDeepSeekLLM(text string) []string {
	parts := []string{text}
	parts = splitDeepSeekParts(parts, func(values []rune, position int) int {
		if values[position] == '\r' || values[position] == '\n' {
			return 1
		}
		return 0
	})
	parts = splitDeepSeekParts(parts, func(values []rune, position int) int {
		start := position
		if unicode.IsSpace(values[position]) {
			position++
		}
		if position >= len(values) || !isDeepSeekLLMLetter(values[position]) {
			return 0
		}
		for position < len(values) && isDeepSeekLLMLetter(values[position]) {
			position++
		}
		return position - start
	})
	parts = splitDeepSeekParts(parts, func(values []rune, position int) int {
		start := position
		if unicode.IsSpace(values[position]) {
			position++
		}
		if position >= len(values) || !isDeepSeekLLMPunctuation(values[position]) {
			return 0
		}
		for position < len(values) && isDeepSeekLLMPunctuation(values[position]) {
			position++
		}
		return position - start
	})
	parts = splitDeepSeekParts(parts, func(values []rune, position int) int {
		if !unicode.IsSpace(values[position]) {
			return 0
		}
		end := position
		for end < len(values) && unicode.IsSpace(values[end]) {
			end++
		}
		if end != len(values) {
			return 0
		}
		return end - position
	})
	parts = splitDeepSeekParts(parts, func(values []rune, position int) int {
		if !isDeepSeekLLMCJK(values[position]) {
			return 0
		}
		end := position + 1
		for end < len(values) && isDeepSeekLLMCJK(values[end]) {
			end++
		}
		return end - position
	})
	return splitDeepSeekParts(parts, func(values []rune, position int) int {
		if !unicode.IsNumber(values[position]) {
			return 0
		}
		end := position + 1
		for end < len(values) && unicode.IsNumber(values[end]) {
			end++
		}
		return end - position
	})
}

func splitDeepSeekParts(parts []string, match func([]rune, int) int) []string {
	output := make([]string, 0, len(parts)*2)
	for _, part := range parts {
		values := []rune(part)
		unmatched := 0
		for position := 0; position < len(values); {
			length := match(values, position)
			if length == 0 {
				position++
				continue
			}
			if unmatched < position {
				output = append(output, string(values[unmatched:position]))
			}
			output = append(output, string(values[position:position+length]))
			position += length
			unmatched = position
		}
		if unmatched < len(values) {
			output = append(output, string(values[unmatched:]))
		}
	}
	return output
}

func parseRuneRanges(class string) [][2]rune {
	values := []rune(class)
	ranges := make([][2]rune, 0, len(values)/2)
	for position := 0; position < len(values); {
		start, end := values[position], values[position]
		if position+2 < len(values) && values[position+1] == '-' {
			end = values[position+2]
			position += 3
		} else {
			position++
		}
		ranges = append(ranges, [2]rune{start, end})
	}
	return ranges
}

func isDeepSeekLLMLetter(value rune) bool {
	for _, span := range deepSeekLLMLetterRanges {
		if value >= span[0] && value <= span[1] {
			return true
		}
	}
	return false
}

func isDeepSeekLLMPunctuation(value rune) bool {
	return value >= '!' && value <= '/' || value >= ':' && value <= '~' ||
		value >= '！' && value <= '／' || value >= '：' && value <= '～' ||
		value >= '‘' && value <= '‟' || value >= '　' && value <= '。'
}

func isDeepSeekLLMCJK(value rune) bool {
	return value >= '一' && value <= '龥' || value >= 'ࠀ' && value <= '一' ||
		value >= '가' && value <= '퟿'
}

func preTokenizeQwen35(text string) []string {
	values := []rune(text)
	output := make([]string, 0, len(values)/3+1)
	for position := 0; position < len(values); {
		start := position
		if values[position] == '\'' {
			if count := contractionLength(values[position:], true); count > 0 {
				position += count
				output = append(output, string(values[start:position]))
				continue
			}
		}

		current := values[position]
		if current != '\r' && current != '\n' && !unicode.IsNumber(current) &&
			(isLetterOrMark(current) ||
				position+1 < len(values) && isLetterOrMark(values[position+1])) {
			position++
			for position < len(values) && isLetterOrMark(values[position]) {
				position++
			}
			output = append(output, string(values[start:position]))
			continue
		}
		if unicode.IsNumber(current) {
			position++
			output = append(output, string(values[start:position]))
			continue
		}

		content := position
		if current == ' ' {
			content++
		}
		if content < len(values) && isQwen35NonWord(values[content]) {
			position = content + 1
			for position < len(values) && isQwen35NonWord(values[position]) {
				position++
			}
			for position < len(values) &&
				(values[position] == '\r' || values[position] == '\n') {
				position++
			}
			output = append(output, string(values[start:position]))
			continue
		}

		if unicode.IsSpace(current) {
			end := position
			lastNewline := -1
			for end < len(values) && unicode.IsSpace(values[end]) {
				if values[end] == '\r' || values[end] == '\n' {
					lastNewline = end + 1
				}
				end++
			}
			if lastNewline >= 0 {
				position = lastNewline
			} else if end < len(values) && end-position > 1 {
				position = end - 1
			} else {
				position = end
			}
			output = append(output, string(values[start:position]))
			continue
		}

		position++
		output = append(output, string(values[start:position]))
	}
	return output
}

func preTokenizeGPT4O(text string) []string {
	values := []rune(text)
	output := make([]string, 0, len(values)/3+1)
	for position := 0; position < len(values); {
		start := position
		letterStart := position
		if values[position] != '\r' && values[position] != '\n' &&
			!unicode.IsLetter(values[position]) && !unicode.IsNumber(values[position]) &&
			position+1 < len(values) && isLetterOrMark(values[position+1]) {
			letterStart++
		}
		if letterStart < len(values) && isLetterOrMark(values[letterStart]) {
			position = letterStart + 1
			for position < len(values) && isLetterOrMark(values[position]) {
				position++
			}
			if count := contractionLength(values[position:], true); count > 0 {
				position += count
			}
			output = append(output, string(values[start:position]))
			continue
		}
		if unicode.IsNumber(values[position]) {
			position++
			for position < len(values) && position-start < 3 && unicode.IsNumber(values[position]) {
				position++
			}
			output = append(output, string(values[start:position]))
			continue
		}
		content := position
		if values[position] == ' ' {
			content++
		}
		if content < len(values) && isQwen35NonWord(values[content]) {
			position = content + 1
			for position < len(values) && isQwen35NonWord(values[position]) {
				position++
			}
			for position < len(values) && (values[position] == '\r' || values[position] == '\n') {
				position++
			}
			output = append(output, string(values[start:position]))
			continue
		}
		if unicode.IsSpace(values[position]) {
			end := position
			lastNewline := -1
			for end < len(values) && unicode.IsSpace(values[end]) {
				if values[end] == '\r' || values[end] == '\n' {
					lastNewline = end + 1
				}
				end++
			}
			if lastNewline >= 0 {
				position = lastNewline
			} else if end < len(values) && end-position > 1 {
				position = end - 1
			} else {
				position = end
			}
			output = append(output, string(values[start:position]))
			continue
		}
		position++
		output = append(output, string(values[start:position]))
	}
	return output
}

func isLetterOrMark(value rune) bool {
	return unicode.IsLetter(value) || unicode.Is(unicode.M, value)
}

func isQwen35NonWord(value rune) bool {
	return !unicode.IsSpace(value) &&
		!unicode.IsLetter(value) &&
		!unicode.Is(unicode.M, value) &&
		!unicode.IsNumber(value)
}

func preTokenizeLlama3(text string) []string {
	values := []rune(text)
	output := make([]string, 0, len(values)/3+1)
	for position := 0; position < len(values); {
		start := position
		if values[position] == '\'' {
			if count := contractionLength(values[position:], true); count > 0 {
				position += count
				output = append(output, string(values[start:position]))
				continue
			}
		}

		letterStart := position
		if values[position] != '\r' && values[position] != '\n' &&
			!unicode.IsLetter(values[position]) && !unicode.IsNumber(values[position]) &&
			position+1 < len(values) && unicode.IsLetter(values[position+1]) {
			letterStart++
		}
		if unicode.IsLetter(values[letterStart]) {
			position = letterStart + 1
			for position < len(values) && unicode.IsLetter(values[position]) {
				position++
			}
			output = append(output, string(values[start:position]))
			continue
		}

		if unicode.IsNumber(values[position]) {
			position++
			for position < len(values) && position-start < 3 && unicode.IsNumber(values[position]) {
				position++
			}
			output = append(output, string(values[start:position]))
			continue
		}

		punctuationStart := position
		if values[position] == ' ' && position+1 < len(values) && isNonWord(values[position+1]) {
			punctuationStart++
		}
		if punctuationStart < len(values) && isNonWord(values[punctuationStart]) {
			position = punctuationStart + 1
			for position < len(values) && isNonWord(values[position]) {
				position++
			}
			for position < len(values) && (values[position] == '\r' || values[position] == '\n') {
				position++
			}
			output = append(output, string(values[start:position]))
			continue
		}

		if unicode.IsSpace(values[position]) {
			end := position
			lastNewline := -1
			for end < len(values) && unicode.IsSpace(values[end]) {
				if values[end] == '\r' || values[end] == '\n' {
					lastNewline = end + 1
				}
				end++
			}
			if lastNewline >= 0 {
				position = lastNewline
			} else if end < len(values) && end-position > 1 {
				position = end - 1
			} else {
				position = end
			}
			output = append(output, string(values[start:position]))
			continue
		}

		position++
		output = append(output, string(values[start:position]))
	}
	return output
}

func preTokenize(text string, qwen2 bool) []string {
	runes := []rune(text)
	output := make([]string, 0, len(runes)/3+1)
	for position := 0; position < len(runes); {
		start := position
		current := runes[position]

		if current == '\'' {
			limit := contractionLength(runes[position:], qwen2)
			if limit > 0 {
				position += limit
				output = append(output, string(runes[start:position]))
				continue
			}
		}

		if qwen2 {
			if current != '\r' && current != '\n' && !unicode.IsNumber(current) &&
				(unicode.IsLetter(current) || position+1 < len(runes) && unicode.IsLetter(runes[position+1])) {
				position++
				for position < len(runes) && unicode.IsLetter(runes[position]) {
					position++
				}
				output = append(output, string(runes[start:position]))
				continue
			}
			if unicode.IsNumber(current) {
				position++
				output = append(output, string(runes[start:position]))
				continue
			}
		} else {
			content := position
			if current == ' ' {
				content++
			}
			if content < len(runes) && unicode.IsLetter(runes[content]) {
				position = content + 1
				for position < len(runes) && unicode.IsLetter(runes[position]) {
					position++
				}
				output = append(output, string(runes[start:position]))
				continue
			}
			if content < len(runes) && unicode.IsNumber(runes[content]) {
				position = content + 1
				for position < len(runes) && unicode.IsNumber(runes[position]) {
					position++
				}
				output = append(output, string(runes[start:position]))
				continue
			}
		}

		content := position
		if current == ' ' {
			content++
		}
		if content < len(runes) && isNonWord(runes[content]) {
			position = content + 1
			for position < len(runes) && isNonWord(runes[position]) {
				position++
			}
			if qwen2 {
				for position < len(runes) && (runes[position] == '\r' || runes[position] == '\n') {
					position++
				}
			}
			output = append(output, string(runes[start:position]))
			continue
		}

		whitespaceEnd := position
		lastNewline := -1
		for whitespaceEnd < len(runes) && unicode.IsSpace(runes[whitespaceEnd]) {
			if runes[whitespaceEnd] == '\r' || runes[whitespaceEnd] == '\n' {
				lastNewline = whitespaceEnd + 1
			}
			whitespaceEnd++
		}
		if qwen2 && lastNewline >= 0 {
			position = lastNewline
			output = append(output, string(runes[start:position]))
			continue
		}
		count := whitespaceEnd - position
		if count > 1 && whitespaceEnd < len(runes) {
			position += count - 1
			output = append(output, string(runes[start:position]))
			continue
		}
		if count > 0 {
			position = whitespaceEnd
			output = append(output, string(runes[start:position]))
			continue
		}

		position++
		output = append(output, string(runes[start:position]))
	}
	return output
}

func contractionLength(values []rune, caseInsensitive bool) int {
	if len(values) < 2 || values[0] != '\'' {
		return 0
	}
	first := values[1]
	if caseInsensitive {
		first = unicode.ToLower(first)
	}
	if first == 's' || first == 't' || first == 'm' || first == 'd' {
		return 2
	}
	if len(values) < 3 {
		return 0
	}
	second := values[2]
	if caseInsensitive {
		second = unicode.ToLower(second)
	}
	if first == 'r' && second == 'e' || first == 'v' && second == 'e' || first == 'l' && second == 'l' {
		return 3
	}
	return 0
}

func isNonWord(value rune) bool {
	return !unicode.IsSpace(value) && !unicode.IsLetter(value) && !unicode.IsNumber(value)
}
