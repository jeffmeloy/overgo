package server

import (
	"net/http"
	"strconv"
	"strings"

	"overgo/internal/tokenizer"
)

// VocabularyInspectionAPI: read-only enumeration of the loaded vocabulary for
// the analysis workbench. Implemented by the inference Runner.
type VocabularyInspectionAPI interface {
	VocabularyLen() int
	VocabularyToken(tokenizer.TokenID) (tokenizer.Token, bool)
}

const (
	analyzeVocabDefaultLimit = 128
	analyzeVocabMaxLimit     = 1000
)

type analyzeVocabResponse struct {
	Size    int                 `json:"size"`    // total vocabulary entries
	Matched int                 `json:"matched"` // entries matching the query
	Offset  int                 `json:"offset"`
	Limit   int                 `json:"limit"`
	Query   string              `json:"query"`
	Tokens  []analyzeVocabToken `json:"tokens"`
}

type analyzeVocabToken struct {
	ID    int32   `json:"id"`
	Text  string  `json:"text"`
	Type  string  `json:"type"`
	Score float32 `json:"score"`
}

// analyzeVocab: a paged, optionally substring-filtered listing of the
// vocabulary. Pure enumeration of raw GGUF facts (id, text, type, score) — no
// statistics, nothing assumed about the data.
func (h *Handler) analyzeVocab(response http.ResponseWriter, request *http.Request) {
	if !requireMethod(response, request, http.MethodGet) {
		return
	}
	api, ok := h.generator.(VocabularyInspectionAPI)
	if !ok {
		writeError(response, http.StatusNotImplemented, "unsupported_operation", "vocabulary inspection is unavailable")
		return
	}
	size := api.VocabularyLen()
	query := request.URL.Query()
	offset := clampNonNegative(parseIntDefault(query.Get("offset"), 0))
	limit := parseIntDefault(query.Get("limit"), analyzeVocabDefaultLimit)
	if limit <= 0 {
		limit = analyzeVocabDefaultLimit
	}
	if limit > analyzeVocabMaxLimit {
		limit = analyzeVocabMaxLimit
	}
	needle := strings.ToLower(strings.TrimSpace(query.Get("query")))

	// One linear pass over the vocabulary: count matches and collect the page
	// window. The analysis surface is not on any hot path, so an O(size) scan
	// per request is fine and keeps the endpoint stateless.
	matched := 0
	tokens := make([]analyzeVocabToken, 0, limit)
	for id := 0; id < size; id++ {
		token, present := api.VocabularyToken(tokenizer.TokenID(id))
		if !present {
			continue
		}
		if needle != "" && !strings.Contains(strings.ToLower(token.Text), needle) {
			continue
		}
		if matched >= offset && len(tokens) < limit {
			tokens = append(tokens, analyzeVocabToken{
				ID:    int32(id),
				Text:  token.Text,
				Type:  tokenTypeName(token.Type),
				Score: token.Score,
			})
		}
		matched++
	}

	writeJSON(response, http.StatusOK, analyzeVocabResponse{
		Size:    size,
		Matched: matched,
		Offset:  offset,
		Limit:   limit,
		Query:   query.Get("query"),
		Tokens:  tokens,
	})
}

// tokenTypeName: the GGUF token-type flag as a stable lowercase label.
func tokenTypeName(kind tokenizer.TokenType) string {
	switch kind {
	case tokenizer.TokenNormal:
		return "normal"
	case tokenizer.TokenUnknown:
		return "unknown"
	case tokenizer.TokenControl:
		return "control"
	case tokenizer.TokenUserDefined:
		return "user_defined"
	case tokenizer.TokenUnused:
		return "unused"
	case tokenizer.TokenByte:
		return "byte"
	default:
		return "undefined"
	}
}

func parseIntDefault(raw string, fallback int) int {
	if raw == "" {
		return fallback
	}
	value, err := strconv.Atoi(raw)
	if err != nil {
		return fallback
	}
	return value
}

func clampNonNegative(value int) int {
	if value < 0 {
		return 0
	}
	return value
}
