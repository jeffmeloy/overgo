package server

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"overgo/internal/tokenizer"
)

// vocabGenerator: a fakeGenerator with a small known vocabulary for exercising
// the /analyze/vocab enumeration.
type vocabGenerator struct {
	fakeGenerator
	tokens []tokenizer.Token
}

func (g *vocabGenerator) VocabularyLen() int { return len(g.tokens) }

func (g *vocabGenerator) VocabularyToken(id tokenizer.TokenID) (tokenizer.Token, bool) {
	if id < 0 || int(id) >= len(g.tokens) {
		return tokenizer.Token{}, false
	}
	return g.tokens[id], true
}

func newVocabGenerator() *vocabGenerator {
	return &vocabGenerator{tokens: []tokenizer.Token{
		{Text: "<s>", Type: tokenizer.TokenControl},
		{Text: "the", Type: tokenizer.TokenNormal, Score: -1},
		{Text: "theory", Type: tokenizer.TokenNormal, Score: -2},
		{Text: "cat", Type: tokenizer.TokenNormal, Score: -3},
		{Text: "<0x0A>", Type: tokenizer.TokenByte},
	}}
}

func TestAnalyzeVocabPagesAndReportsTypes(t *testing.T) {
	handler := newTestHandler(t, newVocabGenerator())
	response := serveTestRequest(handler, http.MethodGet, "/analyze/vocab?offset=1&limit=2", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	var result analyzeVocabResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result.Size != 5 || result.Matched != 5 {
		t.Fatalf("size/matched = %d/%d, want 5/5", result.Size, result.Matched)
	}
	if len(result.Tokens) != 2 {
		t.Fatalf("page length = %d, want 2", len(result.Tokens))
	}
	if result.Tokens[0].ID != 1 || result.Tokens[0].Text != "the" || result.Tokens[0].Type != "normal" {
		t.Fatalf("first page token = %+v", result.Tokens[0])
	}
	if result.Tokens[1].ID != 2 || result.Tokens[1].Text != "theory" {
		t.Fatalf("second page token = %+v", result.Tokens[1])
	}
}

func TestAnalyzeVocabFiltersByQuery(t *testing.T) {
	handler := newTestHandler(t, newVocabGenerator())
	response := serveTestRequest(handler, http.MethodGet, "/analyze/vocab?query=THE", "")
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d", response.Code)
	}
	var result analyzeVocabResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	// Case-insensitive substring: "the" and "theory" match; the total size is
	// still reported unfiltered.
	if result.Size != 5 || result.Matched != 2 || len(result.Tokens) != 2 {
		t.Fatalf("size=%d matched=%d tokens=%d", result.Size, result.Matched, len(result.Tokens))
	}
	for _, token := range result.Tokens {
		if token.Text != "the" && token.Text != "theory" {
			t.Fatalf("unexpected match %q", token.Text)
		}
	}
}

func TestAnalyzeVocabClampsLimitAndTypeLabels(t *testing.T) {
	handler := newTestHandler(t, newVocabGenerator())
	response := serveTestRequest(handler, http.MethodGet, "/analyze/vocab?limit=100000", "")
	var result analyzeVocabResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if result.Limit != analyzeVocabMaxLimit {
		t.Fatalf("limit = %d, want clamp to %d", result.Limit, analyzeVocabMaxLimit)
	}
	// Control and byte flags survive to the wire.
	if result.Tokens[0].Type != "control" || result.Tokens[4].Type != "byte" {
		t.Fatalf("type labels = %q / %q", result.Tokens[0].Type, result.Tokens[4].Type)
	}
}

func TestAnalyzeVocabUnsupportedWhenNoVocabulary(t *testing.T) {
	handler := newTestHandler(t, &fakeGenerator{})
	response := serveTestRequest(handler, http.MethodGet, "/analyze/vocab", "")
	if response.Code != http.StatusNotImplemented {
		t.Fatalf("status = %d, want 501", response.Code)
	}
	var envelope apiErrorEnvelope
	if err := json.Unmarshal(response.Body.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	if envelope.Error.Type != errorCodeUnsupportedOperation {
		t.Fatalf("error type = %q", envelope.Error.Type)
	}
}

func TestAnalyzeVocabRejectsNonGet(t *testing.T) {
	handler := newTestHandler(t, newVocabGenerator())
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/analyze/vocab", nil))
	if response.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", response.Code)
	}
}
