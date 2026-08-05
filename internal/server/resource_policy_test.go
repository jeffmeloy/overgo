package server

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"image"
	"image/png"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRepositoryResponseFilePolicy(t *testing.T) {
	policy, err := LoadResponseFilePolicy(filepath.Join("..", "..", "resource_policy.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if policy.ResponseFiles.Enabled || len(policy.files) != 0 ||
		policy.ResponseTools.Hosted != "deny" || policy.ResponseTools.Custom != "deny" {
		t.Fatalf("default policy = %+v", policy)
	}
}

func TestResponseToolPolicyRejectsUnbackedAllow(t *testing.T) {
	root := t.TempDir()
	policyPath := filepath.Join(root, "policy.yaml")
	document := "schema: 1\nresponse_files:\n  enabled: false\n" +
		"  max_file_bytes: 1\n  allowed_roots: []\n  files: {}\n" +
		"response_tools:\n  hosted: allow\n  custom: deny\n"
	if err := os.WriteFile(policyPath, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadResponseFilePolicy(policyPath); err == nil || !strings.Contains(err.Error(), "external executor") {
		t.Fatalf("tool policy error = %v", err)
	}
}

func TestResponseFilePolicyLoadsAllowedFiles(t *testing.T) {
	root := t.TempDir()
	filePath := filepath.Join(root, "note.txt")
	if err := os.WriteFile(filePath, []byte("bounded note"), 0o600); err != nil {
		t.Fatal(err)
	}
	policyPath := filepath.Join(root, "policy.yaml")
	document := "schema: 1\nresponse_files:\n  enabled: true\n" +
		"  max_file_bytes: 32\n  allowed_roots:\n    - .\n  files:\n" +
		"    file_note:\n      path: note.txt\n      media_type: text/plain\n"
	if err := os.WriteFile(policyPath, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	policy, err := LoadResponseFilePolicy(policyPath)
	if err != nil {
		t.Fatal(err)
	}
	file, ok := policy.ResolveResponseFile("file_note")
	if !ok || string(file.Data) != "bounded note" || file.Filename != "note.txt" {
		t.Fatalf("resolved file = %+v, %v", file, ok)
	}
	file.Data[0] = 'X'
	again, _ := policy.ResolveResponseFile("file_note")
	if string(again.Data) != "bounded note" {
		t.Fatal("resolved resource shares mutable data")
	}
}

func TestResponseFilePolicyRejectsEscapesAndOversizedFiles(t *testing.T) {
	root := t.TempDir()
	allowed := filepath.Join(root, "allowed")
	if err := os.Mkdir(allowed, 0o700); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside.txt")
	if err := os.WriteFile(outside, []byte("outside"), 0o600); err != nil {
		t.Fatal(err)
	}
	policyPath := filepath.Join(root, "policy.yaml")
	document := "schema: 1\nresponse_files:\n  enabled: true\n" +
		"  max_file_bytes: 4\n  allowed_roots:\n    - allowed\n  files:\n" +
		"    file_escape:\n      path: outside.txt\n      media_type: text/plain\n"
	if err := os.WriteFile(policyPath, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadResponseFilePolicy(policyPath); err == nil || !strings.Contains(err.Error(), "outside allowed roots") {
		t.Fatalf("escape error = %v", err)
	}
	document = strings.ReplaceAll(document, "path: outside.txt", "path: allowed/large.txt")
	if err := os.WriteFile(filepath.Join(allowed, "large.txt"), []byte("large"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(policyPath, []byte(document), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadResponseFilePolicy(policyPath); err == nil || !strings.Contains(err.Error(), "byte limit") {
		t.Fatalf("size error = %v", err)
	}
}

type responseFileMap map[string]ResponseFile

func (files responseFileMap) ResolveResponseFile(id string) (ResponseFile, bool) {
	file, ok := files[id]
	return file, ok
}

func TestResponsesInputFileSources(t *testing.T) {
	remote := httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, _ *http.Request) {
		response.Header().Set("Content-Type", "application/json")
		_, _ = response.Write([]byte(`{"remote":true}`))
	}))
	defer remote.Close()
	mediaFetcher, err := newRemoteMediaFetcher(testRemoteMediaPolicy(t, remote.URL))
	if err != nil {
		t.Fatal(err)
	}
	handler := &Handler{
		mediaFetcher: mediaFetcher,
		responseFiles: responseFileMap{
			"file_note": {Data: []byte("mapped"), Filename: "mapped.txt", MediaType: "text/plain"},
		},
	}
	inline := "data:text/plain;base64," + base64.StdEncoding.EncodeToString([]byte("inline"))
	raw, err := json.Marshal([]any{
		map[string]any{"type": "input_file", "file_data": inline, "filename": "inline.txt"},
		map[string]any{"type": "input_file", "file_id": "file_note"},
		map[string]any{"type": "input_file", "file_url": remote.URL + "/data.json", "filename": "remote.json"},
	})
	if err != nil {
		t.Fatal(err)
	}
	messages, err := handler.parseResponsesMessages(context.Background(), raw, "")
	if err != nil {
		t.Fatal(err)
	}
	if len(messages) != 3 || !strings.Contains(messages[0].Content, "inline") ||
		!strings.Contains(messages[1].Content, "mapped") || !strings.Contains(messages[2].Content, `{"remote":true}`) {
		t.Fatalf("file messages = %+v", messages)
	}
}

func TestResponsesImageFileIDProjectsPrompt(t *testing.T) {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 1, 1))); err != nil {
		t.Fatal(err)
	}
	vision := &fakeQwen3VLProjector{}
	handler, err := New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, DefaultTemperature: testNeutralTemperature, DefaultTopP: testFullTopP,
		ImageProjector: vision,
		ResponseFiles: responseFileMap{
			"file_image": {Data: encoded.Bytes(), Filename: "pixel.png", MediaType: "image/png"},
		},
	}, &fakeGenerator{})
	if err != nil {
		t.Fatal(err)
	}
	body := `{"input":[{"role":"user","content":[` +
		`{"type":"input_text","text":"before"},` +
		`{"type":"input_image","file_id":"file_image"},` +
		`{"type":"input_text","text":"after"}]}],"max_output_tokens":1}`
	response := httptest.NewRecorder()
	handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
	if response.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
	}
	if vision.before != "before" || vision.after != "after" {
		t.Fatalf("projection = before=%q after=%q", vision.before, vision.after)
	}
}

func TestResponsesInputFileRejectsUnsafeContent(t *testing.T) {
	handler := &Handler{responseFiles: responseFileMap{
		"file_binary": {Data: []byte{0xff}, Filename: "bad.txt", MediaType: "text/plain"},
	}}
	raw := json.RawMessage(`[{"type":"input_file","file_id":"file_binary"}]`)
	if _, err := handler.parseResponsesMessages(context.Background(), raw, ""); err == nil {
		t.Fatal("invalid UTF-8 file accepted")
	}
}

func TestResponsesHostedAndCustomToolsUseExplicitPolicy(t *testing.T) {
	for _, toolType := range []string{"web_search_preview", "custom", "mcp"} {
		t.Run(toolType, func(t *testing.T) {
			handler := newTestHandler(t, &fakeGenerator{})
			body := `{"input":"test","tools":[{"type":"` + toolType + `"}]}`
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)))
			if response.Code != http.StatusBadRequest ||
				!strings.Contains(response.Body.String(), "external executor") ||
				!strings.Contains(response.Body.String(), "policy") {
				t.Fatalf("status = %d body=%s", response.Code, response.Body.String())
			}
		})
	}
}
