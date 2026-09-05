package server

import (
	"bytes"
	"compress/zlib"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"testing"
)

// TestFrontPageImport pins multimodal import (professional GUI campaign,
// gui-multimodal/import): the capability document declares the document
// kinds every served model accepts beside the media its projectors take,
// with the reason a missing kind is refused; a text document and a PDF
// travel as the Responses API's own file parts and reach the model as
// extracted text; a binary file is refused; and the composer imports by
// attach, drop and paste, refuses from the document's bounds before
// upload, and previews each kind.
func TestFrontPageImport(t *testing.T) {
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	manifest := serveTestRequest(handler, http.MethodGet, "/workspace/manifest", "")
	if manifest.Code != http.StatusOK {
		t.Fatalf("manifest status = %d body=%s", manifest.Code, manifest.Body.String())
	}
	var document struct {
		Model struct {
			Modalities map[string]bool `json:"modalities"`
			Media      struct {
				Accept        []string          `json:"accept"`
				Refusals      map[string]string `json:"refusals"`
				MaxMediaBytes uint64            `json:"max_media_bytes"`
			} `json:"media"`
		} `json:"model"`
	}
	if err := json.Unmarshal(manifest.Body.Bytes(), &document); err != nil {
		t.Fatal(err)
	}
	for _, kind := range []string{"text/plain", "text/markdown", "text/csv", "application/json", "application/pdf"} {
		if !slices.Contains(document.Model.Media.Accept, kind) {
			t.Errorf("capability document does not accept %s: %v", kind, document.Model.Media.Accept)
		}
	}
	if !document.Model.Modalities["document"] || document.Model.Media.Refusals["image"] == "" || document.Model.Media.MaxMediaBytes == 0 {
		t.Fatalf("capability document = %+v", document.Model)
	}

	pdf := testPDF(t, `BT /F1 12 Tf 72 700 Td (Hello ) Tj [(Wor) 20 (ld)] TJ T* (Second \(line\)) Tj T* <48656c6c6f> Tj (\101) Tj <00480065> Tj ET`)
	encode := func(mediaType string, data []byte) string {
		return "data:" + mediaType + ";base64," + base64.StdEncoding.EncodeToString(data)
	}
	body := `{"input":[{"role":"user","content":[{"type":"input_text","text":"summarize"},` +
		`{"type":"input_file","filename":"notes.md","file_data":"` + encode("text/markdown", []byte("# Title\nbody")) + `"},` +
		`{"type":"input_file","filename":"paper.pdf","file_data":"` + encode("application/pdf", pdf) + `"}]}],"max_output_tokens":1}`
	response := serveTestRequest(handler, http.MethodPost, "/v1/responses", body)
	if response.Code != http.StatusOK {
		t.Fatalf("response status = %d body=%s", response.Code, response.Body.String())
	}
	var stored responsesResponse
	if err := json.Unmarshal(response.Body.Bytes(), &stored); err != nil {
		t.Fatal(err)
	}
	messages := serveTestRequest(handler, http.MethodGet, "/interactions/messages?response="+stored.ID, "")
	var chain conversationMessagesResponse
	if err := json.Unmarshal(messages.Body.Bytes(), &chain); err != nil {
		t.Fatal(err)
	}
	if len(chain.Messages) == 0 {
		t.Fatalf("chain = %+v", chain)
	}
	prompt := chain.Messages[0].Content
	for _, needle := range []string{`[file name="notes.md"]`, "# Title\nbody", `[file name="paper.pdf"]`, "Hello World\nSecond (line)\nHelloA\n"} {
		if !strings.Contains(prompt, needle) {
			t.Errorf("prompt lacks %q:\n%s", needle, prompt)
		}
	}
	refused := serveTestRequest(handler, http.MethodPost, "/v1/responses",
		`{"input":[{"role":"user","content":[{"type":"input_file","filename":"x.bin","file_data":"`+encode("application/octet-stream", []byte("raw"))+`"}]}],"max_output_tokens":1}`)
	if refused.Code != http.StatusBadRequest {
		t.Fatalf("binary document status = %d body=%s", refused.Code, refused.Body.String())
	}

	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }
	composer := get("/composer.js")
	for _, needle := range []string{`"dragover"`, `"drop"`, `"paste"`, "media.refusals", "media.max_image_bytes", "media.max_media_bytes",
		"media.max_image_dimension", "media.max_image_pixels", `type: "input_file"`, "item.refusal", "readAsDataURL"} {
		if !strings.Contains(composer, needle) {
			t.Errorf("composer missing %q", needle)
		}
	}
	if !strings.Contains(get("/mod/chat.js"), `"input_file"`) {
		t.Error("chat does not pass file parts through")
	}
	if !strings.Contains(get("/style.css"), ".composer.drop") {
		t.Error("style lacks the drop target")
	}
}

// testPDF builds a one-page document whose content stream is deflated.
func testPDF(t *testing.T, content string) []byte {
	t.Helper()
	var deflated bytes.Buffer
	writer := zlib.NewWriter(&deflated)
	if _, err := writer.Write([]byte(content)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	var pdf bytes.Buffer
	pdf.WriteString("%PDF-1.4\n")
	pdf.WriteString("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
	pdf.WriteString("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
	pdf.WriteString("3 0 obj\n<< /Type /Page /Parent 2 0 R /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
	fmt.Fprintf(&pdf, "4 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n", deflated.Len())
	pdf.Write(deflated.Bytes())
	pdf.WriteString("\nendstream\nendobj\n")
	pdf.WriteString("5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
	pdf.WriteString("trailer\n<< /Root 1 0 R >>\n%%EOF\n")
	return pdf.Bytes()
}
