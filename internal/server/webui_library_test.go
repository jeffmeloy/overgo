package server

import (
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestFrontPageLibrary pins the library lifecycle (professional GUI
// campaign, gui-library): a downloaded dataset directory registers under a
// name and lists in the catalog, a model register resolves the GGUF a
// download directory holds and refuses a directory that holds none or a
// file that is not a model, a validation is refused before its
// prerequisites (a registered model file, a committed Go tree), and the
// Library tab offers register and validate only in that order after a
// download succeeds.
func TestFrontPageLibrary(t *testing.T) {
	// The validation policy is a root document the server reads from its
	// working directory, the repository root every launcher runs it in.
	t.Chdir(filepath.Join("..", ".."))
	handler := newTestHandlerWithRepository(t, responseRecipeGenerator(t, &fakeGenerator{}))
	defer handler.Close()
	corpus := t.TempDir()
	if err := os.WriteFile(filepath.Join(corpus, "notes.txt"), []byte("the sea is wide\nthe sky is high\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	quoted := func(value string) string {
		data, _ := json.Marshal(value)
		return string(data)
	}
	registered := serveTestRequest(handler, http.MethodPost, "/library/register",
		`{"kind":"dataset","name":"lane-notes","directory":`+quoted(corpus)+`,"modality":"text"}`)
	if registered.Code != http.StatusOK {
		t.Fatalf("dataset register status = %d body=%s", registered.Code, registered.Body.String())
	}
	var receipt struct {
		Kind    string `json:"kind"`
		Dataset string `json:"dataset"`
		Files   uint64 `json:"files"`
	}
	if err := json.Unmarshal(registered.Body.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Kind != "dataset" || receipt.Dataset == "" || receipt.Files != 1 {
		t.Fatalf("dataset receipt = %+v", receipt)
	}
	listed := serveTestRequest(handler, http.MethodGet, "/datasets", "")
	if listed.Code != http.StatusOK || !strings.Contains(listed.Body.String(), `"name":"lane-notes"`) {
		t.Fatalf("datasets status = %d body=%s", listed.Code, listed.Body.String())
	}

	empty := t.TempDir()
	for _, body := range []string{
		`{"kind":"model","path":` + quoted(empty) + `}`,
		`{"kind":"model","path":` + quoted(filepath.Join(corpus, "notes.txt")) + `}`,
		`{"kind":"plugin","path":"x"}`,
	} {
		if refused := serveTestRequest(handler, http.MethodPost, "/library/register", body); refused.Code == http.StatusOK {
			t.Fatalf("register accepted %s", body)
		}
	}
	// The refusal must be the empty directory's, not the policy document's:
	// the root policy carries a doc field the decoder has to accept.
	refused := serveTestRequest(handler, http.MethodPost, "/library/validate", `{"path":`+quoted(empty)+`}`)
	if refused.Code != http.StatusBadRequest || strings.Contains(refused.Body.String(), "library_unavailable") {
		t.Fatalf("validate of an empty directory = %d %s", refused.Code, refused.Body.String())
	}

	get := func(path string) string { return serveTestRequest(handler, http.MethodGet, path, "").Body.String() }
	library := get("/mod/discovery.js")
	for _, needle := range []string{`"/library/register"`, `"/library/validate"`, `"/datasets/preview"`, `disabled: !stage.registered`, `job.state !== "succeeded"`, "register · validate"} {
		if !strings.Contains(library, needle) {
			t.Errorf("library module missing %q", needle)
		}
	}
	routes := get("/workspace/routes")
	for _, path := range []string{"/library/register", "/library/validate"} {
		if !strings.Contains(routes, path) {
			t.Errorf("route table lacks %s", path)
		}
	}
}
