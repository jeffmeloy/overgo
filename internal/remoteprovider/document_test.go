package remoteprovider

import (
	"strings"
	"testing"

	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

const documentModelLimit = 16

// TestDeclareDocumentDeclaresEveryModel: a document declares one
// declaration per model over the one provider, in order; an empty model
// list is refused.
func TestDeclareDocumentDeclaresEveryModel(t *testing.T) {
	store := overgodbtest.Open(t, t.TempDir())
	testcheck.NoError(t, NameVariable(t.Context(), store, "OVERGO_DOCUMENT_TEST_KEY"))
	t.Setenv("OVERGO_DOCUMENT_TEST_KEY", "")
	document := Document{
		Name: "fake", Endpoint: "https://fake.example/api/v1", KeyEnvironment: "OVERGO_DOCUMENT_TEST_KEY",
		Models: []string{"vendor/one", "vendor/two"}, ContextLength: 4096,
	}
	declared, err := DeclareDocument(t.Context(), store, documentModelLimit, document, setKeyTestCommit)
	if err != nil || len(declared) != 2 || declared[0].Location != "remote://fake/vendor/one" || declared[1].Location != "remote://fake/vendor/two" ||
		declared[0].Provider.ContextLength != 4096 || Refusal(declared[1].Provider) != "OVERGO_DOCUMENT_TEST_KEY is not set" {
		t.Fatalf("declared = %+v, %v", declared, err)
	}
	listed, err := List(t.Context(), store, 16)
	if err != nil || len(listed) != 2 {
		t.Fatalf("listed = %+v, %v", listed, err)
	}
	if _, err := DeclareDocument(t.Context(), store, documentModelLimit, Document{Name: "fake", Endpoint: document.Endpoint, KeyEnvironment: document.KeyEnvironment}, setKeyTestCommit); err == nil ||
		!strings.Contains(err.Error(), "names no model") {
		t.Fatalf("empty document: %v", err)
	}
}
