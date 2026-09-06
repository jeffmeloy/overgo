package server

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// TestLibraryRegisterDeclaresProvidersThroughTheIntake: the provider kind
// answers that it needs the launcher's intake, then declares through it
// with the request's fields and answers the declared models; the intake's
// refusal is the route's.
func TestLibraryRegisterDeclaresProvidersThroughTheIntake(t *testing.T) {
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, &fakeGenerator{})
	body := `{"kind":"provider","name":"fake","endpoint":"https://fake.example/api/v1","key_environment":"OVERGO_LIBRARY_TEST_KEY","models":["vendor/one","vendor/two"],"context_length":4096}`
	if absent := serveTestRequest(handler, http.MethodPost, "/library/register", body); absent.Code != http.StatusNotImplemented {
		t.Fatalf("provider kind without the intake status=%d body=%s", absent.Code, absent.Body.String())
	}
	handler.config.LibraryIntake.DeclareProvider = func(_ context.Context, store *overgodb.Store, declaration ProviderDeclaration) ([]DeclaredProvider, error) {
		if store != repository || declaration.Name != "fake" || declaration.Endpoint != "https://fake.example/api/v1" ||
			declaration.KeyEnvironment != "OVERGO_LIBRARY_TEST_KEY" || len(declaration.Models) != 2 || declaration.ContextLength != 4096 {
			return nil, errors.New("the declaration lost a field")
		}
		if declaration.Models[0] == "refuse" {
			return nil, errors.New("the provider refused the declaration")
		}
		declared := make([]DeclaredProvider, 0, len(declaration.Models))
		for _, model := range declaration.Models {
			declared = append(declared, DeclaredProvider{Location: "remote://fake/" + model, Model: "model:x", Recipe: "recipe:y", Refusal: "OVERGO_LIBRARY_TEST_KEY is not set"})
		}
		return declared, nil
	}
	declared := serveTestRequest(handler, http.MethodPost, "/library/register", body)
	if declared.Code != http.StatusOK || !strings.Contains(declared.Body.String(), `"location":"remote://fake/vendor/two"`) ||
		!strings.Contains(declared.Body.String(), `"refusal":"OVERGO_LIBRARY_TEST_KEY is not set"`) || !strings.Contains(declared.Body.String(), `"kind":"provider"`) {
		t.Fatalf("declared status=%d body=%s", declared.Code, declared.Body.String())
	}
	refused := serveTestRequest(handler, http.MethodPost, "/library/register", strings.Replace(body, `"vendor/one"`, `"refuse"`, 1))
	if refused.Code != http.StatusUnprocessableEntity || !strings.Contains(refused.Body.String(), "refused the declaration") {
		t.Fatalf("refused status=%d body=%s", refused.Code, refused.Body.String())
	}
}
