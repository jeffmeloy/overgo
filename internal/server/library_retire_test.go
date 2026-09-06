package server

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// TestLibraryProviderRetireRetiresThroughTheIntake: the retirement route
// needs a location and a reason, answers that it needs the launcher's
// intake, then retires through it; the intake's refusal is the route's.
func TestLibraryProviderRetireRetiresThroughTheIntake(t *testing.T) {
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, &fakeGenerator{})
	body := `{"location":"remote://fake/vendor/model","reason":"the endpoint closed"}`
	if incomplete := serveTestRequest(handler, http.MethodPost, "/library/providers/retire", `{"location":"remote://fake/vendor/model"}`); incomplete.Code != http.StatusBadRequest {
		t.Fatalf("retirement without a reason status=%d body=%s", incomplete.Code, incomplete.Body.String())
	}
	if absent := serveTestRequest(handler, http.MethodPost, "/library/providers/retire", body); absent.Code != http.StatusNotImplemented {
		t.Fatalf("retirement without the intake status=%d body=%s", absent.Code, absent.Body.String())
	}
	var retired []string
	handler.config.LibraryIntake.RetireProvider = func(_ context.Context, store *overgodb.Store, location, reason string) error {
		if store != repository || reason != "the endpoint closed" {
			return errors.New("the retirement lost a field")
		}
		if location != "remote://fake/vendor/model" {
			return errors.New("remote provider: no declared model at " + location)
		}
		retired = append(retired, location)
		return nil
	}
	if done := serveTestRequest(handler, http.MethodPost, "/library/providers/retire", body); done.Code != http.StatusOK || !strings.Contains(done.Body.String(), `"retired":true`) || len(retired) != 1 {
		t.Fatalf("retired status=%d body=%s retired=%v", done.Code, done.Body.String(), retired)
	}
	if refused := serveTestRequest(handler, http.MethodPost, "/library/providers/retire", strings.Replace(body, "vendor/model", "vendor/absent", 1)); refused.Code != http.StatusUnprocessableEntity ||
		!strings.Contains(refused.Body.String(), "no declared model") {
		t.Fatalf("refused status=%d body=%s", refused.Code, refused.Body.String())
	}
}
