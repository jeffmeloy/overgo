package server

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"overgo/internal/overgodb"
)

// TestProviderKeyRouteTakesTheLauncherIntake: the route answers that it
// needs the launcher's intake, then places the key through it and names
// the variable; the intake's refusal is the route's.
func TestProviderKeyRouteTakesTheLauncherIntake(t *testing.T) {
	repository, err := overgodb.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = repository.Close() })
	handler := newTestHandlerForRepository(t, repository, &fakeGenerator{})
	body := `{"location":"remote://fake/vendor/model","key":"secret"}`
	if absent := serveTestRequest(handler, http.MethodPost, "/providers/key", body); absent.Code != http.StatusNotImplemented {
		t.Fatalf("route without the intake status=%d body=%s", absent.Code, absent.Body.String())
	}
	handler.config.ProviderKeys = func(_ context.Context, store *overgodb.Store, reference, key string) (string, error) {
		if store != repository || reference != "remote://fake/vendor/model" {
			return "", errors.New("wrong intake arguments")
		}
		if key != "secret" {
			return "", errors.New("the provider refused the key")
		}
		return "OVERGO_ROUTE_TEST_KEY", nil
	}
	placed := serveTestRequest(handler, http.MethodPost, "/providers/key", body)
	if placed.Code != http.StatusOK || !strings.Contains(placed.Body.String(), `"key_environment":"OVERGO_ROUTE_TEST_KEY"`) ||
		!strings.Contains(placed.Body.String(), `"location":"remote://fake/vendor/model"`) {
		t.Fatalf("placed status=%d body=%s", placed.Code, placed.Body.String())
	}
	if refused := serveTestRequest(handler, http.MethodPost, "/providers/key", `{"location":"remote://fake/vendor/model","key":"wrong"}`); refused.Code != http.StatusUnprocessableEntity ||
		!strings.Contains(refused.Body.String(), "refused the key") {
		t.Fatalf("refused status=%d body=%s", refused.Code, refused.Body.String())
	}
}
