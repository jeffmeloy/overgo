package server

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/remoteprovider"
	"overgo/internal/remoterelay"
)

const (
	bindingLimit    = 16
	bindingVariable = "PROVIDER_BINDING_KEY"
	bindingKey      = "declared-key"
	bindingCommit   = "0123456789abcdef0123456789abcdef01234567"
	listingRoute    = "/library/providers/models"
	registerRoute   = "/library/register"
)

// keyRecorder is a provider endpoint that says which keys reached it.
type keyRecorder struct {
	*httptest.Server
	mu   sync.Mutex
	keys []string
}

func newKeyRecorder(t *testing.T) *keyRecorder {
	t.Helper()
	recorder := &keyRecorder{}
	recorder.Server = httptest.NewServer(http.HandlerFunc(func(response http.ResponseWriter, request *http.Request) {
		recorder.mu.Lock()
		defer recorder.mu.Unlock()
		recorder.keys = append(recorder.keys, request.Header.Get("Authorization"))
		fmt.Fprint(response, `{"data":[{"id":"vendor/one"}]}`)
	}))
	t.Cleanup(recorder.Close)
	return recorder
}

func (r *keyRecorder) received() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return strings.Join(r.keys, ",")
}

// bindingIntake is the launcher's provider intake over a commit of the test's own.
func bindingIntake() LibraryIntake {
	return LibraryIntake{
		DeclareProvider: func(ctx context.Context, store *overgodb.Store, declaration ProviderDeclaration) ([]DeclaredProvider, error) {
			declarations, err := remoteprovider.DeclareDocument(ctx, store, bindingLimit, remoteprovider.Document{
				Name: declaration.Name, Endpoint: declaration.Endpoint, KeyEnvironment: declaration.KeyEnvironment, Models: declaration.Models,
			}, bindingCommit)
			declared := make([]DeclaredProvider, 0, len(declarations))
			for _, item := range declarations {
				declared = append(declared, DeclaredProvider{Location: item.Location, Model: item.Model.String()})
			}
			return declared, err
		},
		ListProviderModels: func(ctx context.Context, store *overgodb.Store, listing ProviderListing) ([]ProviderModel, error) {
			listed, err := remoterelay.ListModels(ctx, store, bindingLimit, remoterelay.Listing(listing), nil)
			models := make([]ProviderModel, 0, len(listed))
			for _, model := range listed {
				models = append(models, ProviderModel{ID: model.ID})
			}
			return models, err
		},
		RetireProvider: func(ctx context.Context, store *overgodb.Store, location, reason string) error {
			return remoteprovider.Retire(ctx, store, bindingLimit, location, bindingCommit, reason)
		},
	}
}

func providerRegistration(endpoint, variable string) string {
	return fmt.Sprintf(`{"kind":"provider","name":"bound","endpoint":%q,"key_environment":%q,"models":["vendor/one"]}`, endpoint, variable)
}

// TestAProviderKeyGoesOnlyToItsDeclaredEndpoint: both routes that could send a variable's value hold it to the endpoint a
// declaration bound it to, the server's own secrets go nowhere, and a key the page types reaches the endpoint it came with
// and no record. A retired declaration lets its variable go, as the plain server and the idle shell both do.
//
// Serial: it sets the key's variable.
func TestAProviderKeyGoesOnlyToItsDeclaredEndpoint(t *testing.T) {
	t.Setenv(bindingVariable, bindingKey)
	t.Setenv("OVERGO_HF_TOKEN", "hub-token")
	for name, serve := range map[string]func(*testing.T, *overgodb.Store) http.Handler{
		"server": func(t *testing.T, store *overgodb.Store) http.Handler {
			handler := newTestHandlerForRepository(t, store, &fakeGenerator{})
			handler.config.LibraryIntake = bindingIntake()
			return handler
		},
		"idle shell": func(_ *testing.T, store *overgodb.Store) http.Handler {
			return &IdleShell{Repository: store, Intake: bindingIntake()}
		},
	} {
		t.Run(name, func(t *testing.T) {
			store := overgodbtest.Open(t, t.TempDir())
			target := serve(t, store)
			declared, other := newKeyRecorder(t), newKeyRecorder(t)
			refused := func(what, path, body string) {
				t.Helper()
				before := declared.received() + other.received()
				response := serveTestRequest(target, http.MethodPost, path, body)
				if response.Code != http.StatusUnprocessableEntity || declared.received()+other.received() != before {
					t.Fatalf("%s = %d %s, keys sent: %q, %q", what, response.Code, response.Body, declared.received(), other.received())
				}
			}
			listing := func(endpoint, variable string) string {
				return fmt.Sprintf(`{"endpoint":%q,"key_environment":%q}`, endpoint, variable)
			}

			refused("an undeclared pair's listing", listingRoute, listing(declared.URL, bindingVariable))
			refused("the listing of a server secret", listingRoute, listing(declared.URL, "OVERGO_HF_TOKEN"))
			refused("the declaration of a server secret", registerRoute, providerRegistration(declared.URL, "OVERGO_HF_TOKEN"))

			if response := serveTestRequest(target, http.MethodPost, registerRoute, providerRegistration(declared.URL+"/", bindingVariable)); response.Code != http.StatusOK {
				t.Fatalf("declaration = %d %s", response.Code, response.Body)
			}
			if response := serveTestRequest(target, http.MethodPost, listingRoute, listing(declared.URL, bindingVariable)); response.Code != http.StatusOK ||
				!strings.Contains(response.Body.String(), "vendor/one") || declared.received() != "Bearer "+bindingKey {
				t.Fatalf("declared listing = %d %s, keys %q", response.Code, response.Body, declared.received())
			}
			refused("a listing at another endpoint", listingRoute, listing(other.URL, bindingVariable))
			refused("a declaration at another endpoint", registerRoute, providerRegistration(other.URL, bindingVariable))

			typed := fmt.Sprintf(`{"endpoint":%q,"key":"typed-key"}`, other.URL)
			response := serveTestRequest(target, http.MethodPost, listingRoute, typed)
			if response.Code != http.StatusOK || strings.Contains(response.Body.String(), "typed-key") || other.received() != "Bearer typed-key" {
				t.Fatalf("typed listing = %d %s, keys %q", response.Code, response.Body, other.received())
			}
			if both := serveTestRequest(target, http.MethodPost, listingRoute, fmt.Sprintf(`{"endpoint":%q,"key_environment":"K","key":"k"}`, other.URL)); both.Code != http.StatusBadRequest {
				t.Fatalf("a listing with both = %d %s", both.Code, both.Body)
			}

			if response := serveTestRequest(target, http.MethodPost, "/library/providers/retire", `{"location":"remote://bound/vendor/one","reason":"moved"}`); response.Code != http.StatusOK {
				t.Fatalf("retirement = %d %s", response.Code, response.Body)
			}
			if response := serveTestRequest(target, http.MethodPost, registerRoute, providerRegistration(other.URL, bindingVariable)); response.Code != http.StatusOK {
				t.Fatalf("declaration after the retirement = %d %s", response.Code, response.Body)
			}
		})
	}
}
