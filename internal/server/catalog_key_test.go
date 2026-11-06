package server

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync"
	"sync/atomic"
	"testing"

	"overgo/internal/discovery"
	"overgo/internal/overgodb"
	"overgo/internal/overgodbtest"
	"overgo/internal/testcheck"
)

// TestCatalogTakesTheKeyAndWritesNothing: the catalog shows disk paths, so it takes the key, and neither it nor the scorecard
// writes to the store on the request's path: each answers while the memo's publication is held, which then runs by itself.
func TestCatalogTakesTheKeyAndWritesNothing(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	handler := testcheck.Must(New(Config{
		ModelID: testModelID, MaxTokens: testMaxTokens, APIKey: testAPIKey, Repository: store,
		Scorecard: func(context.Context, *overgodb.Store, *discovery.Memo) ([]ScorecardRow, error) { return nil, nil },
	}, &fakeGenerator{}))(t)
	defer handler.Close()
	held := make(chan struct{})
	release := sync.OnceFunc(func() { close(held) })
	defer release()
	var published atomic.Int32
	handler.publishMemo = func(context.Context) error {
		<-held
		published.Add(1)
		return nil
	}
	read := func(path, credential string) *httptest.ResponseRecorder {
		request := httptest.NewRequest(http.MethodGet, path, nil)
		if credential != "" {
			request.Header.Set("Authorization", credential)
		}
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	head, _ := store.Head()
	if response := read("/catalog/models", ""); response.Code != http.StatusUnauthorized {
		t.Fatalf("a catalog without the key = %d %s", response.Code, response.Body)
	}
	for _, path := range []string{"/catalog/models", "/scorecard"} {
		if response := read(path, testBearerToken); response.Code != http.StatusOK {
			t.Fatalf("%s = %d %s", path, response.Code, response.Body)
		}
	}
	if after, _ := store.Head(); after != head {
		t.Fatal("a read wrote to the store on its request")
	}
	release()
	handler.memoWrites.Wait()
	if published.Load() == 0 {
		t.Fatal("what the reads hashed was never published")
	}
}

// TestACloseCancelsAPendingMemoPublication: a memo publication waits on the store, here on nothing but its context, as
// one waits on a writer lock another process holds; the handler's close ends the publication's lifetime before it waits
// for it, so close returns once the publication sees its cancellation.
func TestACloseCancelsAPendingMemoPublication(t *testing.T) {
	t.Parallel()
	store := overgodbtest.Open(t, t.TempDir())
	handler := testcheck.Must(New(Config{ModelID: testModelID, MaxTokens: testMaxTokens, APIKey: testAPIKey, Repository: store}, &fakeGenerator{}))(t)
	waiting := make(chan struct{})
	var seen error
	handler.publishMemo = func(ctx context.Context) error {
		close(waiting)
		<-ctx.Done()
		seen = context.Cause(ctx)
		return ctx.Err()
	}
	request := httptest.NewRequest(http.MethodGet, "/catalog/models", nil)
	request.Header.Set("Authorization", testBearerToken)
	handler.ServeHTTP(httptest.NewRecorder(), request)
	<-waiting
	testcheck.NoError(t, handler.Close())
	if !errors.Is(seen, errHandlerClosed) {
		t.Fatalf("the memo publication ended by %v, want the handler's close", seen)
	}
}
