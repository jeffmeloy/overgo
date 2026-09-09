package modelswap

import (
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"overgo/internal/clioptions"
	"overgo/internal/processcontrol"
)

func TestMain(m *testing.M) {
	if mode := os.Getenv("OVERGO_TEST_MODEL_STARTUP"); mode != "" {
		clioptions.Main(func() error {
			if mode == "busy" {
				return fmt.Errorf("load device: %w", processcontrol.ErrResourceBusy)
			}
			return errors.New("invalid model bytes")
		})
		return
	}
	os.Exit(m.Run())
}

func TestModelSwapStartupBusy(t *testing.T) {
	for _, mode := range []string{"busy", "invalid"} {
		t.Run(mode, func(t *testing.T) {
			t.Setenv("OVERGO_TEST_MODEL_STARTUP", mode)
			supervisor, err := New(ServerLauncher{Binary: os.Args[0], Store: t.TempDir()}, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer supervisor.Close()
			proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"fixture": {Name: "fixture", Location: "fixture.gguf"}}}
			for range 2 { // Refusal releases the startup owner so an explicit retry can run.
				response := httptest.NewRecorder()
				proxy.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "http://localhost/health?swap=fixture", nil))
				if response.Code != http.StatusServiceUnavailable {
					t.Fatalf("startup = %d %s", response.Code, response.Body)
				}
				busy := strings.Contains(response.Body.String(), `"code":"resource_busy"`)
				if busy != (mode == "busy") {
					t.Fatalf("startup %s misclassified: %s", mode, response.Body)
				}
			}
		})
	}
}
