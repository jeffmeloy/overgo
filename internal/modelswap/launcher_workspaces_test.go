package modelswap

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

// TestLauncherForwardsWorkspaceEnablement proves the operator's launch
// declaration reaches every served child: the serving arguments stay, and
// each enabled workspace adds exactly its flag.
func TestLauncherForwardsWorkspaceEnablement(t *testing.T) {
	for name, workspaces := range map[string]Workspaces{
		"none":          {},
		"training":      {Training: true},
		"model-builder": {ModelBuilder: true},
		"both":          {Training: true, ModelBuilder: true},
	} {
		t.Run(name, func(t *testing.T) {
			record := filepath.Join(t.TempDir(), "arguments")
			t.Setenv("OVERGO_TEST_MODEL_STARTUP", "arguments")
			t.Setenv("OVERGO_TEST_ARGUMENT_RECORD", record)
			supervisor, err := New(ServerLauncher{Binary: os.Args[0], Store: t.TempDir(), Workspaces: workspaces}, 0)
			if err != nil {
				t.Fatal(err)
			}
			defer supervisor.Close()
			proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"fixture": {Name: "fixture", Location: "fixture.gguf"}}}
			response := httptest.NewRecorder()
			proxy.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "http://localhost/health?swap=fixture", nil))
			if response.Code != http.StatusServiceUnavailable {
				t.Fatalf("startup = %d %s", response.Code, response.Body)
			}
			recorded, err := os.ReadFile(record)
			if err != nil {
				t.Fatal(err)
			}
			arguments := strings.Split(strings.TrimSpace(string(recorded)), "\n")
			for _, serving := range []string{"-listen", "-repo", "-model-id"} {
				if !slices.Contains(arguments, serving) {
					t.Fatalf("child lost serving argument %s: %q", serving, arguments)
				}
			}
			if arguments[len(arguments)-1] != "fixture.gguf" {
				t.Fatalf("child location is not last: %q", arguments)
			}
			if slices.Contains(arguments, "-training") != workspaces.Training || slices.Contains(arguments, "-model-builder") != workspaces.ModelBuilder {
				t.Fatalf("workspaces %+v forwarded as %q", workspaces, arguments)
			}
		})
	}
}
