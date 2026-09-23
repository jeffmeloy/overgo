package server

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// TestAgentRoutesHaveOnePrefix holds the agent family to /agents: the
// route table once served /agent/tools, /agent/sessions and
// /agent/provenance beside every other /agents route, and /agent/step and
// /agent/approval beside /agents/step and /agents/approval, which differed
// only in requiring an agent name. The client names none of the singular
// routes either.
func TestAgentRoutesHaveOnePrefix(t *testing.T) {
	t.Parallel()
	agents := 0
	for _, route := range routeCatalog {
		if strings.HasPrefix(route.Path, "/agent/") || route.Path == "/agent" {
			t.Errorf("route %s is outside the /agents prefix", route.Path)
		}
		if strings.HasPrefix(route.Path, "/agents") {
			agents++
		}
	}
	if agents == 0 {
		t.Fatal("the route table holds no /agents route")
	}
	singular := func(source string) bool {
		return strings.Contains(source, `"/agent/`) || strings.Contains(source, `'/agent/`)
	}
	for name, source := range webuiJavaScript(t) {
		if singular(source) {
			t.Errorf("%s calls a singular /agent/ route", name)
		}
	}
	// A page test that stubs a retired path waits forever for a request the page never makes.
	tests, err := filepath.Glob("*_test.go")
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range tests {
		source, err := os.ReadFile(name)
		if err != nil {
			t.Fatal(err)
		}
		if name != "agent_route_prefix_test.go" && singular(string(source)) {
			t.Errorf("%s names a singular /agent/ route", name)
		}
	}
}
