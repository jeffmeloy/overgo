package modelswap

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestLauncherForwardsItsReplyBound holds the launcher to the reply bound it is given (loop-proof/reply-bound): a
// child launched with MaxTokens accepts that many tokens a reply (-max-tokens), and one launched without keeps the
// recipe policy's, as every child did before. The local worker let think asked for 16384 tokens a reply and every
// request was refused at the policy's 4096.
func TestLauncherForwardsItsReplyBound(t *testing.T) {
	const bound = 16384
	for _, maxTokens := range []int{0, bound} {
		record := filepath.Join(t.TempDir(), "arguments")
		t.Setenv("OVERGO_TEST_CHILD_STARTUP", "arguments")
		t.Setenv("OVERGO_TEST_ARGUMENT_RECORD", record)
		supervisor := testcheck.Must(New(ServerLauncher{Binary: os.Args[0], Store: t.TempDir(), MaxTokens: maxTokens}, 0))(t)
		proxy := &Proxy{Supervisor: supervisor, Resolver: mapResolver{"fixture": {Name: "fixture", Location: "fixture.gguf"}}}
		response := httptest.NewRecorder()
		proxy.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "http://localhost/health?swap=fixture", nil))
		testcheck.NoError(t, supervisor.Close())
		if response.Code != http.StatusServiceUnavailable {
			t.Fatalf("startup = %d %s", response.Code, response.Body)
		}
		arguments := strings.Split(strings.TrimSpace(string(testcheck.Must(os.ReadFile(record))(t))), "\n")
		at := slices.Index(arguments, "-max-tokens")
		switch {
		case maxTokens == 0 && at >= 0:
			t.Errorf("a launcher with no reply bound passed the child %q; want the policy's bound kept", arguments)
		case maxTokens != 0 && (at < 0 || at+1 >= len(arguments) || arguments[at+1] != "16384"):
			t.Errorf("a launcher bound to %d tokens a reply passed the child %q; want -max-tokens 16384", maxTokens, arguments)
		}
		if arguments[len(arguments)-1] != "fixture.gguf" {
			t.Errorf("child location is not last: %q", arguments)
		}
	}
}
