package agenttool

import (
	"encoding/json"
	"fmt"
	"net"
	"net/netip"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"overgo/internal/testcheck"
)

// TestOneRefusedAddressList: a tool's endpoint and a media source answer to one list, every prefix of which the dial refuses,
// the multicast and broadcast addresses the old list admitted among them, and public addresses stay reachable.
func TestOneRefusedAddressList(t *testing.T) {
	t.Parallel()
	dial := guardedDialContext(false)
	for _, text := range []string{
		"0.0.0.1", "10.0.0.1", "100.64.0.1", "127.0.0.1", "169.254.1.1", "172.16.0.1", "192.0.0.1", "192.0.2.1",
		"192.168.1.1", "198.18.0.1", "198.51.100.1", "203.0.113.1", "224.0.0.1", "224.0.1.1", "239.1.1.1", "240.0.0.1",
		"255.255.255.255", "::", "::1", "64:ff9b::1", "100::1", "2001::1", "2001:db8::1", "fc00::1", "fe80::1", "ff00::1",
		"::ffff:10.0.0.1",
	} {
		if PublicAddress(netip.MustParseAddr(text)) {
			t.Errorf("%s reads as public", text)
		}
		if _, err := dial(t.Context(), "tcp", net.JoinHostPort(text, "80")); err == nil || !strings.Contains(err.Error(), "private networks are refused") {
			t.Errorf("dial %s: %v", text, err)
		}
	}
	for _, text := range []string{"8.8.8.8", "1.1.1.1", "2606:4700:4700::1111"} {
		if !PublicAddress(netip.MustParseAddr(text)) {
			t.Errorf("%s reads as refused", text)
		}
	}
}

// TestToolEnvironmentProcess is the argv tool's child process, which prints what it received; a normal run does nothing.
func TestToolEnvironmentProcess(t *testing.T) {
	// Serial: it is the child process's entry, which returns at once in a normal run.
	if !slices.Contains(os.Args, "--") {
		return
	}
	fmt.Println(strings.Join(os.Environ(), "\n"))
	os.Exit(0)
}

// TestAToolReceivesOnlyWhatItsManualNames: a variable the manual names reaches an argv tool over the base, and a key in the
// server's environment does not.
func TestAToolReceivesOnlyWhatItsManualNames(t *testing.T) {
	// Serial: it sets the server's environment and finds the tool on PATH.
	executable := testcheck.Must(os.Executable())(t)
	t.Setenv("PATH", filepath.Dir(executable)+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("TOOL_NAMED", "named")
	t.Setenv("TOOL_PROVIDER_KEY", "held-by-the-server")
	manual := testcheck.Must(NewManual(Manual{
		Name: "tool.environment", Description: "Prints the environment it received.", Effect: EffectInspection,
		Transport: Transport{
			Kind: TransportArgv, Program: filepath.Base(executable),
			Args:        []string{"-test.run=^TestToolEnvironmentProcess$", "--"},
			Environment: []string{"TOOL_NAMED"},
		},
	}))(t)
	result := testcheck.Must(NewOperatorExecutor().Invoke(t.Context(), manual, json.RawMessage(`{}`)))(t)
	var received string
	testcheck.NoError(t, json.Unmarshal(result, &received))
	if !strings.Contains(received, "TOOL_NAMED=named") || strings.Contains(received, "held-by-the-server") || !strings.Contains(strings.ToUpper(received), "PATH=") {
		t.Fatalf("the tool received:\n%s", received)
	}
	if _, err := NewManual(Manual{
		Name: "tool.badname", Description: "Names a variable no system has.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportArgv, Program: "tool", Environment: []string{"A=B"}},
	}); err == nil {
		t.Fatal("a variable name carrying a value was admitted")
	}
	if _, err := NewManual(Manual{
		Name: "tool.nowhere", Description: "Names a variable for a transport with no program.", Effect: EffectInspection,
		Transport: Transport{Kind: TransportHTTP, URL: "http://example.invalid/tool", Environment: []string{"TOOL_NAMED"}},
	}); err == nil {
		t.Fatal("an http tool named an environment")
	}
}
