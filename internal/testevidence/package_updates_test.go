package testevidence

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"maps"
	"net"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	"overgo/internal/processcontrol"
	"overgo/internal/testskip"
)

func packageEvent(action, pkg, test, output string) string {
	data, _ := json.Marshal(map[string]string{"Action": action, "Package": pkg, "Test": test, "Output": output})
	return string(data) + "\n"
}

func TestSlowPublicationProcess(t *testing.T) {
	announce := os.Getenv("OVERGO_TEST_SLOW_PUBLICATION")
	if announce == "" {
		return
	}
	for _, name := range []string{"first", "second"} {
		_, _ = io.WriteString(os.Stdout, packageEvent("start", name, "", "")+packageEvent("run", name, "TestWorks", ""))
		if name == "second" {
			// Exceed pipe capacity without retaining the raw stream in memory.
			for range 128 {
				_, _ = io.WriteString(os.Stdout, packageEvent("output", name, "TestWorks", strings.Repeat("x", 32768)))
			}
		}
		_, _ = io.WriteString(os.Stdout, packageEvent("pass", name, "TestWorks", "")+packageEvent("pass", name, "", ""))
	}
	// Both streams feed the parser, so the fixture announces its drained
	// output by connecting to the parent's listener.
	connection, err := net.Dial("tcp", announce)
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close()
	os.Exit(0)
}

func TestTerminalPublicationDoesNotBlockOutput(t *testing.T) {
	ctx := t.Context()
	drained, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer drained.Close()
	release, entered, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
	unblock := sync.OnceFunc(func() { close(release) })
	var report GoTestReport
	var runErr error
	var updates []string
	go func() {
		report, runErr = RunGoTestCommand(ctx, processcontrol.Command{
			Path: os.Args[0], Args: []string{"-test.run=^TestSlowPublicationProcess$"},
			Env: append(os.Environ(), "OVERGO_TEST_SLOW_PUBLICATION="+drained.Addr().String()),
		}, GoTestOptions{DiagnosticBytes: 1024, Observe: func(name string, passed bool, tests map[string]string) error {
			if name == "first" {
				close(entered)
				<-release
			}
			updates = append(updates, name)
			return nil
		}})
		close(done)
	}()
	defer func() { unblock(); <-done }()
	select {
	case <-entered:
	case <-ctx.Done():
		t.Fatal(context.Cause(ctx))
	}
	// The fixture drains its whole output while the first publication is
	// still held; its connection proves the output was never blocked.
	connection, err := drained.Accept()
	if err != nil {
		t.Fatal(err)
	}
	_ = connection.Close()
	select {
	case <-done:
		t.Fatal("command returned before terminal publication finished")
	default:
	}
	unblock()
	<-done
	if runErr != nil || report.PassedPackages != 2 || !slices.Equal(updates, []string{"first", "second"}) {
		t.Fatalf("terminal publications = %v, packages=%d, error=%v", updates, report.PassedPackages, runErr)
	}
}

func TestQueuedPublicationPreservesRevocationsAndErrors(t *testing.T) {
	failure := errors.New("publication unavailable")
	var received []bool
	var receivedTests []map[string]string
	observe, flush := queuedPackageUpdates(func(_ string, passed bool, tests map[string]string) error {
		received = append(received, passed)
		receivedTests = append(receivedTests, tests)
		if passed {
			return failure
		}
		return nil
	})
	tests := map[string]string{"TestWorks": "pass"}
	_ = observe("package", true, tests)
	tests["TestWorks"] = "fail"
	_ = observe("package", false, nil)
	if err := flush(); !errors.Is(err, failure) || !slices.Equal(received, []bool{true, false}) {
		t.Fatalf("publication failure or revocation lost: %v, %v", received, err)
	}
	if receivedTests[0]["TestWorks"] != "pass" || len(receivedTests[1]) != 0 {
		t.Fatal("queued named verdict mutated or revocation retained passing names")
	}
}

func TestPackageNamedVerdicts(t *testing.T) {
	start := packageEvent("start", "example", "", "")
	passed := packageEvent("run", "example", "TestWorks", "") + packageEvent("pass", "example", "TestWorks", "")
	terminal := packageEvent("pass", "example", "", "")
	skipped := packageEvent("run", "example", "TestIntegration", "") + packageEvent("output", "example", "TestIntegration", testskip.ShortIntegration) + packageEvent("skip", "example", "TestIntegration", "")
	for _, tc := range []struct {
		name, stream string
		short        bool
		want         map[string]string
	}{
		{"passed", start + passed + terminal, false, map[string]string{"TestWorks": "pass"}},
		{"short exclusion", start + passed + skipped + terminal, true, map[string]string{"TestWorks": "pass", "TestIntegration": "skip"}},
		{"complete exclusion", start + passed + skipped + terminal, false, nil},
		{"missing assertion", start + terminal, false, nil},
		{"missing start", start + packageEvent("pass", "example", "TestWorks", "") + terminal, false, nil},
		{"duplicate", start + passed + passed + terminal, false, nil},
		{"failed", start + strings.ReplaceAll(passed, `"pass"`, `"fail"`) + terminal, false, nil},
		{"unfinished", start + packageEvent("run", "example", "TestWorks", "") + terminal, false, nil},
		{"late unavailable", start + passed + terminal + packageEvent("output", "example", "TestWorks", "UNAVAILABLE: fixture"), false, nil},
		{"malformed tail", start + passed + terminal + "{invalid\n", false, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var got map[string]string
			_, _ = GoTestJSONReader(strings.NewReader(tc.stream), tc.short, len(tc.stream), func(pkg string, credit bool, tests map[string]string) error {
				if pkg != "example" || !credit && len(tests) != 0 {
					t.Fatal("invalid named transition")
				}
				got = maps.Clone(tests)
				return nil
			})
			if !maps.Equal(got, tc.want) {
				t.Fatalf("named verdicts=%v want=%v", got, tc.want)
			}
		})
	}
}

func TestTerminalPackageUpdates(t *testing.T) {
	good := packageEvent("start", "good", "", "") + packageEvent("run", "good", "TestWorks", "") +
		packageEvent("pass", "good", "TestWorks", "") + packageEvent("pass", "good", "", "")
	for _, tc := range []struct {
		name, tail string
		revoke     bool
	}{
		{"failed sibling", packageEvent("start", "bad", "", "") + packageEvent("fail", "bad", "", ""), false},
		{"unfinished sibling", packageEvent("start", "bad", "", ""), false},
		{"skipped sibling", packageEvent("start", "bad", "", "") + packageEvent("skip", "bad", "", ""), false},
		{"malformed stream", "not json\n", true},
		{"duplicate terminal", packageEvent("pass", "good", "", ""), true},
		{"late test", packageEvent("run", "good", "TestLate", ""), true},
		{"late unavailable", packageEvent("output", "good", "", "UNAVAILABLE: required fixture absent"), true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			reader, writer := io.Pipe()
			t.Cleanup(func() { _ = reader.Close(); _ = writer.Close() })
			updates := make(chan bool, 2)
			finished := make(chan GoTestReport, 1)
			go func() {
				report, _ := GoTestJSONReader(reader, false, 1024, func(pkg string, passed bool, tests map[string]string) error {
					if pkg != "good" {
						return errors.New("unexpected package credited")
					}
					updates <- passed
					return nil
				})
				finished <- report
			}()
			if _, err := io.WriteString(writer, good); err != nil {
				t.Fatal(err)
			}
			// The receipt must arrive while the stream remains open.
			if passed := <-updates; !passed {
				t.Fatal("terminal package was not credited before sibling execution")
			}
			if _, err := io.WriteString(writer, tc.tail); err != nil {
				t.Fatal(err)
			}
			_ = writer.Close()
			report := <-finished
			if report.PackagePassed("good") == tc.revoke {
				t.Fatalf("final package verdict disagrees with stream: %+v", report)
			}
			close(updates)
			var later []bool
			for value := range updates {
				later = append(later, value)
			}
			want := []bool(nil)
			if tc.revoke {
				want = []bool{false}
			}
			if !slices.Equal(later, want) {
				t.Fatalf("later updates=%v want=%v", later, want)
			}
		})
	}
	t.Run("no vacuous or skipped credit", func(t *testing.T) {
		for _, transcript := range []string{
			packageEvent("start", "empty", "", "") + packageEvent("pass", "empty", "", ""),
			strings.Replace(good, `"Action":"run"`, `"Action":"output"`, 1),
			strings.Replace(good, `"Action":"pass"`, `"Action":"skip"`, 1),
		} {
			calls := 0
			_, err := GoTestJSONReader(strings.NewReader(transcript), false, 1024, func(string, bool, map[string]string) error { calls++; return nil })
			if err != nil || calls != 0 {
				t.Fatalf("vacuous stream credited: calls=%d error=%v", calls, err)
			}
		}
	})
}
