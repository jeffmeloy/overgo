package gate

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/jsonfile"
)

// claimAudit: collected audit lines of one claimant.
type claimAudit struct{ lines []string }

func (a *claimAudit) record(line string) { a.lines = append(a.lines, line) }

func (a *claimAudit) text() string { return strings.Join(a.lines, "\n") }

// TestHostWideDeviceClaim pins: every worktree of a repository resolves the
// same claim file; a second claimant waits with the holder recorded and
// takes the device once released; a stale heartbeat, which a dead holder
// leaves behind, frees the device with the reason audited; a release by a
// non-holder is refused;
// the exclusive-device check runs under the claim and leaves none behind.
func TestHostWideDeviceClaim(t *testing.T) {
	repo := t.TempDir()
	runGitFixture(t, repo, "init", "-q")
	runGitFixture(t, repo, "config", "user.email", "claim@example.invalid")
	runGitFixture(t, repo, "config", "user.name", "Claim Test")
	if err := os.WriteFile(filepath.Join(repo, "README"), []byte("claim\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	runGitFixture(t, repo, "add", "README")
	runGitFixture(t, repo, "commit", "-q", "-m", "claim fixture")
	other := filepath.Join(t.TempDir(), "lane")
	runGitFixture(t, repo, "worktree", "add", "-q", other)
	path, err := deviceClaimPath(repo)
	if err != nil {
		t.Fatal(err)
	}
	if otherPath, err := deviceClaimPath(other); err != nil || otherPath != path {
		t.Fatalf("worktree claim path = %s, %v; want %s", otherPath, err, path)
	}

	self := os.Getpid()
	first := deviceClaim{PID: self, Worktree: repo, PlanRef: "a/one", Check: "device"}
	second := deviceClaim{PID: self, Worktree: other, PlanRef: "b/two", Check: "device"}
	releaseFirst, err := acquireDeviceClaim(t.Context(), path, first, time.Second, 10*time.Millisecond, (&claimAudit{}).record)
	if err != nil {
		t.Fatal(err)
	}
	waiting := &claimAudit{}
	if _, err := acquireDeviceClaim(t.Context(), path, second, 200*time.Millisecond, 20*time.Millisecond, waiting.record); err == nil || !strings.Contains(err.Error(), "held by pid") {
		t.Fatalf("second claimant took a held device: %v", err)
	}
	if !strings.Contains(waiting.text(), "held by pid") || !strings.Contains(waiting.text(), "plan a/one") || !strings.Contains(waiting.text(), repo) {
		t.Fatalf("waiting claimant did not record the holder:\n%s", waiting.text())
	}
	if err := releaseFirst(); err != nil {
		t.Fatal(err)
	}
	released := &claimAudit{}
	releaseSecond, err := acquireDeviceClaim(t.Context(), path, second, time.Second, 10*time.Millisecond, released.record)
	if err != nil {
		t.Fatalf("released device was not acquired: %v", err)
	}
	if err := releaseSecond(); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("claim survived release: %v", err)
	}

	for name, holder := range map[string]deviceClaim{
		"stale heartbeat": {PID: self, Worktree: other, PlanRef: "b/two", Started: time.Now(), Heartbeat: time.Now().Add(-2 * time.Minute)},
	} {
		if err := jsonfile.Write(path, holder, 0o600); err != nil {
			t.Fatal(err)
		}
		audit := &claimAudit{}
		release, err := acquireDeviceClaim(t.Context(), path, first, time.Second, 10*time.Millisecond, audit.record)
		if err != nil || !strings.Contains(audit.text(), "released expired holder") {
			t.Fatalf("%s: acquire = %v audit=%s", name, err, audit.text())
		}
		if err := release(); err != nil {
			t.Fatal(err)
		}
	}

	// A one-hour heartbeat interval keeps the holder from rewriting the moved claim.
	releaseMoved, err := acquireDeviceClaim(t.Context(), path, first, time.Second, time.Hour, (&claimAudit{}).record)
	if err != nil {
		t.Fatal(err)
	}
	if err := jsonfile.Write(path, second, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := releaseMoved(); err == nil || !strings.Contains(err.Error(), "release refused") {
		t.Fatalf("release of a moved claim = %v", err)
	}
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}

	g := &gateContext{repo: repo, planRef: "a/one"}
	ran := false
	wrapped := g.withDeviceClaim(automationcheck.Check{
		Descriptor: automationcheck.Descriptor{Name: "device"},
		Run: func(context.Context, automationcheck.Invocation) (bool, string, error) {
			ran = true
			var holder deviceClaim
			if err := jsonfile.DecodeStrict(path, &holder); err != nil || holder.PID != self || holder.Check != "device" {
				t.Fatalf("device check ran without the claim: %+v %v", holder, err)
			}
			return false, "", nil
		},
	})
	if _, _, err := wrapped.Run(t.Context(), automationcheck.Invocation{}); err != nil || !ran {
		t.Fatalf("wrapped device check = %v ran=%t", err, ran)
	}
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatalf("claim survived the device check: %v", err)
	}
}
