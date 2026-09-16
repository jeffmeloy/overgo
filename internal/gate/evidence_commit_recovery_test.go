package gate

import (
	"bufio"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"overgo/internal/authoritylock"
	"overgo/internal/processlock"
	"overgo/internal/runrecord"
)

func TestEvidenceCommitRecoveryProcess(t *testing.T) {
	if os.Getenv("OVERGO_TEST_EVIDENCE_COMMIT_RECOVERY") == "" {
		return
	}
	if err := Run(Options{StorePath: StorePath, PlanRef: "ratchet/recover", PathsCSV: "candidate.go", MessageFile: "must-not-open-a-new-attempt"}); err != nil {
		t.Fatal(err)
	}
}

func TestEvidenceCommitRecoveryAcceptance(t *testing.T) {
	t.Parallel()
	t.Run("normal gate finishes a committed evidence batch", func(t *testing.T) {
		fixture := newInterruptedCommitFixture(t)
		// The public gate uses the canonical store name; the recovery fixture's
		// paths are otherwise entirely inside this isolated repository.
		if err := os.Rename(filepath.Join(fixture.repo, fixture.storePath), filepath.Join(fixture.repo, StorePath)); err != nil {
			t.Fatal(err)
		}
		fixture.storePath = StorePath
		attempt, batch := successfulInterruptedAttemptBatch(t, fixture, true, "gate/final/"+fixture.preparation.ID.String())
		g := gateContext{repo: fixture.repo, preparation: fixture.preparation}
		if err := g.oweRecord(batch, errors.New("injected final publication outage")); err == nil {
			t.Fatal("outage did not retain recording debt")
		}
		producer, inputs := terminalEvidenceFixture(t, fixture.repo)
		ledger, err := producer.openPackageEvidence()
		if err != nil {
			t.Fatal(err)
		}
		packages := []string{"fixture/good", "fixture/pending"}
		if err := ledger.prepare(t.Context(), packages, "complete", inputs, producer.retryCache); err != nil {
			t.Fatal(err)
		}
		acquisitions := map[string]int{}
		acquire := func(pkg string) {
			acquisitions[pkg]++
			if err := ledger.record(t.Context(), pkg, true); err != nil {
				t.Fatal(err)
			}
		}
		acquire("fixture/good")
		if err := producer.closeStore(); err != nil {
			t.Fatal(err)
		}
		// A missing message file makes any accidental new gate execution fail.
		// Recovery must finish the existing commit and return before acquisition.
		writer, err := processlock.Acquire(filepath.Join(fixture.repo, StorePath, "overgodb.lock"), 0o600)
		if err != nil {
			t.Fatal(err)
		}
		defer writer.Close()
		child := exec.CommandContext(t.Context(), os.Args[0], "-test.run=^TestEvidenceCommitRecoveryProcess$", "-test.v")
		child.Dir = fixture.repo
		child.Env = append(os.Environ(), "OVERGO_TEST_EVIDENCE_COMMIT_RECOVERY=1")
		output, err := child.StdoutPipe()
		if err != nil {
			t.Fatal(err)
		}
		child.Stderr = child.Stdout
		if err := child.Start(); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = child.Process.Kill() })
		ready := make(chan struct{})
		done := make(chan error, 1)
		var transcript strings.Builder
		go func() {
			scanner := bufio.NewScanner(output)
			for scanner.Scan() {
				line := scanner.Text()
				transcript.WriteString(line + "\n")
				if line == "gate: admission open canonical store started" {
					close(ready)
				}
			}
			done <- errors.Join(scanner.Err(), child.Wait())
		}()
		select {
		case <-ready:
		case err := <-done:
			t.Fatalf("gate did not reach store admission: %v\n%s", err, transcript.String())
		}
		// A gate that failed instead of waiting answers the receive below
		// with its error once the writer releases.
		if err := writer.Close(); err != nil {
			t.Fatal(err)
		}
		if err := <-done; err != nil {
			t.Fatalf("recovery child: %v\n%s", err, transcript.String())
		}
		if !strings.Contains(transcript.String(), "state=not_busy scope=gate") {
			t.Fatal("recovery did not report its released authority")
		}
		released, err := authoritylock.Acquire(fixture.repo)
		if err != nil {
			t.Fatalf("completed gate retained authority: %v", err)
		}
		if err := released.Close(); err != nil {
			t.Fatal(err)
		}
		ledger, err = producer.openPackageEvidence()
		if err != nil {
			t.Fatal(err)
		}
		if err := ledger.prepare(t.Context(), packages, "complete", inputs, producer.retryCache); err != nil {
			t.Fatal(err)
		}
		pending, reused, err := producer.packageCachePartition(packages, "complete", inputs)
		if err != nil || reused != 1 || len(pending) != 1 || pending[0] != "fixture/pending" {
			t.Fatalf("recovery lost exact receipts: %v, %d, %v", pending, reused, err)
		}
		for _, pkg := range pending {
			acquire(pkg)
		}
		if acquisitions["fixture/good"] != 1 || acquisitions["fixture/pending"] != 1 {
			t.Fatalf("recovery repeated acquisition: %v", acquisitions)
		}
		if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
			t.Fatalf("recovery made another commit: %s", head)
		}
		if count := recoveryGit(t, fixture.repo, "rev-list", "--count", fixture.parent+"..HEAD"); count != "1" {
			t.Fatalf("commit count=%s", count)
		}
		assertNoCommitIntent(t, fixture.repo)
		if _, err := os.Stat(filepath.Join(fixture.repo, gateDebtFile)); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("recording debt remains: %v", err)
		}
		store := openRecoveryStore(t, fixture)
		defer store.Close()
		if _, err := runrecord.VerifyAttemptGate(t.Context(), store, attempt); err != nil {
			t.Fatal(err)
		}
		t.Logf("real gate recovery=1 commit=1; synthetic acquisitions=%v; independent pending receipt retained; excludes full producer-to-commit workflow and model evaluation", acquisitions)
	})
	for _, fault := range []string{"corrupt debt", "corrupt intent", "corrupt locator", "different branch", "changed candidate"} {
		t.Run("refuse "+fault, func(t *testing.T) {
			fixture := newInterruptedCommitFixture(t)
			_, batch := successfulInterruptedAttemptBatch(t, fixture, true, "gate/final/"+fixture.preparation.ID.String())
			g := gateContext{repo: fixture.repo, preparation: fixture.preparation}
			if err := g.oweRecord(batch, errors.New("injected final publication outage")); err == nil {
				t.Fatal("outage lost")
			}
			switch fault {
			case "corrupt debt", "corrupt intent", "corrupt locator":
				name := map[string]string{"corrupt debt": gateDebtFile, "corrupt intent": gateCommitIntentFile, "corrupt locator": gateHeartbeatFile}[fault]
				if err := os.WriteFile(filepath.Join(fixture.repo, name), []byte("corrupt authority"), 0o600); err != nil {
					t.Fatal(err)
				}
			case "different branch":
				runGitFixture(t, fixture.repo, "checkout", "-q", "-b", "foreign-branch")
			case "changed candidate":
				if err := os.WriteFile(filepath.Join(fixture.repo, fixture.sourcePath), []byte("package candidate\nconst value = 3\n"), 0o644); err != nil {
					t.Fatal(err)
				}
			}
			lock, err := authoritylock.Acquire(fixture.repo)
			if err != nil {
				t.Fatal(err)
			}
			defer lock.Close()
			store := openRecoveryStore(t, fixture)
			defer store.Close()
			before, sequence := store.Head()
			var snapshots = map[string]string{}
			for _, name := range []string{gateDebtFile, gateCommitIntentFile, fixture.sourcePath} {
				data, err := os.ReadFile(filepath.Join(fixture.repo, name))
				if err != nil {
					t.Fatal(err)
				}
				snapshots[name] = string(data)
			}
			if ref, err := admitPendingGateState(fixture.repo, fixture.storePath, store); err == nil || ref != "" {
				t.Fatalf("unsafe recovery admitted: %q, %v", ref, err)
			}
			if err := store.Refresh(t.Context()); err != nil {
				t.Fatal(err)
			}
			if head, count := store.Head(); head != before || count != sequence {
				t.Fatal("refusal changed store authority")
			}
			if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); head != fixture.commit {
				t.Fatal("refusal changed Git revision")
			}
			for name, before := range snapshots {
				data, err := os.ReadFile(filepath.Join(fixture.repo, name))
				if err != nil || string(data) != before {
					t.Fatalf("refusal changed %s: %v", name, err)
				}
			}
		})
	}
	t.Run("publication succeeded before supervisor restart", func(t *testing.T) {
		fixture := newInterruptedCommitFixture(t)
		_, batch := successfulInterruptedAttemptBatch(t, fixture, true, "gate/final/"+fixture.preparation.ID.String())
		store := openRecoveryStore(t, fixture)
		defer store.Close()
		if _, err := store.Commit(t.Context(), batch); err != nil {
			t.Fatal(err)
		}
		before, sequence := store.Head()
		g := gateContext{repo: fixture.repo, preparation: fixture.preparation}
		if err := g.oweRecord(batch, errors.New("supervisor lost publication response")); err == nil {
			t.Fatal("outage lost")
		}
		lock, err := authoritylock.Acquire(fixture.repo)
		if err != nil {
			t.Fatal(err)
		}
		defer lock.Close()
		ref, err := admitPendingGateState(fixture.repo, fixture.storePath, store)
		if err != nil || ref != "ratchet/recover" {
			t.Fatalf("resume = %q, %v", ref, err)
		}
		if head, count := store.Head(); head != before || count != sequence {
			t.Fatal("restart duplicated final publication")
		}
		if head := recoveryGit(t, fixture.repo, "rev-parse", "HEAD"); strings.TrimSpace(head) != fixture.commit {
			t.Fatal("restart duplicated commit")
		}
	})
}
