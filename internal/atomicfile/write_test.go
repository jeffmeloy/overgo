package atomicfile

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

const crashBoundaryExitCode = 86

func TestFailedReplacementPreservesExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document")
	original := []byte("before")
	if err := os.WriteFile(path, original, 0o600); err != nil {
		t.Fatal(err)
	}
	failed := errors.New("replace interrupted")
	err := write(path, []byte("after"), 0o600, func(string, string) error { return failed })
	if !errors.Is(err, failed) {
		t.Fatalf("write error = %v, want %v", err, failed)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(original) {
		t.Fatalf("failed replacement changed target: got %q want %q", got, original)
	}
}

func TestWriteReplacesExistingFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := Write(path, []byte("after"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "after" {
		t.Fatalf("replacement = %q, want after", got)
	}
}

func TestWriteRequiresExistingParent(t *testing.T) {
	directory := filepath.Join(t.TempDir(), "caller-owned")
	path := filepath.Join(directory, "document")
	if err := Write(path, []byte("after"), 0o600); err == nil {
		t.Fatal("write created a caller-owned parent directory")
	}
	if _, err := os.Stat(directory); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("parent directory state = %v, want absent", err)
	}
}

func TestCompareAndSwapPreservesReplacementBeforeDetach(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := compareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o600, swapHooks{
		beforeDetach: func(path string) {
			if writeErr := Write(path, []byte("concurrent"), 0o600); writeErr != nil {
				t.Fatalf("concurrent replacement: %v", writeErr)
			}
		},
	})
	if !errors.Is(err, ErrChanged) {
		t.Fatalf("compare-and-swap error = %v, want changed", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "concurrent" {
		t.Fatalf("refused swap retained %q, want concurrent", got)
	}
	matches, err := filepath.Glob(filepath.Join(scratch, ".atomic-swap-old-*"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("preserved quarantine = (%v, %v), want one", matches, err)
	}
	preserved, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(preserved) != "concurrent" {
		t.Fatalf("quarantine = %q, want concurrent", preserved)
	}
}

func TestCompareAndSwapPreservesReplacementAfterDetach(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	err := compareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o600, swapHooks{
		beforeInstall: func(path string) {
			if writeErr := os.WriteFile(path, []byte("concurrent"), 0o600); writeErr != nil {
				t.Fatalf("concurrent creation: %v", writeErr)
			}
		},
	})
	if !errors.Is(err, ErrChanged) {
		t.Fatalf("compare-and-swap error = %v, want changed", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "concurrent" {
		t.Fatalf("refused swap retained %q, want concurrent", got)
	}
	matches, err := filepath.Glob(filepath.Join(scratch, ".atomic-swap-old-*"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("preserved quarantine = (%v, %v), want one", matches, err)
	}
	preserved, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatal(err)
	}
	if string(preserved) != "before" {
		t.Fatalf("quarantine = %q, want before", preserved)
	}
}

func TestCompareAndSwapReplacesExactState(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CompareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o600); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "after" {
		t.Fatalf("replacement = %q, want after", got)
	}
}

func TestCompareAndSwapRefusesNonemptyScratch(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(scratch, "foreign"), []byte("tamper"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := CompareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o600); err == nil {
		t.Fatal("compare-and-swap accepted nonempty scratch")
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "before" {
		t.Fatalf("refused swap changed target to %q", got)
	}
}

func TestCompareAndSwapKeepsScratchOutsideTargetDirectory(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	observedScratch := false
	if err := compareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o600, swapHooks{
		afterDetach: func(string) {
			matches, err := filepath.Glob(filepath.Join(scratch, ".atomic-swap-*"))
			if err != nil {
				t.Fatal(err)
			}
			observedScratch = len(matches) != 0
		},
	}); err != nil {
		t.Fatal(err)
	}
	if !observedScratch {
		t.Fatal("swap did not keep its transient entries in caller-owned scratch")
	}
	matches, err := filepath.Glob(filepath.Join(directory, ".atomic-swap-*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(matches) != 0 {
		t.Fatalf("target directory contains swap scratch: %v", matches)
	}
}

func TestCompareAndSwapFilesystemBoundariesRemainRecoverable(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	var boundaries []string
	requireState := func(boundary, want string, present bool) {
		t.Helper()
		boundaries = append(boundaries, boundary)
		got, err := os.ReadFile(path)
		if !present {
			if !errors.Is(err, os.ErrNotExist) {
				t.Fatalf("%s target error = %v, want absent", boundary, err)
			}
			return
		}
		if err != nil {
			t.Fatalf("%s target: %v", boundary, err)
		}
		if string(got) != want {
			t.Fatalf("%s target = %q, want %q", boundary, got, want)
		}
	}
	if err := compareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o600, swapHooks{
		beforeDetach:  func(string) { requireState("before-detach", "before", true) },
		afterDetach:   func(string) { requireState("after-detach", "", false) },
		beforeInstall: func(string) { requireState("before-install", "", false) },
		afterInstall:  func(string) { requireState("after-install", "after", true) },
		afterRemove:   func(string) { requireState("after-remove", "after", true) },
	}); err != nil {
		t.Fatal(err)
	}
	want := []string{"before-detach", "after-detach", "before-install", "after-install", "after-remove"}
	if len(boundaries) != len(want) {
		t.Fatalf("boundaries = %v, want %v", boundaries, want)
	}
	for index := range want {
		if boundaries[index] != want[index] {
			t.Fatalf("boundaries = %v, want %v", boundaries, want)
		}
	}
}

func TestCompareAndSwapProcessDeathAtFilesystemBoundaries(t *testing.T) {
	tests := []struct {
		boundary string
		present  bool
		content  string
		resolved string
	}{
		{boundary: "during-next-preparation", present: true, content: "before", resolved: "before"},
		{boundary: "during-witness-reservation", present: true, content: "before", resolved: "before"},
		{boundary: "during-old-reservation", present: true, content: "before", resolved: "before"},
		{boundary: "before-detach", present: true, content: "before", resolved: "before"},
		{boundary: "after-detach", resolved: "before"},
		{boundary: "before-install", resolved: "before"},
		{boundary: "after-install", present: true, content: "after", resolved: "after"},
		{boundary: "after-remove", present: true, content: "after", resolved: "after"},
	}
	for _, test := range tests {
		t.Run(test.boundary, func(t *testing.T) {
			root := t.TempDir()
			directory := filepath.Join(root, "documents")
			scratch := filepath.Join(root, "scratch")
			if err := os.Mkdir(directory, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.Mkdir(scratch, 0o700); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(directory, "document")
			if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
				t.Fatal(err)
			}
			process := exec.Command(os.Args[0], "-test.run=^TestCompareAndSwapCrashBoundaryProcess$")
			process.Env = append(
				os.Environ(),
				"OVERGO_ATOMICFILE_CRASH_BOUNDARY="+test.boundary,
				"OVERGO_ATOMICFILE_CRASH_PATH="+path,
				"OVERGO_ATOMICFILE_CRASH_SCRATCH="+scratch,
			)
			err := process.Run()
			exitError, exitFailure := errors.AsType[*exec.ExitError](err)
			if !exitFailure || exitError.ExitCode() != crashBoundaryExitCode {
				t.Fatalf("crash helper error = %v, want exit %d", err, crashBoundaryExitCode)
			}
			got, err := os.ReadFile(path)
			if !test.present {
				if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("target error = %v, want absent", err)
				}
			} else if err != nil {
				t.Fatal(err)
			} else if string(got) != test.content {
				t.Fatalf("target = %q, want %q", got, test.content)
			}
			resolved, err := RecoverSwap(path, scratch, []byte("before"), []byte("after"), 0o600)
			if err != nil {
				t.Fatalf("recover %s: %v", test.boundary, err)
			}
			if string(resolved) != test.resolved {
				t.Fatalf("resolved state = %q, want %q", resolved, test.resolved)
			}
			entries, err := os.ReadDir(scratch)
			if err != nil {
				t.Fatal(err)
			}
			if len(entries) != 0 {
				t.Fatalf("resolved scratch retains %d entries", len(entries))
			}
			repeated, err := RecoverSwap(path, scratch, []byte("before"), []byte("after"), 0o600)
			if err != nil || string(repeated) != test.resolved {
				t.Fatalf("repeated recovery = (%q, %v), want %q", repeated, err, test.resolved)
			}
		})
	}
}

func TestCompareAndSwapCrashBoundaryProcess(t *testing.T) {
	boundary := os.Getenv("OVERGO_ATOMICFILE_CRASH_BOUNDARY")
	if boundary == "" {
		return
	}
	path := os.Getenv("OVERGO_ATOMICFILE_CRASH_PATH")
	scratch := os.Getenv("OVERGO_ATOMICFILE_CRASH_SCRATCH")
	crash := func(string) { os.Exit(crashBoundaryExitCode) }
	hooks := swapHooks{}
	switch boundary {
	case "during-next-preparation":
		hooks.afterPrepareCreate = crash
	case "during-witness-reservation":
		hooks.afterWitnessReservation = crash
	case "during-old-reservation":
		hooks.afterOldReservation = crash
	case "before-detach":
		hooks.beforeDetach = crash
	case "after-detach":
		hooks.afterDetach = crash
	case "before-install":
		hooks.beforeInstall = crash
	case "after-install":
		hooks.afterInstall = crash
	case "after-remove":
		hooks.afterRemove = crash
	default:
		t.Fatalf("unknown crash boundary %q", boundary)
	}
	if err := compareAndSwap(path, scratch, []byte("before"), []byte("after"), 0o600, hooks); err != nil {
		t.Fatal(err)
	}
	t.Fatal("compare-and-swap returned without reaching its crash boundary")
}

func TestRecoverSwapRefusesNonRegularPrephaseResidue(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	residue := filepath.Join(scratch, swapPreparingNextPrefix+"not-a-file")
	if err := os.Mkdir(residue, 0o700); err != nil {
		t.Fatal(err)
	}
	if _, err := RecoverSwap(path, scratch, []byte("before"), []byte("after"), 0o600); err == nil {
		t.Fatal("recovery accepted non-regular pre-phase residue")
	}
	if _, err := os.Stat(residue); err != nil {
		t.Fatalf("refused residue was changed: %v", err)
	}
}

func TestRecoverSwapPreservesConcurrentTarget(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}
	next := filepath.Join(scratch, ".atomic-swap-next-test")
	witness := filepath.Join(scratch, ".atomic-swap-witness-test")
	old := filepath.Join(scratch, ".atomic-swap-old-test")
	if err := os.WriteFile(next, []byte("after"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Link(path, witness); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(path, old); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("concurrent"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := RecoverSwap(path, scratch, []byte("before"), []byte("after"), 0o600); !errors.Is(err, ErrChanged) {
		t.Fatalf("recovery error = %v, want changed", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "concurrent" {
		t.Fatalf("refused recovery changed target to %q", got)
	}
}

func TestRecoverSwapCleansExactResolvedWitnessResidue(t *testing.T) {
	root := t.TempDir()
	directory := filepath.Join(root, "documents")
	scratch := filepath.Join(root, "scratch")
	if err := os.Mkdir(directory, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Mkdir(scratch, 0o700); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(directory, "document")
	if err := os.WriteFile(path, []byte("after"), 0o600); err != nil {
		t.Fatal(err)
	}
	witness := filepath.Join(scratch, ".atomic-swap-witness-resolved")
	if err := os.WriteFile(witness, []byte("before"), 0o600); err != nil {
		t.Fatal(err)
	}

	resolved, err := RecoverSwap(path, scratch, []byte("before"), []byte("after"), 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if string(resolved) != "after" {
		t.Fatalf("resolved state = %q, want after", resolved)
	}
	entries, err := os.ReadDir(scratch)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("resolved scratch retains %d entries", len(entries))
	}
}
