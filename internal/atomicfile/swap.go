package atomicfile

import (
	"bytes"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"overgo/internal/checked"
	"overgo/internal/fsatomic"
)

// ErrChanged reports that CompareAndSwap did not replace the exact expected
// pathname state. A conflicting writer's bytes are never overwritten.
var ErrChanged = errors.New("atomic file changed")

const (
	swapNextPrefix             = ".atomic-swap-next-"
	swapWitnessPrefix          = ".atomic-swap-witness-"
	swapOldPrefix              = ".atomic-swap-old-"
	swapPreparingNextPrefix    = ".atomic-swap-preparing-next-"
	swapReservingWitnessPrefix = ".atomic-swap-reserving-witness-"
	swapReservingOldPrefix     = ".atomic-swap-reserving-old-"
)

type swapHooks struct {
	afterPrepareCreate      func(string)
	afterWitnessReservation func(string)
	afterOldReservation     func(string)
	beforeDetach            func(string)
	afterDetach             func(string)
	beforeInstall           func(string)
	afterInstall            func(string)
	afterRemove             func(string)
}

// CompareAndSwap atomically replaces the exact expected bytes and permission
// mode at path. Scratch must be an existing caller-owned directory on the same
// filesystem as path. It is a pathname CAS for atomic-replacement writers: the
// current entry is detached and validated before the replacement is installed
// with an O_EXCL hard link. Sanctioned in-place writers must additionally share
// a caller-owned lock because portable filesystems cannot revoke an already
// open writable handle.
//
// If a competing pathname replacement wins, its bytes remain at path. In the
// rare case that the displaced entry cannot be restored without overwriting a
// newer winner, the error names the quarantine file that preserves it.
//
// Windows requires a filesystem that supports hard links and permits reopening
// both exact states for metadata flush. Those capabilities are proven before
// detach, so unsupported filesystems or read-only states fail with path intact.
// A power loss after success may restore an exact scratch cleanup entry because
// Win32 has no documented write-through unlink; RecoverSwap validates and
// retires that harmless residue while the selected target remains durable.
func CompareAndSwap(path, scratch string, expected, replacement []byte, mode fs.FileMode) error {
	return compareAndSwap(path, scratch, expected, replacement, mode, swapHooks{})
}

func compareAndSwap(
	path, scratch string,
	expected, replacement []byte,
	mode fs.FileMode,
	hooks swapHooks,
) (err error) {
	directory := filepath.Dir(path)
	entries, err := os.ReadDir(scratch)
	if err != nil {
		return err
	}
	if len(entries) != 0 {
		return errors.New("atomic file: swap scratch is not empty")
	}
	preparing, err := prepareWithHook(
		scratch,
		swapPreparingNextPrefix+"*",
		replacement,
		mode,
		hooks.afterPrepareCreate,
	)
	if err != nil {
		return err
	}
	prepared, err := publishedSwapName(scratch, preparing, swapPreparingNextPrefix, swapNextPrefix)
	if err != nil {
		_ = os.Chmod(preparing, mode|fs.ModePerm)
		_ = os.Remove(preparing)
		return err
	}
	preserveEvidence := false
	defer func() {
		if !preserveEvidence {
			_ = os.Chmod(preparing, mode|fs.ModePerm)
			_ = os.Remove(preparing)
			_ = os.Chmod(prepared, mode|fs.ModePerm)
			_ = os.Remove(prepared)
		}
	}()
	if err := fsatomic.SyncFile(preparing); err != nil {
		return fmt.Errorf("atomic file: sync prepared replacement entry: %w", err)
	}
	if err := fsatomic.Replace(preparing, prepared); err != nil {
		return fmt.Errorf("atomic file: publish prepared replacement entry: %w", err)
	}
	preparing = ""
	witness, err := reservedSwapName(
		scratch,
		swapReservingWitnessPrefix,
		swapWitnessPrefix,
		hooks.afterWitnessReservation,
	)
	if err != nil {
		return err
	}
	defer func() {
		if !preserveEvidence {
			_ = os.Remove(witness)
		}
	}()
	quarantine, err := reservedSwapName(
		scratch,
		swapReservingOldPrefix,
		swapOldPrefix,
		hooks.afterOldReservation,
	)
	if err != nil {
		return err
	}
	defer func() {
		if !preserveEvidence {
			_ = os.Remove(quarantine)
		}
	}()
	if err := os.Link(path, witness); err != nil {
		return fmt.Errorf("atomic file: witness current entry: %w", err)
	}
	if err := exactFileState(witness, expected, mode); err != nil {
		return err
	}
	if err := fsatomic.SyncFile(witness); err != nil {
		return fmt.Errorf("atomic file: sync current-entry witness: %w", err)
	}
	if err := fsatomic.SyncDirectory(scratch); err != nil {
		return err
	}
	if hooks.beforeDetach != nil {
		hooks.beforeDetach(path)
	}
	if err := fsatomic.Replace(path, quarantine); err != nil {
		return fmt.Errorf("atomic file: detach current entry: %w", err)
	}
	preserveEvidence = true
	if hooks.afterDetach != nil {
		hooks.afterDetach(path)
	}
	if err := fsatomic.SyncDirectory(scratch); err != nil {
		return fmt.Errorf("atomic file: sync detached evidence: %w", err)
	}
	if err := fsatomic.SyncDirectory(directory); err != nil {
		return fmt.Errorf("atomic file: sync detached target: %w", err)
	}
	matched, err := sameFile(witness, quarantine)
	if err != nil {
		return err
	}
	if !matched || exactFileState(quarantine, expected, mode) != nil {
		if restoreErr := restoreDetached(quarantine, path); restoreErr != nil {
			return fmt.Errorf("%w; displaced bytes preserved at %s: %v", ErrChanged, quarantine, restoreErr)
		}
		return fmt.Errorf("%w; displaced bytes preserved at %s", ErrChanged, quarantine)
	}
	if hooks.beforeInstall != nil {
		hooks.beforeInstall(path)
	}
	if err := os.Link(prepared, path); err != nil {
		return fmt.Errorf("%w; expected bytes preserved at %s; install replacement: %v", ErrChanged, quarantine, err)
	}
	if err := fsatomic.SyncFile(path); err != nil {
		return fmt.Errorf("atomic file: sync installed replacement: %w", err)
	}
	if hooks.afterInstall != nil {
		hooks.afterInstall(path)
	}
	installed, statErr := sameFile(prepared, path)
	stateErr := exactFileState(path, replacement, mode)
	oldStateErr := exactFileState(quarantine, expected, mode)
	if statErr != nil || !installed || stateErr != nil || oldStateErr != nil {
		return fmt.Errorf("%w; expected bytes preserved at %s", ErrChanged, quarantine)
	}
	if err := fsatomic.SyncDirectory(directory); err != nil {
		return fmt.Errorf("atomic file: sync installed replacement: %w", err)
	}
	if err := os.Remove(quarantine); err != nil {
		return fmt.Errorf("atomic file: remove replaced entry %s: %w", quarantine, err)
	}
	quarantine = ""
	if err := fsatomic.SyncFile(witness); err != nil {
		return fmt.Errorf("atomic file: sync replaced-entry retirement: %w", err)
	}
	if hooks.afterRemove != nil {
		hooks.afterRemove(path)
	}
	if err := os.Remove(witness); err != nil {
		return fmt.Errorf("atomic file: remove resolved witness: %w", err)
	}
	witness = ""
	if err := os.Remove(prepared); err != nil {
		return fmt.Errorf("atomic file: remove resolved replacement: %w", err)
	}
	prepared = ""
	if err := fsatomic.SyncFile(path); err != nil {
		return fmt.Errorf("atomic file: sync replacement-evidence retirement: %w", err)
	}
	preserveEvidence = false
	return fsatomic.SyncDirectory(scratch)
}

// RecoverSwap resolves an interrupted CompareAndSwap from its exact private
// scratch evidence. A pre-install interruption restores expected; a completed
// install retains replacement. Recognized pre-phase preparation and reservation
// residue is non-authoritative and retired only after the target is validated.
// Missing, ambiguous, or tampered authoritative evidence is refused, and exact
// evidence is removed in a retry-safe order.
func RecoverSwap(
	path, scratch string,
	expected, replacement []byte,
	mode fs.FileMode,
) ([]byte, error) {
	entries, err := os.ReadDir(scratch)
	if err != nil {
		return nil, err
	}
	var next, witness, old, prephase []string
	for _, entry := range entries {
		name := entry.Name()
		fullPath := filepath.Join(scratch, name)
		switch {
		case strings.HasPrefix(name, swapNextPrefix):
			next = append(next, fullPath)
		case strings.HasPrefix(name, swapWitnessPrefix):
			witness = append(witness, fullPath)
		case strings.HasPrefix(name, swapOldPrefix):
			old = append(old, fullPath)
		case hasSwapNamePrefix(name, swapPreparingNextPrefix),
			hasSwapNamePrefix(name, swapReservingWitnessPrefix),
			hasSwapNamePrefix(name, swapReservingOldPrefix):
			info, err := os.Lstat(fullPath)
			if err != nil {
				return nil, err
			}
			if !info.Mode().IsRegular() {
				return nil, errors.New("atomic file: interrupted swap pre-phase residue is not a regular file")
			}
			prephase = append(prephase, fullPath)
		default:
			return nil, errors.New("atomic file: interrupted swap scratch contains an unknown entry")
		}
	}
	if len(next) > 1 || len(witness) > 1 || len(old) > 1 {
		return nil, errors.New("atomic file: interrupted swap scratch is ambiguous")
	}
	current, readErr := os.ReadFile(path)
	missing := errors.Is(readErr, fs.ErrNotExist)
	if readErr != nil && !missing {
		return nil, readErr
	}
	if len(next) == 0 {
		if missing {
			return nil, errors.New("atomic file: target is absent without interrupted swap evidence")
		}
		if err := requireEitherExactState(path, expected, replacement, mode); err != nil {
			return nil, err
		}
		if err := validateResolvedSwapResidue(witness, old, expected, mode); err != nil {
			return nil, err
		}
		if err := fsatomic.SyncFile(path); err != nil {
			return nil, err
		}
		if err := retireSwapEvidence(path, scratch, prephase, old, witness, nil); err != nil {
			return nil, err
		}
		return current, nil
	}
	if len(old) > len(witness) || len(witness) > len(next) {
		return nil, errors.New("atomic file: interrupted swap scratch is ambiguous")
	}
	if err := exactFileState(next[0], replacement, mode); err != nil {
		return nil, errors.New("atomic file: interrupted replacement evidence changed")
	}
	if len(witness) != 0 {
		if err := exactFileState(witness[0], expected, mode); err != nil {
			return nil, errors.New("atomic file: interrupted witness evidence changed")
		}
	}
	if len(old) != 0 {
		if err := exactFileState(old[0], expected, mode); err != nil {
			return nil, errors.New("atomic file: interrupted quarantine evidence changed")
		}
		matched, err := sameFile(witness[0], old[0])
		if err != nil || !matched {
			return nil, errors.New("atomic file: interrupted quarantine has no exact witness")
		}
	}
	if missing {
		if len(old) != 1 {
			return nil, errors.New("atomic file: absent target has no exact detached quarantine")
		}
		if err := os.Link(old[0], path); err != nil {
			if errors.Is(err, fs.ErrExist) {
				return nil, fmt.Errorf("%w; detached target was concurrently created", ErrChanged)
			}
			return nil, fmt.Errorf("atomic file: restore detached target: %w", err)
		}
		if err := fsatomic.SyncFile(path); err != nil {
			return nil, fmt.Errorf("atomic file: sync restored target: %w", err)
		}
		current = expected
	}
	before, beforeErr := false, error(nil)
	if len(witness) != 0 {
		before, beforeErr = sameFile(path, witness[0])
	}
	after, afterErr := sameFile(path, next[0])
	if len(witness) == 0 && !after && exactFileState(path, expected, mode) == nil {
		before = true
	}
	if beforeErr != nil || afterErr != nil || before == after {
		return nil, fmt.Errorf("%w; interrupted target differs from its exact swap states", ErrChanged)
	}
	if before {
		if err := exactFileState(path, expected, mode); err != nil {
			return nil, fmt.Errorf("%w; restored target changed", ErrChanged)
		}
		current = expected
	} else {
		if err := exactFileState(path, replacement, mode); err != nil {
			return nil, fmt.Errorf("%w; installed target changed", ErrChanged)
		}
		current = replacement
	}
	// Make the selected target link durable before retiring the only phase
	// evidence that can reconstruct an interrupted detached state.
	if err := fsatomic.SyncFile(path); err != nil {
		return nil, err
	}
	if err := fsatomic.SyncDirectory(filepath.Dir(path)); err != nil {
		return nil, err
	}
	if err := retireSwapEvidence(path, scratch, prephase, old, witness, next); err != nil {
		return nil, err
	}
	return current, nil
}

func validateResolvedSwapResidue(witness, old []string, expected []byte, mode fs.FileMode) error {
	if len(witness) != 0 {
		if err := exactFileState(witness[0], expected, mode); err != nil {
			return errors.New("atomic file: resolved witness residue changed")
		}
	}
	if len(old) != 0 {
		if err := exactFileState(old[0], expected, mode); err != nil {
			return errors.New("atomic file: resolved quarantine residue changed")
		}
	}
	if len(witness) != 0 && len(old) != 0 {
		matched, err := sameFile(witness[0], old[0])
		if err != nil || !matched {
			return errors.New("atomic file: resolved quarantine residue has no exact witness")
		}
	}
	return nil
}

func retireSwapEvidence(path, scratch string, prephase, old, witness, next []string) error {
	oldPath, _ := checked.First(old)
	witnessPath, _ := checked.First(witness)
	nextPath, _ := checked.First(next)
	if oldPath != "" {
		if err := os.Remove(oldPath); err != nil {
			return fmt.Errorf("atomic file: remove resolved quarantine: %w", err)
		}
		if witnessPath != "" {
			if err := fsatomic.SyncFile(witnessPath); err != nil {
				return fmt.Errorf("atomic file: sync quarantine retirement: %w", err)
			}
		}
	}
	if witnessPath != "" {
		if err := os.Remove(witnessPath); err != nil {
			return fmt.Errorf("atomic file: remove resolved witness: %w", err)
		}
	}
	if nextPath != "" {
		if err := os.Remove(nextPath); err != nil {
			return fmt.Errorf("atomic file: remove resolved replacement: %w", err)
		}
		if err := fsatomic.SyncFile(path); err != nil {
			return fmt.Errorf("atomic file: sync replacement retirement: %w", err)
		}
	}
	for _, residue := range prephase {
		if err := os.Remove(residue); err != nil {
			return fmt.Errorf("atomic file: remove resolved pre-phase residue: %w", err)
		}
	}
	return fsatomic.SyncDirectory(scratch)
}

func requireEitherExactState(path string, expected, replacement []byte, mode fs.FileMode) error {
	if err := exactFileState(path, expected, mode); err == nil {
		return nil
	}
	if err := exactFileState(path, replacement, mode); err == nil {
		return nil
	}
	return errors.New("atomic file: target differs from its exact swap states")
}

func publishedSwapName(directory, createdPath, createdPrefix, finalPrefix string) (string, error) {
	name := filepath.Base(createdPath)
	if !hasSwapNamePrefix(name, createdPrefix) {
		return "", errors.New("atomic file: invalid private swap preparation name")
	}
	return filepath.Join(directory, finalPrefix+strings.TrimPrefix(name, createdPrefix)), nil
}

func hasSwapNamePrefix(name, prefix string) bool {
	return strings.HasPrefix(name, prefix) && len(name) > len(prefix)
}

func reservedSwapName(
	directory, reservationPrefix, finalPrefix string,
	afterCreate func(string),
) (name string, err error) {
	file, err := os.CreateTemp(directory, reservationPrefix+"*")
	if err != nil {
		return "", err
	}
	reservation := file.Name()
	if afterCreate != nil {
		afterCreate(reservation)
	}
	defer func() {
		if err != nil {
			_ = file.Close()
			_ = os.Remove(reservation)
		}
	}()
	if err := file.Close(); err != nil {
		return "", err
	}
	if err := os.Remove(reservation); err != nil {
		return "", err
	}
	if err := fsatomic.SyncDirectory(directory); err != nil {
		return "", err
	}
	return publishedSwapName(directory, reservation, reservationPrefix, finalPrefix)
}

func exactFileState(path string, expected []byte, mode fs.FileMode) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return err
	}
	info, err := os.Lstat(path)
	if err != nil {
		return err
	}
	if !info.Mode().IsRegular() {
		return ErrChanged
	}
	modeChanged := runtime.GOOS != "windows" && info.Mode().Perm() != mode.Perm()
	if !bytes.Equal(data, expected) || modeChanged {
		return ErrChanged
	}
	return nil
}

func sameFile(left, right string) (bool, error) {
	leftInfo, err := os.Stat(left)
	if err != nil {
		return false, err
	}
	rightInfo, err := os.Stat(right)
	if err != nil {
		return false, err
	}
	return os.SameFile(leftInfo, rightInfo), nil
}

func restoreDetached(detached, path string) error {
	if err := os.Link(detached, path); err != nil {
		return err
	}
	return fsatomic.SyncFile(path)
}
