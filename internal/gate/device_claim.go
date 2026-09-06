package gate

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"overgo/internal/automationcheck"
	"overgo/internal/clioptions"
	"overgo/internal/jsonfile"
	"overgo/internal/runrecord"
)

const (
	// deviceClaimFile: beside the Git state lock in the common directory,
	// so every worktree of the repository sees the same claim.
	deviceClaimFile = "overgo-device-claim.json"
	// deviceClaimWait bounds how long a second claimant waits for the holder.
	deviceClaimWait = 2 * time.Hour
	// deviceClaimPoll: interval between holder checks while waiting, and the
	// heartbeat refresh interval of the holder.
	deviceClaimPoll = 5 * time.Second
)

// deviceClaim: the recorded holder of the physical device.
type deviceClaim struct {
	PID       int       `json:"pid"`
	Worktree  string    `json:"worktree"`
	PlanRef   string    `json:"plan_ref"`
	Check     string    `json:"check"`
	Started   time.Time `json:"started"`
	Heartbeat time.Time `json:"heartbeat"`
}

// deviceClaimPath: the claim file in the Git common directory.
func deviceClaimPath(repo string) (string, error) {
	lock, err := gateGitStateProcessLockPath(repo)
	if err != nil {
		return "", err
	}
	return filepath.Join(filepath.Dir(lock), deviceClaimFile), nil
}

// claimExpired: a holder is gone when its heartbeat is older than the stale
// window; a dead holder stops refreshing, so its claim expires the same way.
func claimExpired(claim deviceClaim, now time.Time) (bool, string) {
	if age := now.Sub(claim.Heartbeat); age > runrecord.DefaultHeartbeatStaleAfter {
		return true, fmt.Sprintf("holder pid %d heartbeat is %s stale", claim.PID, age.Truncate(time.Second))
	}
	return false, ""
}

// createDeviceClaim: exclusive create; false when a claim already exists.
func createDeviceClaim(path string, claim deviceClaim) (bool, error) {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, clioptions.OutputFileMode)
	if errors.Is(err, os.ErrExist) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if err := file.Close(); err != nil {
		return false, err
	}
	return true, jsonfile.Write(path, claim, clioptions.OutputFileMode)
}

// acquireDeviceClaim: takes the host-wide claim or waits for its holder; a
// stale holder's claim is removed with the reason audited; the
// returned release removes the claim and stops the heartbeat.
func acquireDeviceClaim(
	ctx context.Context, path string, claim deviceClaim, wait, poll time.Duration, audit func(string),
) (func() error, error) {
	waitingSince := time.Time{}
	for {
		now := time.Now()
		claim.Started, claim.Heartbeat = cmpTime(claim.Started, now), now
		created, err := createDeviceClaim(path, claim)
		if err != nil {
			return nil, fmt.Errorf("device claim: %w", err)
		}
		if created {
			if !waitingSince.IsZero() {
				audit(fmt.Sprintf("device claim: acquired after waiting %s", now.Sub(waitingSince).Truncate(time.Second)))
			}
			return startDeviceClaimHeartbeat(path, claim, poll), nil
		}
		var holder deviceClaim
		readable := jsonfile.DecodeStrict(path, &holder) == nil
		if !readable {
			// A claim mid-write is not readable and is waited on; one that
			// stays unreadable past the stale window is abandoned.
			info, statErr := os.Stat(path)
			if statErr == nil && now.Sub(info.ModTime()) > runrecord.DefaultHeartbeatStaleAfter {
				if err := os.Remove(path); err == nil {
					audit("device claim: removed an unreadable stale claim")
					continue
				}
			}
		}
		if expired, reason := claimExpired(holder, now); readable && expired {
			if err := os.Remove(path); err != nil && !errors.Is(err, os.ErrNotExist) {
				return nil, fmt.Errorf("device claim: release expired holder: %w", err)
			}
			audit("device claim: released expired holder: " + reason)
			continue
		}
		if waitingSince.IsZero() {
			waitingSince = now
			audit(fmt.Sprintf("device claim: held by pid %d worktree %s plan %s check %s since %s; waiting up to %s",
				holder.PID, holder.Worktree, holder.PlanRef, holder.Check, holder.Started.UTC().Format(time.RFC3339), wait))
		}
		if now.Sub(waitingSince) >= wait {
			return nil, fmt.Errorf("device claim: held by pid %d worktree %s plan %s for longer than %s", holder.PID, holder.Worktree, holder.PlanRef, wait)
		}
		select {
		case <-ctx.Done():
			return nil, fmt.Errorf("device claim: %w", ctx.Err())
		case <-time.After(poll):
		}
	}
}

// cmpTime: the first time unless zero.
func cmpTime(value, fallback time.Time) time.Time {
	if value.IsZero() {
		return fallback
	}
	return value
}

// startDeviceClaimHeartbeat: refreshes the holder heartbeat every poll; the
// returned release stops it and removes the claim only while this holder owns it.
func startDeviceClaimHeartbeat(path string, claim deviceClaim, poll time.Duration) func() error {
	stop, done := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(done)
		ticker := time.Tick(poll)
		for {
			select {
			case <-ticker:
				claim.Heartbeat = time.Now()
				_ = jsonfile.Write(path, claim, clioptions.OutputFileMode)
			case <-stop:
				return
			}
		}
	}()
	return func() error {
		close(stop)
		<-done
		var holder deviceClaim
		if err := jsonfile.DecodeStrict(path, &holder); err != nil {
			return fmt.Errorf("device claim: release: %w", err)
		}
		if holder.PID != claim.PID || holder.Worktree != claim.Worktree {
			return fmt.Errorf("device claim: release refused: claim moved to pid %d worktree %s", holder.PID, holder.Worktree)
		}
		return os.Remove(path)
	}
}

// withDeviceClaim: the exclusive-device check runs under the host-wide claim.
func (g *gateContext) withDeviceClaim(check automationcheck.Check) automationcheck.Check {
	run := check.Run
	check.Run = func(ctx context.Context, invocation automationcheck.Invocation) (bool, string, error) {
		path, err := deviceClaimPath(g.repo)
		if err != nil {
			return false, "", err
		}
		release, err := acquireDeviceClaim(ctx, path, deviceClaim{
			PID: os.Getpid(), Worktree: g.repo, PlanRef: g.planRef, Check: check.Descriptor.Name,
		}, deviceClaimWait, deviceClaimPoll, func(line string) { g.audit = append(g.audit, line) })
		if err != nil {
			return false, "", err
		}
		skipped, detail, err := run(ctx, invocation)
		return skipped, detail, errors.Join(err, release())
	}
	return check
}
