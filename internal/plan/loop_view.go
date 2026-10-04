package plan

import (
	"context"
	"path/filepath"
	"strings"

	"overgo/internal/processcontrol"
)

// LoopView is what the workbench shows of a worktree's unattended loop:
// whether a live cmd/loop supervisor owns it, whether a stop holds it and
// why, the row it works next, and the row it last landed.
type LoopView struct {
	Running    bool
	Stopped    bool
	StopReason string
	Complete   bool
	Row        string
	RowTitle   string
	Landing    LoopLanding
}

// LoopLanding is the worktree's last landing: its commit, the row it
// landed, its subject and when it was committed.
type LoopLanding struct {
	Commit  string
	Row     string
	Subject string
	At      string
}

// workbenchWorker is the worker a stop recorded from the workbench names.
const workbenchWorker = "workbench"

// landingFieldSeparator separates the landing log line's commit, time,
// subject and body.
const landingFieldSeparator = "\x00"

// ReadLoopView reads the loop of the worktree at root. The next row is the
// plan's, shown even while a stop holds it.
func ReadLoopView(ctx context.Context, root string) (LoopView, error) {
	repository, head, err := resolveRevision(ctx, root, gitHeadRevision)
	if err != nil {
		return LoopView{}, err
	}
	running, err := processcontrol.CampaignRunning(repository)
	if err != nil {
		return LoopView{}, err
	}
	stop, err := ReadStop(ctx, repository, head, ExecutionUnattended)
	if err != nil {
		return LoopView{}, err
	}
	document, err := Load(filepath.Join(repository, filepath.FromSlash(Path)))
	if err != nil {
		return LoopView{}, err
	}
	dispatch := DispatchOf(document, "")
	view := LoopView{Running: running, Stopped: stop.Blocked, Complete: dispatch.Complete}
	if stop.Blocked {
		view.StopReason = stop.Problem
		if stop.Event != nil {
			view.StopReason = stop.Event.ReasonCode + stopReasonSeparator + " " + stop.Event.Detail
		}
	}
	if !dispatch.Complete {
		view.Row, view.RowTitle = dispatch.Item+stepReferenceSeparator+dispatch.Step, dispatch.StepTitle
	}
	log, err := gitCommand(ctx, repository, "log", "-1", "--first-parent", "--grep=^"+LandedRowTrailer+": ", "--format=%H%x00%cI%x00%s%x00%B")
	if err != nil {
		return LoopView{}, err
	}
	commit, rest, _ := strings.Cut(strings.TrimSpace(string(log)), landingFieldSeparator)
	at, rest, _ := strings.Cut(rest, landingFieldSeparator)
	subject, body, found := strings.Cut(rest, landingFieldSeparator)
	if !found {
		return view, nil
	}
	view.Landing = LoopLanding{Commit: commit, At: at, Subject: subject}
	for line := range strings.Lines(body) {
		if row, found := strings.CutPrefix(strings.TrimSpace(line), LandedRowTrailer+": "); found {
			view.Landing.Row = row
		}
	}
	return view, nil
}

// StopLoop records a user stop on the worktree at root from the workbench,
// scoped to every execution mode; recording it wakes a live supervisor to
// honour it. A worktree a stop already holds is left as it is.
func StopLoop(ctx context.Context, root, detail string) error {
	repository, head, err := resolveRevision(ctx, root, gitHeadRevision)
	if err != nil {
		return err
	}
	current, err := ReadStop(ctx, repository, head, ExecutionInteractive)
	if err != nil || current.State == stopActive {
		return err
	}
	role, err := AutomationRole("")
	if err != nil {
		return err
	}
	if current.Event != nil {
		role = current.Event.Lane
	}
	_, err = RecordStop(ctx, ControlEvent{
		Kind: ControlStop, Lane: role, Worker: workbenchWorker, Worktree: filepath.ToSlash(repository), Mode: ExecutionAll,
		ReasonCode: "user-stop", Detail: detail, CodeCommit: head, Previous: current.ID,
	})
	return err
}
