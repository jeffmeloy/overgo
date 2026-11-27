package processcontrol

import (
	"errors"
	"fmt"
	"io"
	"os"

	"overgo/internal/processmeasure"
)

// CommitNeed is what a process about to take commit finds on the machine: the commit it needs, the machine's commit
// left and its limit (MachineCommit), and the commit overgo's own other processes hold, which they give back as their
// work ends. Flash-Next's open needs some 116 GiB of it; on 2026-11-24 its pins met the limit mid-load, the machine at
// 0.0 GiB left of 156.8 with 40.6 GiB held by the rest of the machine, and the shortfall was met as a pinning failure
// instead of being said (flashnext-memory/reserve).
type CommitNeed struct {
	Needed, Left, Limit, Own uint64
}

// ErrCommitShort is the sentinel of a commit the machine cannot give, overgo's own other processes' commit given back.
var ErrCommitShort = errors.New("processcontrol: the machine's commit is short")

// Decide is what the process does with the commit it needs: it goes on where the machine has it left; it waits, as a
// card turn waits, where overgo's own other processes hold the shortfall; and it fails at once, naming the commit it
// needs, the commit left and the limit, where the shortfall is more than they hold or the need more than the limit,
// as nothing overgo runs gives it.
func (n CommitNeed) Decide() (wait bool, err error) {
	switch {
	case n.Needed <= n.Left:
		return false, nil
	case n.Needed > n.Limit:
		return false, fmt.Errorf("%w: it needs %.1f GiB, more than the machine's %.1f GiB limit, with %.1f GiB left",
			ErrCommitShort, gibibytes(n.Needed), gibibytes(n.Limit), gibibytes(n.Left))
	case n.Needed-n.Left <= n.Own:
		return true, nil
	}
	return false, fmt.Errorf("%w: it needs %.1f GiB, and the machine has %.1f GiB left of its %.1f GiB limit, overgo's own other processes holding %.1f GiB of it",
		ErrCommitShort, gibibytes(n.Needed), gibibytes(n.Left), gibibytes(n.Limit), gibibytes(n.Own))
}

// ownMatters says whether the commit overgo's own other processes hold can change the decision: only a need past the
// commit left and within the limit waits on them, and reading them reads every turn's record.
func (n CommitNeed) ownMatters() bool { return n.Left < n.Needed && n.Needed <= n.Limit }

// gibibytes is bytes in GiB.
func gibibytes(bytes uint64) float64 { return float64(bytes) / (1 << 30) }

// TakeCommit takes the commit a process needs, needed bytes, for purpose through take, an ordinary commit that grows a
// system-managed page file as far as Windows grows it (driver.Library.ReserveCommit). Where take fails, the machine's
// own figures decide (Decide): while overgo's own other processes hold the shortfall, it waits for each of them to give
// its turn up, as a card turn waits, saying so on output, and takes again; where the shortfall is more than they hold, it
// fails at once naming the figures, before anything is pinned. A take that fails with the commit left is said with them.
func TakeCommit(needed uint64, purpose string, take func() error, output io.Writer) error {
	for {
		taken := take()
		if taken == nil {
			return nil
		}
		left, limit, err := MachineCommit()
		if err != nil {
			return errors.Join(taken, err)
		}
		need := CommitNeed{Needed: needed, Left: left, Limit: limit}
		var holders []turnRecord
		if need.ownMatters() {
			holders, need.Own = ownCommit()
		}
		wait, short := need.Decide()
		switch {
		case short != nil:
			return fmt.Errorf("%s: %w: %w", purpose, short, taken)
		case !wait:
			return fmt.Errorf("%s: taking %.1f GiB of commit failed with %.1f GiB left of the machine's %.1f GiB limit: %w",
				purpose, gibibytes(needed), gibibytes(left), gibibytes(limit), taken)
		}
		fmt.Fprintf(output, "%s waits for %.1f GiB of commit for %s, %.1f GiB left of the machine's %.1f GiB limit and %.1f GiB held by overgo's own other processes\n",
			processLabel(uint32(os.Getpid())), gibibytes(needed), purpose, gibibytes(left), gibibytes(limit), gibibytes(need.Own))
		for _, holder := range holders {
			release, err := TakeTurn(holder.Name, "commit for "+purpose, output)
			if err != nil {
				return errors.Join(taken, err)
			}
			if err := release(); err != nil {
				return errors.Join(taken, err)
			}
		}
	}
}

// ownCommit is the commit overgo's own other processes hold: the holders of the turns this process neither holds nor
// rides in its starter's place (heldTurnRecords, heldTurns), each process's own commit counted once
// (processmeasure.ProcessCommit). A holder whose turns or commit cannot be read holds none the wait could free.
func ownCommit() ([]turnRecord, uint64) {
	records, err := heldTurnRecords("")
	if err != nil {
		return nil, 0
	}
	riding := heldTurns()
	var holders []turnRecord
	counted := map[uint32]bool{}
	var own uint64
	for _, record := range records {
		if holder, rides := riding[record.Name]; rides && holder == record.Process {
			continue
		}
		holders = append(holders, record)
		if counted[record.Process] {
			continue
		}
		counted[record.Process] = true
		if commit, err := processmeasure.ProcessCommit(record.Process); err == nil {
			own += commit
		}
	}
	return holders, own
}
