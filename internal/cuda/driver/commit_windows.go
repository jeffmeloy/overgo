//go:build windows

package driver

import (
	"errors"
	"fmt"
)

var (
	virtualAlloc = kernel32.NewProc("VirtualAlloc")
	virtualFree  = kernel32.NewProc("VirtualFree")
)

const (
	memCommit     = 0x1000
	memReserve    = 0x2000
	memDecommit   = 0x4000
	memRelease    = 0x8000
	pageReadWrite = 0x04
	// commitPage is the page an ordinary commit is charged in.
	commitPage = 4 << 10
)

// commitReservation is an ordinary commit a library holds for its own allocations: Windows grows a system-managed page
// file at once for an ordinary commit past its limit, but not for the driver's requests, which fail at the limit and
// grow it after (tmp/flash-next-load/pagefile: 45 GiB on the card, then pins failed at 47 GiB pinned with the limit
// moved 102.8 -> 118.3 GiB; a 60 GiB ordinary commit grew it 107.0 -> 124.7 at once and 60 GiB of pins took its place
// without a failure). Each allocation decommits as much of it as it takes just before its call (handOver), so the
// charge never falls for the page file to shrink; given is its part decommitted so.
type commitReservation struct {
	base         uintptr
	bytes, given uint64
}

// ReserveCommit makes the commit reservation the library's allocations then take their commit from, of bytes.
func (l *Library) ReserveCommit(bytes uint64) error {
	bytes = (bytes + commitPage - 1) / commitPage * commitPage
	l.allocationMu.Lock()
	defer l.allocationMu.Unlock()
	if l.reservation.base != 0 {
		return errors.New("driver: the library holds a commit reservation already")
	}
	base, _, err := virtualAlloc.Call(0, uintptr(bytes), memCommit|memReserve, pageReadWrite)
	if base == 0 {
		commit, commitErr := MachineCommit()
		return errors.Join(fmt.Errorf("reserving %d bytes of commit, the machine at %s: VirtualAlloc: %w", bytes, commit, err), commitErr)
	}
	l.reservation = commitReservation{base: base, bytes: bytes}
	return nil
}

// handOver decommits as much of the reservation as an allocation of bytes takes, just before it takes it.
func (l *Library) handOver(bytes uint64) {
	l.allocationMu.Lock()
	defer l.allocationMu.Unlock()
	l.decommit(min((bytes+commitPage-1)/commitPage*commitPage, l.reservation.bytes-l.reservation.given))
}

// decommit gives back bytes of the reservation's part not yet given, allocationMu held.
func (l *Library) decommit(bytes uint64) {
	reservation := &l.reservation
	if reservation.base == 0 || bytes == 0 {
		return
	}
	if ok, _, _ := virtualFree.Call(reservation.base+uintptr(reservation.given), uintptr(bytes), memDecommit); ok != 0 {
		reservation.given += bytes
	}
}

// ReleaseCommit keeps keep bytes of the reservation for the allocations to come and gives back the rest; keeping none
// ends it.
func (l *Library) ReleaseCommit(keep uint64) error {
	l.allocationMu.Lock()
	defer l.allocationMu.Unlock()
	reservation := &l.reservation
	if reservation.base == 0 {
		return nil
	}
	if keep == 0 {
		ok, _, err := virtualFree.Call(reservation.base, 0, memRelease)
		*reservation = commitReservation{}
		if ok == 0 {
			return fmt.Errorf("releasing the commit reservation: VirtualFree: %w", err)
		}
		return nil
	}
	if held := reservation.bytes - reservation.given; held > keep {
		l.decommit((held - keep) / commitPage * commitPage)
	}
	return nil
}

// Commit is the machine's commit as the library's allocations meet it: what its reservation still holds is left to
// them.
func (l *Library) Commit() (Commit, error) {
	commit, err := MachineCommit()
	l.allocationMu.Lock()
	defer l.allocationMu.Unlock()
	commit.Left += l.reservation.bytes - l.reservation.given
	return commit, err
}
