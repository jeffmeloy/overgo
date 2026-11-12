//go:build windows

package driver

import (
	"testing"

	"overgo/internal/testcheck"
)

// TestACommitReservationIsTakenByTheAllocationsItHolds reserves commit with an ordinary commit, hands parts of it to
// allocations, keeps a part and gives back the rest: what it holds is counted as left to the library's allocations
// (Commit), each allocation takes its own bytes from it in whole pages, and a second reservation beside the first is
// refused.
func TestACommitReservationIsTakenByTheAllocationsItHolds(t *testing.T) {
	library := &Library{}
	const mib = 1 << 20
	testcheck.NoError(t, library.ReserveCommit(64*mib))
	t.Cleanup(func() { testcheck.NoError(t, library.ReleaseCommit(0)) })
	if err := library.ReserveCommit(mib); err == nil {
		t.Fatal("a second reservation beside the first was made")
	}
	held := func() uint64 { return library.reservation.bytes - library.reservation.given }
	if held() != 64*mib {
		t.Fatalf("the reservation holds %d bytes, want 64 MiB", held())
	}
	library.handOver(mib + 1)
	if want := uint64(64*mib - mib - commitPage); held() != want {
		t.Fatalf("an allocation of a MiB and a byte left %d bytes held, want %d: whole pages", held(), want)
	}
	machine := testcheck.Must(MachineCommit())(t)
	seen := testcheck.Must(library.Commit())(t)
	if seen.Left < held() || seen.Left > machine.Left+held()+64*mib {
		t.Fatalf("the library sees %d bytes of commit left beside the machine's %d and %d held", seen.Left, machine.Left, held())
	}
	testcheck.NoError(t, library.ReleaseCommit(16*mib))
	if held() != 16*mib {
		t.Fatalf("keeping 16 MiB held %d bytes", held())
	}
	library.handOver(32 * mib)
	if held() != 0 {
		t.Fatalf("an allocation past what is held left %d bytes held", held())
	}
	library.handOver(mib)
	testcheck.NoError(t, library.ReleaseCommit(0))
	if library.reservation != (commitReservation{}) {
		t.Fatal("releasing the reservation left it held")
	}
}
