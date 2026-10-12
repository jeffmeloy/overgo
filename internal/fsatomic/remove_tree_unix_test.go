//go:build !windows

package fsatomic

import "testing"

// removeTreeHere holds what RemoveTree does on one system alone. Off Windows
// a tree is removed where it stands, with no rename and no name left behind,
// so there is nothing more to hold than the test that calls this does.
func removeTreeHere(t *testing.T) { t.Helper() }
