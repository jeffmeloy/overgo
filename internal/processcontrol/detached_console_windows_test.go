//go:build windows

package processcontrol

import "testing"

// TestDetachedChildrenOpenNoConsoleWindow holds detached children to a hidden
// console: DETACHED_PROCESS left them without one, so every console program
// they started (go, git, test binaries) opened its own window on the desktop.
func TestDetachedChildrenOpenNoConsoleWindow(t *testing.T) {
	t.Parallel()
	flags := DetachedSysProcAttr().CreationFlags
	if flags&createNoWindow == 0 {
		t.Fatalf("creation flags %#x omit CREATE_NO_WINDOW", flags)
	}
	if flags&0x00000008 != 0 { // DETACHED_PROCESS
		t.Fatalf("creation flags %#x keep DETACHED_PROCESS", flags)
	}
}
