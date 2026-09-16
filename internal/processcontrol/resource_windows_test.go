//go:build windows

package processcontrol

import "testing"

func TestResourceContention(t *testing.T) {
	name := sharedAdmissionName(t)
	holdSharedAdmission(t, sharedAdmissionProbe{Name: name, Exclusive: true})
	assertSharedAdmission(t, name, true, true)
}
