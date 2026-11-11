// Package envsetting reads the settings a process's environment holds, each through the function of its kind, so that
// reading one declares it. A result setting is one a capability's answer can depend on: the count of an execution
// identity (internal/inferencesurface) reads each name a package's code passes to Result, quoted as it is written, and
// the value of each joins the identity that package's answers are kept under. A neutral setting changes nothing a
// capability computes, and Neutral takes the reason where it is read. Code an identity counts reads the environment
// through no other function: a read of its own is refused, since nothing says whether an answer depends on it.
package envsetting

import "os"

// Result reads a setting a capability's answer can depend on. The name is written as a quoted literal where it is
// read, so the count can read it.
func Result(name string) string { return os.Getenv(name) }

// Neutral reads a setting that changes nothing a capability computes (a diagnostic, a bound passed on to a child, where
// overgo's data lies), because says why where it is read.
func Neutral(name, because string) string { return os.Getenv(name) }

// NeutralLookup is Neutral, saying too whether the setting is set at all.
func NeutralLookup(name, because string) (string, bool) { return os.LookupEnv(name) }
