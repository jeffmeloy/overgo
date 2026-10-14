package gitauthority

// WriterArguments returns Git command arguments that harden every repository
// component written by an authority transaction. Command-line configuration
// intentionally overrides weaker repository and user configuration for this
// one child process.
func WriterArguments(arguments ...string) []string {
	configured := make([]string, 0, 5+len(arguments))
	configured = append(
		configured,
		"--no-replace-objects",
		"-c", "core.fsync=all",
		"-c", "core.fsyncMethod=fsync",
	)
	return append(configured, arguments...)
}
