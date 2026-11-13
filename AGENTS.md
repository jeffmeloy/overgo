# overgo

skill.md is the only written guidance; each of its rules names the check that holds it.

- Next step: `go run ./cmd/plan -next`
- Check it: `go run ./cmd/plan -verify`
- Commit it: `go run ./cmd/loop -land <item>/<step>`

When a command refuses or something breaks, find its cause in the code or by reproducing it before changing anything, and where else it occurs; cure it by removing what failed, else fixing what exists, adding last. Don't stop to ask.
