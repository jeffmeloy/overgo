// Package fsatomic owns the durable-rename primitives shared by every
// staged-file publisher: a payload is flushed, renamed into place, and
// the receiving directory is synchronized where the platform supports
// it. One owner keeps the platform split from being re-derived beside
// each store.
package fsatomic
