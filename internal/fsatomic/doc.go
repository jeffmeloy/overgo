// Package fsatomic owns the durable namespace primitives shared by staged-file
// publishers. Replace makes a same-volume move durable before returning;
// SyncFile persists metadata changes made through hard links; SyncDirectory
// persists the containing namespace where the operating system documents that
// operation. One owner keeps the platform split from being re-derived beside
// each store.
package fsatomic
