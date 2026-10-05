package apimanifest

// HeldOpenHeader is set by a server on a response that is a subscription a page holds open for as long as it
// listens (the workbench's activity stream). A swap proxy in front of the server does not wait for such a
// response to end when it changes the model, though it keeps the child resident while it lasts.
const HeldOpenHeader = "X-Overgo-Held-Open"
