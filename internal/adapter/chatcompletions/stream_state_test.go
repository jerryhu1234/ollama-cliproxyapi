package chatcompletions

import "testing"

func TestStreamSnapshotProgressIgnoresNoise(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	sc.Feed([]byte(":"))
	if sc.StreamSnapshot().Progress != 0 {
		t.Fatal(sc.StreamSnapshot())
	}
	sc.Feed([]byte(" ping\n\n"))
	if sc.StreamSnapshot().Progress != 0 {
		t.Fatal(sc.StreamSnapshot())
	}
	sc.Feed([]byte(`data: {"id":"r","choices":[{"delta":{"content":"x"}}]}` + "\n"))
	a := sc.StreamSnapshot()
	if !a.Started || a.Progress == 0 {
		t.Fatal(a)
	}
	sc.Feed([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n"))
	b := sc.StreamSnapshot()
	if !b.FinishSeen || b.TerminalSeen {
		t.Fatal(b)
	}
	sc.Feed([]byte("data: [DONE]\n"))
	c := sc.StreamSnapshot()
	if !c.DoneSeen || !c.TerminalSeen || c.TerminalKind != "completed" {
		t.Fatal(c)
	}
}

func TestChatUsageOnlyProgressChangesOnce(t *testing.T) {
	sc := NewStreamConverter("openai-response")
	role := []byte(`data: {"id":"r","choices":[{"delta":{"role":"assistant"}}]}` + "\n")
	sc.Feed(role)
	p := sc.StreamSnapshot().Progress
	sc.Feed(role)
	if sc.StreamSnapshot().Progress != p {
		t.Fatal("duplicate role advanced")
	}
	usage := []byte(`data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":2}}` + "\n")
	sc.Feed(usage)
	if sc.StreamSnapshot().Progress <= p {
		t.Fatal("usage did not advance")
	}
	p = sc.StreamSnapshot().Progress
	sc.Feed(usage)
	if sc.StreamSnapshot().Progress != p {
		t.Fatal("duplicate usage advanced")
	}
}
