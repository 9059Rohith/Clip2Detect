import test from "node:test";
import assert from "node:assert/strict";
import handler from "./label.js";

function response() {
  return {
    code: 200,
    headers: {},
    setHeader(name, value) { this.headers[name] = value; return this; },
    status(value) { this.code = value; return this; },
    json(value) { this.body = value; return this; },
  };
}

test("health reports missing key without exposing a secret", async () => {
  const previous = process.env.OPENAI_API_KEY;
  delete process.env.OPENAI_API_KEY;
  const reply = response();
  await handler({ method: "GET" }, reply);
  assert.equal(reply.code, 200);
  assert.equal(reply.body.configured, false);
  assert.equal(reply.body.model, "gpt-5.6-luna");
  if (previous) process.env.OPENAI_API_KEY = previous;
});

test("live labeling validates input and normalizes structured output", async () => {
  const previous = process.env.OPENAI_API_KEY;
  const originalFetch = globalThis.fetch;
  process.env.OPENAI_API_KEY = "test-key-not-real";
  const invalid = response();
  await handler({ method: "POST", body: { frame: 33 }, headers: {} }, invalid);
  assert.equal(invalid.code, 400);
  let called = false;
  globalThis.fetch = async (url, options) => {
    called = true;
    assert.equal(url, "https://api.openai.com/v1/responses");
    assert.equal(options.headers.Authorization, "Bearer test-key-not-real");
    const payload = JSON.parse(options.body);
    assert.equal(payload.model, "gpt-5.6-luna");
    assert.equal(payload.input[0].content[1].image_url, "https://clip2detect.vercel.app/sample/raw/frame_000031.jpg");
    return { ok: true, json: async () => ({ output: [{ content: [{ type: "output_text", text: JSON.stringify({ objects: [{ class_name: "fruit", x: 10, y: 20, width: 40, height: 45 }] }) }] }] }) };
  };
  try {
    const reply = response();
    await handler({ method: "POST", body: { frame: 31 }, headers: { "x-forwarded-for": "test-browser" } }, reply);
    assert.equal(reply.code, 200);
    assert.equal(called, true);
    assert.deepEqual(reply.body.objects, [{ class: "fruit", xyxy: [10, 20, 50, 65] }]);
    assert.equal(JSON.stringify(reply.body).includes("test-key-not-real"), false);
  } finally {
    globalThis.fetch = originalFetch;
    if (previous) process.env.OPENAI_API_KEY = previous;
    else delete process.env.OPENAI_API_KEY;
  }
});
