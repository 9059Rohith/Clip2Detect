// The browser sends only a sample frame number. The OpenAI key stays in Vercel.
const WIDTH = 640;
const HEIGHT = 360;
const MODEL = process.env.OPENAI_LABEL_MODEL || "gpt-5.6-luna";
const cache = new Map();
const calls = new Map();

const BOX_SCHEMA = {
  type: "json_schema",
  name: "clip2detect_boxes",
  strict: true,
  schema: {
    type: "object",
    properties: {
      objects: {
        type: "array",
        items: {
          type: "object",
          properties: {
            class_name: { type: "string", enum: ["fruit", "bomb"] },
            x: { type: "number" },
            y: { type: "number" },
            width: { type: "number" },
            height: { type: "number" },
          },
          required: ["class_name", "x", "y", "width", "height"],
          additionalProperties: false,
        },
      },
    },
    required: ["objects"],
    additionalProperties: false,
  },
};

function normalize(objects) {
  if (!Array.isArray(objects)) throw new Error("OpenAI response had no object list");
  return objects.slice(0, 20).flatMap((object) => {
    if (!object || !["fruit", "bomb"].includes(object.class_name)) return [];
    const numbers = [object.x, object.y, object.width, object.height].map(Number);
    if (numbers.some((value) => !Number.isFinite(value))) return [];
    const [x, y, width, height] = numbers;
    const xyxy = [
      Math.max(0, Math.min(WIDTH, x)),
      Math.max(0, Math.min(HEIGHT, y)),
      Math.max(0, Math.min(WIDTH, x + width)),
      Math.max(0, Math.min(HEIGHT, y + height)),
    ].map((value) => Math.round(value));
    if (xyxy[2] <= xyxy[0] || xyxy[3] <= xyxy[1]) return [];
    return [{ class: object.class_name, xyxy }];
  });
}

function outputText(response) {
  return (response.output || [])
    .flatMap((item) => item.content || [])
    .filter((item) => item.type === "output_text" && typeof item.text === "string")
    .map((item) => item.text)
    .join("");
}

function isRateLimited(ip) {
  const now = Date.now();
  const history = (calls.get(ip) || []).filter((time) => now - time < 60_000);
  if (history.length >= 3) return true;
  history.push(now);
  calls.set(ip, history);
  return false;
}

export default async function handler(request, response) {
  response.setHeader("Cache-Control", "no-store");
  if (request.method === "GET") {
    return response.status(200).json({ configured: Boolean(process.env.OPENAI_API_KEY), model: MODEL });
  }
  if (request.method !== "POST") {
    response.setHeader("Allow", "GET, POST");
    return response.status(405).json({ error: "Method not allowed" });
  }
  if (!process.env.OPENAI_API_KEY) {
    return response.status(503).json({ error: "OpenAI API key is not configured" });
  }
  if (Number(request.headers["content-length"] || 0) > 128) {
    return response.status(413).json({ error: "Request is too large" });
  }
  let body;
  try {
    body = typeof request.body === "string" ? JSON.parse(request.body || "{}") : request.body;
  } catch {
    return response.status(400).json({ error: "Invalid JSON body" });
  }
  const frame = body?.frame;
  if (!Number.isInteger(frame) || frame < 1 || frame > 32) {
    return response.status(400).json({ error: "Frame must be an integer from 1 to 32" });
  }
  if (cache.has(frame)) return response.status(200).json(cache.get(frame));
  const ip = String(request.headers["x-forwarded-for"] || request.socket?.remoteAddress || "unknown").split(",")[0].trim();
  if (isRateLimited(ip)) return response.status(429).json({ error: "Please wait before requesting another analysis" });

  const frameName = `frame_${String(frame).padStart(6, "0")}.jpg`;
  const imageUrl = `https://clip2detect.vercel.app/sample/raw/${frameName}`;
  const abort = new AbortController();
  const timer = setTimeout(() => abort.abort(), 50_000);
  try {
    const upstream = await fetch("https://api.openai.com/v1/responses", {
      method: "POST",
      signal: abort.signal,
      headers: {
        Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: MODEL,
        store: false,
        max_output_tokens: 1200,
        reasoning: { effort: "low" },
        input: [{
          role: "user",
          content: [
            { type: "input_text", text: "Find each visible fruit and bomb in this 640 by 360 gameplay image. Return tight pixel bounding boxes for the objects only; ignore text, HUD, and decorative elements. x and y are the top-left corner." },
            { type: "input_image", image_url: imageUrl, detail: "high" },
          ],
        }],
        text: { format: BOX_SCHEMA },
      }),
    });
    if (!upstream.ok) {
      return response.status(upstream.status === 429 ? 429 : 502).json({ error: "OpenAI labeling is temporarily unavailable" });
    }
    const result = await upstream.json();
    const text = outputText(result);
    if (!text) throw new Error("OpenAI response contained no output text");
    const objects = normalize(JSON.parse(text).objects);
    const payload = { frame, model: MODEL, source: "live-openai-api", objects };
    cache.set(frame, payload);
    return response.status(200).json(payload);
  } catch {
    return response.status(502).json({ error: "Could not complete image labeling" });
  } finally {
    clearTimeout(timer);
  }
}

export const config = { maxDuration: 60 };
