/* 离线标注工具的离线自检：用最小 DOM 桩跑通脚本，再验证解析/写回/R 规则。
   用法： node tools/dataset_batches/_selftest_offline_tool.js
   通过后可以删掉（它不是分发内容的一部分）。 */
const fs = require("fs");
const path = require("path");

const htmlPath = path.join(__dirname, "标注工具_离线版.html");
const html = fs.readFileSync(htmlPath, "utf8");
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if (!m) throw new Error("没找到 <script> 段");
const js = m[1];

const ctx = new Proxy({}, {
  get: (t, k) => (k === "measureText" ? () => ({ width: 12 })
    : k === "canvas" ? null : () => {}),
  set: () => true,
});
function makeEl() {
  return {
    style: {}, dataset: {}, children: [], textContent: "",
    classList: { toggle() {}, add() {}, remove() {} },
    appendChild(c) { this.children.push(c); },
    addEventListener() {}, onclick: null,
    getContext: () => ctx,
    getBoundingClientRect: () => ({ left: 0, top: 0 }),
    width: 0, height: 0, clientWidth: 1200, clientHeight: 800,
  };
}
const els = {};
global.window = { addEventListener() {}, devicePixelRatio: 1 };
global.document = {
  getElementById: (id) => (els[id] = els[id] || makeEl()),
  createElement: () => makeEl(),
};
global.Image = class { set src(v) { if (this.onload) this.onload(); } };
global.URL = { createObjectURL: () => "blob:x", revokeObjectURL() {} };
global.Blob = class { constructor(p) { this.parts = p; } };
global.setTimeout = () => 0;

const probe = `
globalThis.__t = { CLASSES, CLASS_CN, parseLabel, labelText, fromXYXY, boxXYXY,
                   applyRiderRule,
                   setItems: (v) => { items = v; idx = 0; sel = -1; undo = {}; },
                   getBoxes: () => items[0].boxes,
                   getDirty: () => items[0].dirty };
`;
// eslint-disable-next-line no-eval
eval(js + probe);
const t = globalThis.__t;

let failed = 0;
function check(name, cond, extra) {
  console.log((cond ? "PASS  " : "FAIL  ") + name + (cond ? "" : "   " + (extra || "")));
  if (!cond) failed++;
}

check("类别表 9 类且顺序未变",
  t.CLASSES.join(",") === "car,truck,bus,motorcycle,bicycle,person,sign,bollard,tricycle");

const parsed = t.parseLabel("3 0.467044 0.627552 0.098409 0.150847\r\n0 0.177549 0.553880 0.142422 0.073192\n\n");
check("解析 YOLO 标注两行", parsed.length === 2 && parsed[0][0] === 3, JSON.stringify(parsed));
check("类别号被取整", Number.isInteger(parsed[1][0]));

const txt = t.labelText({ boxes: parsed });
const back = t.parseLabel(txt);
check("写回格式可再解析且数值不漂移",
  back.length === 2 && Math.abs(back[0][3] - parsed[0][3]) < 1e-6,
  JSON.stringify(txt));
check("写回小数位为 6 位", /^3 0\.467044 0\.627552 0\.098409 0\.150847/.test(txt.split("\n")[0]), txt.split("\n")[0]);

const r = t.boxXYXY([0, 0.5, 0.5, 0.2, 0.1]);
check("中心宽高 → 角点", Math.abs(r[0] - 0.4) < 1e-9 && Math.abs(r[3] - 0.55) < 1e-9, JSON.stringify(r));
const b2 = t.fromXYXY(7, r);
check("角点 → 中心宽高 可逆", Math.abs(b2[1] - 0.5) < 1e-9 && Math.abs(b2[4] - 0.1) < 1e-9, JSON.stringify(b2));

// R 规则：骑车人（与摩托重叠的行人）应被删掉，旁边站着的行人保留
const boxes = [
  [3, 0.50, 0.60, 0.10, 0.20],   // 摩托
  [5, 0.50, 0.55, 0.06, 0.14],   // 骑它的人 → 该删
  [5, 0.10, 0.50, 0.05, 0.20],   // 路边行人 → 该留
  [0, 0.80, 0.70, 0.10, 0.10],   // 汽车
];
t.setItems([{ name: "a.jpg", boxes: boxes.map(b => b.slice()), dirty: false }]);
t.applyRiderRule();
const after = t.getBoxes();
check("R 规则只删骑乘中的行人（4 框 → 3 框）", after.length === 3, JSON.stringify(after));
check("路边行人（0.10 那个）被留下",
  after.some(b => b[0] === 5 && Math.abs(b[1] - 0.10) < 1e-9), JSON.stringify(after));
check("载具框不受影响",
  after.filter(b => b[0] === 3 || b[0] === 0).length === 2, JSON.stringify(after));

check("R 规则后这张被标记为已改动（保存时会写回）", t.getDirty() === true);

console.log(failed ? `\n${failed} 项未通过` : "\n全部通过");
process.exit(failed ? 1 : 0);
